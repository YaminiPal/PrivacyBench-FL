import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.logger import logger
from preprocessing.schema_check import validate_schema
from preprocessing.split_clients import ClientSplitter
from preprocessing.local_preprocess import LocalPreprocessor


def _load_schema(schema_path: str = "dataset_schema.json") -> dict:
    with open(schema_path, "r") as f:
        return json.load(f)


def _encode_temporal_chunks(df: pd.DataFrame, target_col: str) -> dict:
    """Split a client's temporal data into T1 (old), T2, T3 (latest) chunks."""
    n = len(df)
    t1_end = int(n * 0.4)
    t2_end = int(n * 0.7)
    return {
        "T1": df.iloc[:t1_end],
        "T2": df.iloc[t1_end:t2_end],
        "T3": df.iloc[t2_end:],
    }


def _build_global_categories(df: pd.DataFrame, cat_cols: list) -> list:
    """Fixed categorical vocab from schema columns — no target leakage."""
    categories = []
    for col in cat_cols:
        vals = df[col].astype(str).str.strip().replace("?", "Unknown").unique().tolist()
        categories.append(sorted(vals))
    return categories


def _preprocess_client_shard(
    df: pd.DataFrame,
    schema: dict,
    target_col: str,
    global_categories: list = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    client_dir: str = None,
):
    """Phase 3: local train/val split and one-time preprocessing per client."""
    num_cols = [c for c in schema.get("num_features", []) if c in df.columns]
    cat_cols = [c for c in schema.get("cat_features", []) if c in df.columns]
    leakage = schema.get("leakage_features", [])
    num_cols = [c for c in num_cols if c not in leakage]
    cat_cols = [c for c in cat_cols if c not in leakage]

    train_df, val_df = train_test_split(
        df, test_size=val_fraction, random_state=seed,
        stratify=df[target_col] if df[target_col].nunique() > 1 else None,
    )

    preprocessor = LocalPreprocessor(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        target_col=target_col,
        categories=global_categories,
    )
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)

    if client_dir:
        preprocessor.save_state(client_dir)

    return X_train, y_train, X_val, y_val, preprocessor


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    if len(X) == 0:
        return None
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_federated_data(
    config: dict,
    force_repartition: bool = False,
) -> dict:
    """
    Full Phases 1-3: schema check, client split, local preprocessing.
    Returns loaders, schema, input_dim, and temporal eval loaders when applicable.
    """
    raw_path = config.get("data", {}).get("raw_path", "data/raw/adult.csv")
    client_dir = config.get("data", {}).get("client_dir", "data/clients")
    split_mode = config.get("data", {}).get("split_mode", "temporal")
    num_clients = config.get("data", {}).get("num_clients", 3)
    batch_size = config.get("training", {}).get("batch_size", 32)
    seed = config.get("seed", 42)

    # Phase 1: Schema check
    logger.info("PHASE 1: Dataset ingestion and schema check")
    clean_df, target_col, schema = validate_schema(dataset_path=raw_path)

    # Phase 2: Client partitioning
    client_files = sorted(
        [f for f in os.listdir(client_dir) if f.endswith(".csv")]
    ) if os.path.exists(client_dir) else []

    if force_repartition or not client_files:
        logger.info(f"PHASE 2: Partitioning data ({split_mode}, {num_clients} clients)")
        splitter = ClientSplitter(num_clients=num_clients, seed=seed)
        client_shards = splitter.split(clean_df, mode=split_mode, target_col=target_col)
        splitter.save_clients(client_shards, client_dir)
    else:
        logger.info(f"PHASE 2: Loading existing client shards from {client_dir}")
        client_shards = {}
        for fname in client_files:
            cid = fname.replace(".csv", "")
            client_shards[cid] = pd.read_csv(os.path.join(client_dir, fname))

    # Build fixed categorical vocabulary from schema (no numeric stats leakage)
    num_cols = [c for c in schema.get("num_features", []) if c in clean_df.columns]
    cat_cols = [c for c in schema.get("cat_features", []) if c in clean_df.columns]
    leakage = schema.get("leakage_features", [])
    num_cols = [c for c in num_cols if c not in leakage]
    cat_cols = [c for c in cat_cols if c not in leakage]
    global_categories = _build_global_categories(clean_df, cat_cols)

    # Phase 3: Local preprocessing (one-time per client)
    logger.info("PHASE 3: Local client preprocessing")
    train_loaders = {}
    val_loaders = {}
    temporal_loaders = {}
    preprocessors = {}

    for cid, shard_df in client_shards.items():
        if "income" not in shard_df.columns and len(shard_df.columns) == 15:
            shard_df.columns = [
                "age", "workclass", "fnlwgt", "education",
                "education-num", "marital-status", "occupation",
                "relationship", "race", "sex",
                "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income",
            ]
        shard_df = shard_df.copy()
        if target_col in shard_df.columns:
            shard_df[target_col] = shard_df[target_col].astype(str).str.strip()

        shard_df = shard_df.drop(
            columns=[c for c in schema.get("leakage_features", []) if c in shard_df.columns],
            errors="ignore",
        )

        client_preprocess_dir = os.path.join(client_dir, cid)
        X_tr, y_tr, X_val, y_val, prep = _preprocess_client_shard(
            shard_df, schema, target_col,
            global_categories=global_categories,
            val_fraction=0.2, seed=seed, client_dir=client_preprocess_dir,
        )
        preprocessors[cid] = prep
        train_loaders[cid] = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
        val_loaders[cid] = _make_loader(X_val, y_val, batch_size, shuffle=False)

        if split_mode == "temporal":
            chunks = _encode_temporal_chunks(shard_df, target_col)
            temporal_loaders[cid] = {}
            for period, chunk_df in chunks.items():
                if len(chunk_df) == 0:
                    continue
                X_chunk, y_chunk = prep.transform(chunk_df)
                temporal_loaders[cid][period] = _make_loader(
                    X_chunk, y_chunk, batch_size, shuffle=False
                )

    dims = {cid: train_loaders[cid].dataset[0][0].shape[0] for cid in train_loaders}
    if len(set(dims.values())) > 1:
        raise ValueError(f"Inconsistent feature dims across clients: {dims}")
    input_dim = next(iter(dims.values()))
    logger.info(f"Processed feature dimension: {input_dim}")

    return {
        "train_loaders": train_loaders,
        "val_loaders": val_loaders,
        "temporal_loaders": temporal_loaders,
        "preprocessors": preprocessors,
        "schema": schema,
        "target_col": target_col,
        "input_dim": input_dim,
        "client_shards": client_shards,
    }
