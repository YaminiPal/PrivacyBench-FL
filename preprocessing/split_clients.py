import numpy as np
import pandas as pd
import os
import sys

# 1. Dynamically locate the root directory (PrivacyBench-FL) and inject it into Python's search path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.logger import logger

class ClientSplitter:
    def __init__(self, num_clients=3, seed=42):
        self.num_clients = num_clients
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def split(self, df: pd.DataFrame, mode="iid", target_col="income", alpha=0.5):
        if mode == "iid":
            return self._iid_split(df)
        elif mode == "non_iid":
            return self._non_iid_split(df, target_col, alpha)
        elif mode == "quantity_skew":
            return self._quantity_skew_split(df)
        elif mode == "temporal":
            return self._temporal_split(df, target_col)
        else:
            raise ValueError(f"Invalid split mode: {mode}")

    def _iid_split(self, df):
        shuffled_df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        chunks = np.array_split(shuffled_df, self.num_clients)
        return {
            f"client_{i}": pd.DataFrame(chunk).reset_index(drop=True)
            for i, chunk in enumerate(chunks)
        }

    def _non_iid_split(self, df, target_col, alpha=0.5):
        classes = df[target_col].unique()
        client_dict = {f"client_{i}": [] for i in range(self.num_clients)}

        for c in classes:
            class_df = df[df[target_col] == c].copy()
            class_df = class_df.sample(frac=1, random_state=self.seed)

            proportions = self.rng.dirichlet(np.repeat(alpha, self.num_clients))
            counts = (proportions * len(class_df)).astype(int)
            counts[-1] += len(class_df) - counts.sum()

            start = 0
            for i in range(self.num_clients):
                end = start + counts[i]
                if end > start:
                    client_dict[f"client_{i}"].append(class_df.iloc[start:end])
                start = end

        for k in client_dict:
            if client_dict[k]:
                client_dict[k] = (
                    pd.concat(client_dict[k])
                    .sample(frac=1, random_state=self.seed)
                    .reset_index(drop=True)
                )
            else:
                client_dict[k] = pd.DataFrame(columns=df.columns)

        return client_dict

    def _quantity_skew_split(self, df):
        shuffled_df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        proportions = self.rng.dirichlet(np.ones(self.num_clients))
        sizes = (proportions * len(shuffled_df)).astype(int)
        sizes[-1] += len(shuffled_df) - sizes.sum()

        client_data = {}
        start = 0
        for i, size in enumerate(sizes):
            chunk = shuffled_df.iloc[start:start + size]
            client_data[f"client_{i}"] = pd.DataFrame(chunk).reset_index(drop=True)
            start += size

        return client_data

    # ==========================================
    # 🛠️ FIXED TEMPORAL SPLIT (FLAT DICTIONARY OUTPUT)
    # ==========================================
    def _temporal_split(self, df, target_col):
        """
        Simulates genuine chronological time-progression blocks inside a flat 
        organizational matrix layout to prevent downstream file-reading errors.
        """
        total_len = len(df)
        t1_end = int(total_len * 0.4)
        t2_end = int(total_len * 0.7)

        t1_global = df.iloc[:t1_end].copy()
        t2_global = df.iloc[t1_end:t2_end].copy()
        t3_global = df.iloc[t2_end:].copy()

        # Inject sequential macro-environmental drifts forward through time
        t2_global = self._inject_drift(t2_global, target_col, shift_pct=0.15, feature_multiplier=1.10)
        t3_global = self._inject_drift(t3_global, target_col, shift_pct=0.30, feature_multiplier=1.25)

        # Temporary sub-structures to hold sharded time horizons
        # Temporary sub-structures to hold sharded time horizons
        client_shards = {f"client_{i}": [] for i in range(self.num_clients)}

        def distribute_and_append_block(block_df):
            shuffled_block = block_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            chunks = np.array_split(shuffled_block, self.num_clients)
            for idx in range(self.num_clients):
                chunk_df = chunks[idx]
                if isinstance(chunk_df, pd.DataFrame):
                    clean_df_shard = chunk_df.reset_index(drop=True)
                else:
                    clean_df_shard = pd.DataFrame(chunk_df, columns=df.columns).reset_index(drop=True)
                client_shards[f"client_{idx}"].append(clean_df_shard)

        distribute_and_append_block(t1_global)
        distribute_and_append_block(t2_global)
        distribute_and_append_block(t3_global)

        # 🔥 CONSOLIDATION STEP: Merge timeline slices cleanly back into flat dataframes
        output = {}
        for i in range(self.num_clients):
            output[f"client_{i}"] = pd.concat(client_shards[f"client_{i}"], ignore_index=True)

        return output

    def _inject_drift(self, df, target_col, shift_pct=0.2, feature_multiplier=1.15):
        df = df.copy()
        if len(df) == 0:
            return df

        mask = self.rng.random(len(df)) < shift_pct
        if mask.sum() > 0:
            # Map clean inverse target transformations across standard string data frames
            if df[target_col].dtype in [np.int64, np.int32]:
                df.loc[mask, target_col] = 1 - df.loc[mask, target_col]
            else:
                # Fallback support for string binary indicators like '<=50K' / '>50K'
                df.loc[mask, target_col] = df.loc[mask, target_col].apply(
                    lambda x: " >50K" if "<=" in str(x) else " <=50K"
                )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        for col in numeric_cols:
            df[col] = df[col] * feature_multiplier

        return df

    def save_clients(self, client_shards: dict, output_directory: str = "data/clients/"):
        """Persist partitioned client shards to disk."""
        os.makedirs(output_directory, exist_ok=True)
        for item in os.listdir(output_directory):
            item_path = os.path.join(output_directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

        for client_name, client_df in client_shards.items():
            out_path = os.path.join(output_directory, f"{client_name}.csv")
            client_df.to_csv(out_path, index=False)
            logger.info(f"Saved client shard -> '{out_path}' ({len(client_df)} samples)")

if __name__ == "__main__":
    raw_data_path = "data/raw/adult.csv"
    output_directory = "data/clients/"
    
    logger.info("🎬 Initializing Raw Preprocessing Data Partitioning Phase...")
    
    if not os.path.exists(raw_data_path):
        logger.error(f"❌ Master dataset file missing at '{raw_data_path}'.")
        sys.exit(1)
    
    from preprocessing.schema_check import validate_schema
    master_df, target_column, _ = validate_schema(dataset_path=raw_data_path)
    logger.info(f"Loaded and validated master file with shape: {master_df.shape}")

    # Change strategy here parameter dynamically to test splits: 'iid', 'non_iid', 'quantity_skew', 'temporal'
    active_strategy = "temporal"

    splitter = ClientSplitter(num_clients=3, seed=42)
    partitioned_clients = splitter.split(master_df, mode=active_strategy, target_col=target_column)
    
    splitter.save_clients(partitioned_clients, output_directory)
    logger.info("Success! Raw client partitions stored without leakage features.")