"""
PrivacyBench-FL: Full 8-Phase Federated Learning Pipeline
"""
import os
import yaml
import torch
import numpy as np

from utils.logger import logger
from utils.helpers import set_seed, verify_config_keys

from preprocessing.data_pipeline import prepare_federated_data

from models.logistic_regression import FederatedLogisticRegression
from models.mlp import FederatedMLP

from federated.server import UnifiedServer
from federated.client import FLClient
from federated.trainer import FLTrainer
from federated.client_selection import ClientSelector

from privacy.clipping import AdaptiveClipper
from privacy.quantization import QuantizationEngine

from evaluation.early_stopping import FederatedEarlyStopper
from evaluation.privacy_accounting import FederatedPrivacyAccountant
from evaluation.metrices import FederatedMetricsEngine
from evaluation.communication import CommunicationChannel
from evaluation.benchmarking import FLBenchmark

from continual_learning.drift_simulation import TabularDriftSimulator
from continual_learning.forgetting_metrices import FederatedForgettingAnalyticEngine

from dashboard.pages.results import FederatedResultCompiler


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_type, input_dim, num_classes, config):
    if model_type == "lr":
        return FederatedLogisticRegression(input_dim=input_dim, output_dim=num_classes)
    if model_type == "mlp":
        mlp_cfg = config.get("model", {})
        return FederatedMLP(
            input_dim=input_dim,
            hidden_dims=mlp_cfg.get("mlp_hidden_dims", [64, 32]),
            output_dim=num_classes,
            dropout_rate=mlp_cfg.get("dropout", 0.3),
            use_normalization=mlp_cfg.get("use_groupnorm", True),
            num_groups=mlp_cfg.get("groupnorm_groups", 4),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def run_model_experiment(model_type, config, data_bundle, device):
    input_dim = data_bundle["input_dim"]
    num_classes = config["num_classes"]

    server_model = build_model(model_type, input_dim, num_classes, config)
    server = UnifiedServer(model=server_model)

    clients = {}
    for cid in data_bundle["train_loaders"]:
        local_model = build_model(model_type, input_dim, num_classes, config)
        clients[cid] = FLClient(client_id=cid, model=local_model, device=device)

    dp_cfg = config.get("dp", {})
    quant_cfg = config.get("quantization", {})

    clipper = AdaptiveClipper(
        initial_target_clip=dp_cfg.get("clip_norm", 1.0)
    ) if dp_cfg.get("enabled") else None

    quantizer = QuantizationEngine(bits=quant_cfg.get("bits", 8)) if quant_cfg.get("enabled") else None
    comm_channel = CommunicationChannel(quantizer=quantizer) if quantizer else None

    selector = ClientSelector(
        fraction=config.get("client_fraction", 1.0),
        strategy=config.get("selection_strategy", "privacy_aware"),
        seed=config.get("seed", 42),
    )

    es_cfg = config.get("early_stopping", {})
    early_stopper = FederatedEarlyStopper(
        patience=es_cfg.get("patience", 5),
        min_delta=es_cfg.get("min_delta", 1e-4),
        target_accuracy=es_cfg.get("target_accuracy", 0.85),
        max_epsilon=dp_cfg.get("max_epsilon", 8.0),
    )

    privacy_accountant = FederatedPrivacyAccountant(
        target_delta=dp_cfg.get("delta", 1e-5)
    )

    drift_cfg = config.get("drift", {})
    drift_engine = TabularDriftSimulator(
        mode=drift_cfg.get("mode", "covariate_shift"),
        severity=drift_cfg.get("severity", 0.35),
        seed=config.get("seed", 42),
    ) if drift_cfg.get("enabled") else None

    forgetting_engine = FederatedForgettingAnalyticEngine()

    trainer = FLTrainer(
        server=server,
        clients=clients,
        train_loaders=data_bundle["train_loaders"],
        val_loaders=data_bundle["val_loaders"],
        metrics_engine=FederatedMetricsEngine(),
        early_stopper=early_stopper,
        privacy_accountant=privacy_accountant,
        client_selector=selector,
        clipper=clipper,
        quantizer=quantizer,
        comm_channel=comm_channel,
        dp_config=dp_cfg,
        forgetting_engine=forgetting_engine,
        temporal_loaders=data_bundle.get("temporal_loaders"),
        device=device,
    )

    history = trainer.train(
        rounds=config["rounds"],
        local_epochs=config["local_epochs"],
        lr=config["lr"],
        noise_multiplier=dp_cfg.get("noise_multiplier", 1.2),
        lambda_reg=config.get("lambda_reg", 1e-4),
        drift_config=drift_cfg if drift_cfg.get("enabled") else None,
        drift_engine=drift_engine,
        batch_size=config.get("training", {}).get("batch_size", 32),
    )

    benchmark = FLBenchmark(trainer=trainer, server=server, clients=clients)
    benchmark.results[model_type] = {"history": history, "time": 0.0}
    logger.info(f"Benchmark final accuracy ({model_type}): {history['accuracy'][-1]:.4f}")

    return history


def main():
    config = load_config("config.yaml")
    verify_config_keys(config, ["seed", "rounds", "dp", "num_classes", "data"])

    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu"
    logger.info(f"Device: {device}")

    # Phases 1-3: schema, split, local preprocessing
    data_bundle = prepare_federated_data(
        config,
        force_repartition=config.get("data", {}).get("force_repartition", False),
    )
    config["input_dim"] = data_bundle["input_dim"]

    # Phase 4-7: train LR and MLP, evaluate
    model_types = config.get("experiment", {}).get("compare_models", ["lr", "mlp"])
    results = {}

    for model_type in model_types:
        logger.info(f"\n{'='*60}\nPhase 4-7: Training {model_type.upper()}\n{'='*60}")
        results[model_type] = run_model_experiment(model_type, config, data_bundle, device)

    # Phase 8: save results for dashboard
    os.makedirs("experiments", exist_ok=True)
    np.save("experiments/full_system_results.npy", results)

    import pandas as pd
    flat_rows = []
    for model_name, hist in results.items():
        for i, rnd in enumerate(hist.get("round", [])):
            flat_rows.append({
                "model": model_name,
                "round": rnd,
                "accuracy": hist["accuracy"][i],
                "loss": hist["loss"][i],
                "epsilon": hist["epsilon"][i],
                "traffic_mb": hist["total_traffic_mb"][i],
            })
    pd.DataFrame(flat_rows).to_csv("experiments/training_history.csv", index=False)

    compiler = FederatedResultCompiler(
        results_path="experiments/full_system_results.npy",
        output_dir="results",
    )
    compiler.compile_and_report()

    logger.info("Pipeline complete. Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
