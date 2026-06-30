"""Shared experiment helpers — all scripts delegate to the main pipeline."""
import os
import sys
import copy

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from main import load_config, run_model_experiment
from preprocessing.data_pipeline import prepare_federated_data
from utils.helpers import set_seed


def run_with_overrides(overrides: dict) -> dict:
    """Run pipeline with config overrides (used by baseline/dp/ablation scripts)."""
    config = load_config("config.yaml")
    config = copy.deepcopy(config)

    def deep_merge(base, patch):
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    deep_merge(config, overrides)
    set_seed(config["seed"])

    import torch
    device = "cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu"

    data_bundle = prepare_federated_data(
        config,
        force_repartition=config.get("data", {}).get("force_repartition", False),
    )
    config["input_dim"] = data_bundle["input_dim"]

    model_types = overrides.get("experiment", {}).get(
        "compare_models", config.get("experiment", {}).get("compare_models", ["lr"])
    )

    results = {}
    for model_type in model_types:
        results[model_type] = run_model_experiment(model_type, config, data_bundle, device)
    return results
