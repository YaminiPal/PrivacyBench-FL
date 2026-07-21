"""Generated-artifact lifecycle management for reproducible experiments."""
from pathlib import Path
import shutil

from utils.logger import logger


def clear_previous_run_outputs(config: dict) -> None:
    """Remove only generated experiment artifacts before a fresh run.

    Source data, client partitions, configuration, and committed documentation
    are never touched. This prevents stale charts/reports/checkpoints from being
    mistaken for the current experiment's results.
    """
    if not config.get("logging", {}).get("clear_previous_outputs", True):
        logger.info("Keeping previous generated outputs by configuration.")
        return

    generated_files = (
        Path("experiments/full_system_results.npy"),
        Path("experiments/training_history.csv"),
        Path("results/benchmark_report.md"),
    )
    for path in generated_files:
        if path.exists():
            path.unlink()
            logger.info(f"Removed previous generated output: {path}")

    checkpoint_dir = Path(config.get("logging", {}).get("checkpoint_dir", "checkpoints"))
    if checkpoint_dir.exists() and checkpoint_dir.is_dir():
        for item in checkpoint_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.info(f"Cleared previous checkpoints: {checkpoint_dir}")
