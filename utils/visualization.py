import pandas as pd
from utils.logger import logger
from continual_learning.forgetting_metrices import FederatedForgettingAnalyticEngine,FederatedVisualizer

# 1. Load your history log
history_df = pd.read_csv("experiments/training_history.csv")

# 2. Reconstruct the structure needed for your visualizer/analytic engine
# We convert the dataframe into a dict of lists or a grouped object
system_history_logs = {
    "mlp": history_df[history_df["model"] == "mlp"],
    "lr": history_df[history_df["model"] == "lr"]
}

# 3. Feed the Forgetting Analytic Engine
# Assuming the engine needs specific snapshots per round/model
forgetting_monitor = FederatedForgettingAnalyticEngine()
for _, row in history_df.iterrows():
    # Feeding the engine individual model snapshots from the CSV
    forgetting_monitor.record_eval_snapshot(
        model_name=row["model"],
        round_idx=row["round"],
        accuracy=row["accuracy"]
    )

# 4. Trigger charts
logger.info("Generating performance evaluation charts from training_history.csv...")

visualizer = FederatedVisualizer(output_dir="results/plots")

# Chart 1: Privacy vs Utility (using Epsilon and Accuracy)
visualizer.plot_privacy_utility_curve(system_history_logs["mlp"])

# Chart 2: Forgetting Matrices
visualizer.plot_forgetting_matrix(forgetting_monitor.performance_matrix)

# Chart 3: Comparative Dashboard
visualizer.plot_system_comparison_dashboard(system_history_logs)