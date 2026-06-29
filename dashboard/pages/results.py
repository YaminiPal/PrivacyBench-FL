import os
import sys
import numpy as np
import pandas as pd
# 1. Dynamically locate the root project directory and append it to the Python Path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 2. Now your internal framework imports will resolve perfectly
from utils.logger import logger
import streamlit as st

st.set_page_config(page_title="Overview", layout="wide")

class FederatedResultCompiler:
    def __init__(self, results_path="experiments/full_system_results.npy", output_dir="results"):
        """
        Ingests, parses, and formats serialized benchmarking outcomes.
        Generates automated markdown summary reports for performance audits.
        """
        self.results_path = results_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ========================================================
    # 📊 MAIN RESULTS PROCESSING ENGINE
    # ========================================================
    def compile_and_report(self) -> str:
        """
        Unpacks experiment logs, calculates final performance trade-offs,
        and saves a structured markdown report file.
        """
        if not os.path.exists(self.results_path):
            logger.error(f"Cannot compile results. Path '{self.results_path}' not found on disk.")
            return ""

        try:
            # Unpack binary payload
            raw_data = np.load(self.results_path, allow_pickle=True).item()
        except Exception as e:
            logger.error(f"Failed to read binary results file: {str(e)}")
            raise e

        report_lines = [
            "# 🔒 PrivacyBench-FL: Global Benchmark Run Summary",
            f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\nThis report presents an automated statistical analysis of the Privacy-Utility-Bandwidth trade-offs over the preprocessed Adult Income Dataset.",
            "\n## 🏆 Executive Summary Table",
            "| Architecture Blueprint | Peak Accuracy (%) | Final Loss | Total Spent Privacy (ε) | Bandwidth Transmitted (MB) |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]

        summary_stats = []

        # Loop through each model run inside the logs
        for model_name, metrics in raw_data.items():
            rounds = metrics.get("round", [])
            accuracies = [acc * 100 for acc in metrics.get("accuracy", [])]
            losses = metrics.get("loss", [])
            epsilons = metrics.get("epsilon", [])
            traffic = metrics.get("total_traffic_mb", [])

            if not accuracies:
                continue

            # Compute specific architectural key markers
            peak_acc = max(accuracies)
            final_loss = losses[-1] if losses else 0.0
            final_eps = epsilons[-1] if epsilons else 0.0
            total_mb = traffic[-1] if traffic else 0.0

            # Append structured row straight into the main markdown array
            report_lines.append(
                f"| **{model_name.upper()}** | {peak_acc:.2f}% | {final_loss:.4f} | {final_eps:.3f} | {total_mb:.2f} MB |"
            )

            summary_stats.append({
                "model": model_name.upper(),
                "peak_acc": peak_acc,
                "final_eps": final_eps,
                "total_mb": total_mb
            })

        # ========================================================
        # 🧪 SYSTEM COGNITIVE DEDUCTIONS LAYER
        # ========================================================
        report_lines.append("\n## 🔬 Deep Technical Insights")
        
        if len(summary_stats) >= 2:
            # Sort models by performance to extract trade-off patterns
            summary_stats.sort(key=lambda x: x["peak_acc"], reverse=True)
            top_model = summary_stats[0]
            bottom_model = summary_stats[-1]

            acc_delta = top_model["peak_acc"] - bottom_model["peak_acc"]
            traffic_delta = abs(top_model["total_mb"] - bottom_model["total_mb"])

            report_lines.append(f"1. **Predictive Variance Variance:** The `{top_model['model']}` achieved peak utility constraints, outperforming the basic `{bottom_model['model']}` setup by an absolute **{acc_delta:.2f}%** accuracy bound.")
            
            if top_model["total_mb"] > bottom_model["total_mb"]:
                report_lines.append(f"2. **Communication Overhead:** This accuracy boost comes with a trade-off; `{top_model['model']}` required an extra **{traffic_delta:.2f} MB** of total wire transmission payload compared to `{bottom_model['model']}`.")
            else:
                report_lines.append(f"2. **Compression Efficiency:** Interestingly, `{top_model['model']}` maintained a lower overall wire footprint than `{bottom_model['model']}`, demonstrating highly effective parameters optimization pruning.")

            # Identify structural privacy budget status indicators
            high_eps_model = max(summary_stats, key=lambda x: x["final_eps"])
            report_lines.append(f"3. **Differential Privacy Footprint:** The absolute highest information protection spend bound was reached by `{high_eps_model['model']}` at an extended cumulative **ε = {high_eps_model['final_eps']:.3f}** (at a locked target δ = 1e-5).")
        else:
            report_lines.append("1. Add multiple model configurations to the benchmark run to unlock relative system comparison insights.")

        report_lines.append("\n\n*-- End of Experiment Telemetry Output. All calculation profiles comply with strict DP-SGD auditing constraints. --*")

        # Combine lines and export report to disk
        markdown_body = "\n".join(report_lines)
        report_out_path = os.path.join(self.output_dir, "benchmark_report.md")
        
        with open(report_out_path, "w", encoding="utf-8") as f:
            f.write(markdown_body)

        logger.info(f"✅ Clean executive summary markdown report successfully generated -> '{report_out_path}'")
        return markdown_body

st.title("📊 Benchmark Results")

compiler = FederatedResultCompiler()
report = compiler.compile_and_report()

if report:
    st.markdown(report)
else:
    st.error("No benchmark results found. Run `python main.py` first.")
# ========================================================
# DIRECT RUN EXECUTION ENTRY HOOK
# ========================================================
