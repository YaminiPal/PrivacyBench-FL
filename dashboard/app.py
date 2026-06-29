import os
import streamlit as pd_stream
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assert clean page configurations matching enterprise visualization standards
pd_stream.set_page_config(
    page_title="PrivacyBench-FL Diagnostics",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================
# 📊 DATA INGESTION ENGINE
# ========================================================
@pd_stream.cache_data
def load_benchmark_payload(file_path="experiments/full_system_results.npy"):
    """Reads metrics exported from main.py / run_full_system.py."""
    if not os.path.exists(file_path):
        return None
    return np.load(file_path, allow_pickle=True).item()

# Instantiate the structured data load pass
results = load_benchmark_payload()

if results is None:
    pd_stream.warning(
        "No experiment results found. Run `python main.py` first to generate "
        "`experiments/full_system_results.npy`."
    )
    pd_stream.stop()

pd_stream.sidebar.markdown("## Experiment Filters")
selected_models = pd_stream.sidebar.multiselect(
    "Target Model Architecture Matrix:",
    options=list(results.keys()),
    default=list(results.keys())
)

pd_stream.sidebar.markdown("---")
pd_stream.sidebar.markdown(
    """
    ### 🛡️ System Blueprint
    This panel isolates hyperparameter tracking across model spaces. It correlates **Differential Privacy** clipping profiles against model size metrics.
    """
)

# ========================================================
# 🏛️ HEADLINE ANALYTICS METRICS DASHBOARD
# ========================================================
pd_stream.title("🔒 PrivacyBench-FL: Enterprise Diagnostic Center")
pd_stream.markdown("Real-time distributed metrics reporting over the preprocessed Adult Income Dataset.")

if not selected_models:
    pd_stream.warning("Please select at least one architecture baseline inside the sidebar control panel.")
else:
    # Compile dynamic high-level operational statistics
    columns_layout = pd_stream.columns(len(selected_models))
    
    for idx, model_name in enumerate(selected_models):
        model_data = results[model_name]
        peak_accuracy = max(model_data["accuracy"]) * 100
        final_epsilon = model_data["epsilon"][-1]
        cumulative_wire_mb = model_data.get("total_traffic_mb", model_data.get("traffic", [0]))[-1]
        
        with columns_layout[idx]:
            pd_stream.metric(
                label=f"🏆 {model_name.upper()} Peak Testing Accuracy",
                value=f"{peak_accuracy:.2f} %",
                delta=f"Base Line Input Dim: 108"
            )
            pd_stream.metric(
                label=f"🔒 Spent Privacy Loss (ε)",
                value=f"{final_epsilon:.3f}",
                delta=f"Target δ = 1e-5",
                delta_color="inverse"
            )
            pd_stream.metric(
                label=f"📡 Cumulative Data Wire Overhead",
                value=f"{cumulative_wire_mb:.2f} MB",
                delta="8-Bit Uniform Compression Active"
            )

    pd_stream.markdown("---")

    # ========================================================
    # 📈 ACCURACY CONVERGENCE VS PRIVACY EXPEDITION PLOTS
    # ========================================================
    pd_stream.subheader("📈 Convergence Mechanics & Privacy Loss Slopes")
    
    tabs = pd_stream.tabs(["Predictive Accuracy Bounds", "Privacy Cost Epsilon (ε) Progressions"])
    
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for model_name in selected_models:
            ax.plot(
                results[model_name]["round"], 
                [acc * 100 for acc in results[model_name]["accuracy"]], 
                marker='o', label=model_name.upper(), linewidth=2
            )
        ax.set_xlabel("Global Communication Handshake Rounds", fontweight="bold")
        ax.set_ylabel("Holdout Test Accuracy Bounds (%)", fontweight="bold")
        ax.set_title("Unified Target Performance Curve Convergence Timeline", fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        pd_stream.pyplot(fig)
        plt.close()

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for model_name in selected_models:
            ax.plot(
                results[model_name]["round"], 
                results[model_name]["epsilon"], 
                marker='s', linestyle='--', label=f"{model_name.upper()} Privacy Spent", linewidth=2
            )
        ax.set_xlabel("Global Communication Handshake Rounds", fontweight="bold")
        ax.set_ylabel("Expended Cumulative Epsilon Value (ε)", fontweight="bold")
        ax.set_title("Privacy Accountancy Leakage Curves (Rényi Differential Privacy Map)", fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        pd_stream.pyplot(fig)
        plt.close()

    # ========================================================
    # 📡 WIRE CONGESTION ANALYSIS BLOCK
    # ========================================================
    pd_stream.markdown("---")
    bottom_col1, bottom_col2 = pd_stream.columns(2)

    with bottom_col1:
        pd_stream.subheader("📡 Infrastructure Bandwidth Consumption Metrics")
        fig, ax = plt.subplots(figsize=(6, 4))
        
        labels = [m.upper() for m in selected_models]
        traffic_values = [
            results[m].get("total_traffic_mb", results[m].get("traffic", [0]))[-1]
            for m in selected_models
        ]
        
        bars = ax.bar(labels, traffic_values, color=['#2ca02c', '#1f77b4'][:len(labels)], alpha=0.8, width=0.4)
        ax.set_ylabel("Total Encoded Bytes Exchanged (MB)", fontweight="bold")
        ax.set_title("Accumulated Network Payload Sizes Across Channels", fontweight="bold", pad=10)
        ax.grid(True, axis='y', linestyle="--", alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f} MB',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
                        
        pd_stream.pyplot(fig)
        plt.close()

    with bottom_col2:
        pd_stream.subheader("📋 Raw Data Telemetry Registry Ledger")
        pd_stream.markdown("Flattened structural historical run logs used to construct analytical dashboards:")
        
        flat_records = []
        for model_name in selected_models:
            m_data = results[model_name]
            for r_idx in range(len(m_data["round"])):
                flat_records.append({
                    "Architecture Blueprint": model_name.upper(),
                    "Comm Round": m_data["round"][r_idx],
                    "Test Accuracy (%)": round(m_data["accuracy"][r_idx] * 100, 2),
                    "Objective Loss": round(m_data["loss"][r_idx], 4),
                    "Epsilon Budget (ε)": round(m_data["epsilon"][r_idx], 3),
                    "Wire Footprint (MB)": round(
                        model_data.get("total_traffic_mb", model_data.get("traffic", [0]))[r_idx], 2
                    ),
                })
        
        # ... (Your existing code ending with the dataframe rendering) ...
    pd_stream.dataframe(pd.DataFrame(flat_records), use_container_width=True, hide_index=True)

    # ========================================================
    # 🧠 CONTINUAL LEARNING / DRIFT METRICS (PHASE 6)
    # ========================================================
    pd_stream.markdown("---")
    pd_stream.subheader("🧠 Continual Learning: Concept Drift & Forgetting Matrix")
    
    # Check if the metrics exist in your results object
    has_forgetting = any("forgetting_T1" in results[m] for m in selected_models)
    
    if has_forgetting:
        fig, ax = plt.subplots(figsize=(10, 4))
        for model_name in selected_models:
            m_data = results[model_name]
            rounds = m_data.get("round", [])
            # Visualize the T1 (initial state) vs T3 (drifted state) accuracy gap
            t1 = [v * 100 for v in m_data.get("forgetting_T1", [0]*len(rounds))]
            t3 = [v * 100 for v in m_data.get("forgetting_T3", [0]*len(rounds))]
            
            ax.plot(rounds, t1, marker="o", linestyle=":", label=f"{model_name.upper()} Original T1")
            ax.plot(rounds, t3, marker="s", label=f"{model_name.upper()} Drifted T3")
            
        ax.set_xlabel("Federated Rounds", fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontweight="bold")
        ax.set_title("System Memory Stability: Forgetting Curve under Drift", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        pd_stream.pyplot(fig)
        plt.close()
    else:
        pd_stream.info("ℹ️ Forgetting metrics not detected in this experiment trace. Ensure `temporal` drift injection was active.")