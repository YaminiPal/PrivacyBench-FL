import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Training Monitor", layout="wide")

st.title("🚀 Federated Learning Training Monitor")

RESULTS_CSV = "experiments/training_history.csv"


def load_history(path: str):
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load training history: {e}")
        return None


history = load_history(RESULTS_CSV)

if history is None:
    st.warning(
        "No training history found.\n\n"
        "Run an experiment first so that "
        "`experiments/training_history.csv` is generated."
    )
    st.stop()

required_cols = ["round", "accuracy", "loss"]

missing = [c for c in required_cols if c not in history.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# -------------------------------------------------
# KPIs
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

latest = history.iloc[-1]

col1.metric(
    "Current Round",
    int(latest["round"])
)

col2.metric(
    "Accuracy",
    f"{latest['accuracy']:.4f}"
)

col3.metric(
    "Loss",
    f"{latest['loss']:.4f}"
)

# -------------------------------------------------
# Accuracy Curve
# -------------------------------------------------

st.subheader("Accuracy vs Communication Round")

fig, ax = plt.subplots(figsize=(8, 4))

for model_name, group in history.groupby("model"):
    group = group.sort_values("round")
    ax.plot(
        group["round"],
        group["accuracy"],
        marker="o",
        linewidth=2,
        label=model_name.upper(),
    )

ax.set_xlabel("Round")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Communication Round")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Loss Curve
# -------------------------------------------------

st.subheader("Loss vs Communication Round")

fig, ax = plt.subplots(figsize=(8, 4))

for model_name, group in history.groupby("model"):
    group = group.sort_values("round")
    ax.plot(
        group["round"],
        group["loss"],
        marker="o",
        linewidth=2,
        label=model_name.upper(),
    )

ax.set_xlabel("Round")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Privacy Budget
# -------------------------------------------------

if "epsilon" in history.columns:
    st.subheader("Privacy Budget (ε)")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        history["round"],
        history["epsilon"],
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("ε")
    ax.grid(True)

    st.pyplot(fig)

# -------------------------------------------------
# Communication Cost
# -------------------------------------------------

traffic_column = None

if "total_traffic_mb" in history.columns:
    traffic_column = "total_traffic_mb"
elif "traffic" in history.columns:
    traffic_column = "traffic"

if traffic_column is not None:
    st.subheader("Communication Cost")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        history["round"],
        history[traffic_column],
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Traffic (MB)")
    ax.grid(True)

    st.pyplot(fig)

# -------------------------------------------------
# Raw Metrics Table
# -------------------------------------------------

st.subheader("Training History")

st.dataframe(history, use_container_width=True)

# -------------------------------------------------
# Download Results
# -------------------------------------------------

csv_bytes = history.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Training History",
    data=csv_bytes,
    file_name="training_history.csv",
    mime="text/csv",
)