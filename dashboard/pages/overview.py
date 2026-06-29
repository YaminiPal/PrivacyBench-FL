import os
import yaml
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Overview", layout="wide")

st.title("🧠 PrivacyBench-FL Overview")
st.markdown(
    """
    This dashboard summarizes the current Federated Learning experiment,
    including dataset information, system configuration, and enabled features.
    """
)

CONFIG_PATH = "config.yaml"
DATASET_PATH = "data/raw/adult.csv"
SCHEMA_PATH = "dataset_schema.json"


# ======================================================
# LOAD CONFIG
# ======================================================
config = {}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Could not read config.yaml: {e}")
else:
    st.warning("config.yaml not found.")


# ======================================================
# SYSTEM CONFIGURATION
# ======================================================
st.header("⚙️ System Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Federated Learning")
    st.write(f"**Rounds:** {config.get('rounds', 'N/A')}")
    st.write(f"**Local Epochs:** {config.get('local_epochs', 'N/A')}")
    st.write(f"**Learning Rate:** {config.get('lr', 'N/A')}")
    st.write(f"**Device:** {config.get('device', 'cpu')}")

with col2:
    dp_cfg = config.get("dp", {})
    drift_cfg = config.get("drift", {})

    st.markdown("#### Privacy & Drift")
    st.write(f"**Noise Multiplier:** {dp_cfg.get('noise_multiplier', 'N/A')}")
    st.write(f"**Max ε:** {dp_cfg.get('max_epsilon', 'N/A')}")
    st.write(f"**Target δ:** {dp_cfg.get('delta', 'N/A')}")
    st.write(f"**Drift Mode:** {drift_cfg.get('mode', 'Disabled')}")


# ======================================================
# DATASET SUMMARY
# ======================================================
st.header("📊 Dataset Summary")

if os.path.exists(DATASET_PATH):
    try:
        from preprocessing.schema_check import SchemaChecker
        checker = SchemaChecker()
        df = checker.load_data(DATASET_PATH)

        c1, c2, c3 = st.columns(3)

        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))

        if "income" in df.columns:
            c3.metric("Target Classes", df["income"].nunique())
        else:
            c3.metric("Target Classes", "N/A")

        with st.expander("Preview Dataset"):
            st.dataframe(df.head(), use_container_width=True)

        with st.expander("Column Names"):
            st.write(list(df.columns))

    except Exception as e:
        st.error(f"Unable to load dataset: {e}")

else:
    st.info("Raw dataset not found at data/raw/adult.csv. Run `python main.py` first.")


# ======================================================
# ENABLED FEATURES
# ======================================================
st.header("🚀 Enabled Features")

features = [
    "✅ Federated Learning (FedAvg)",
    "✅ Differential Privacy",
    "✅ Adaptive Gradient Clipping",
    "✅ Client Selection",
    "✅ Quantization / Communication Compression",
    "✅ Non-IID Data Partitioning",
    "✅ Temporal / Drift Simulation",
    "✅ Continual Learning Evaluation",
    "✅ Privacy Accounting",
    "✅ Early Stopping",
]

for feature in features:
    st.write(feature)


# ======================================================
# PROJECT PIPELINE
# ======================================================
st.header("🔄 Pipeline")

st.code(
    """
Raw Dataset
      │
      ▼
Schema Validation
      │
      ▼
Client Split (IID / Non-IID / Temporal)
      │
      ▼
Local Client Preprocessing
      │
      ▼
Federated Training
      │
      ├── Client Selection
      ├── Differential Privacy
      ├── Quantization
      └── FedAvg Aggregation
      │
      ▼
Evaluation & Privacy Accounting
      │
      ▼
Dashboard Visualization
""",
    language="text",
)