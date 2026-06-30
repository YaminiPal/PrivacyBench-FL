import pandas as pd
import json
from typing import Dict, List


class SchemaChecker:
    def __init__(self, target_column="income"):
        self.target_column = target_column

        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.leakage_features: List[str] = []
        self.schema: Dict = {}

    # -----------------------------
    # Load dataset
    # -----------------------------
    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=None)

        # Adult dataset column names (fixed schema)
        df.columns = [
            "age", "workclass", "fnlwgt", "education",
            "education-num", "marital-status", "occupation",
            "relationship", "race", "sex",
            "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]

        return df

    # -----------------------------
    # Detect feature types
    # -----------------------------
    def detect_feature_types(self, df: pd.DataFrame):
        for col in df.columns:

            if col == self.target_column:
                continue

            # numeric detection
            if df[col].dtype in ["int64", "float64"]:
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)

    # -----------------------------
    # Leakage detection logic
    # -----------------------------
    def detect_leakage(self):
        leakage_keywords = ["id", "fnlwgt"]

        for feature in self.numeric_features + self.categorical_features:
            if any(k in feature.lower() for k in leakage_keywords):
                self.leakage_features.append(feature)

        # domain-based leakage flagging
        high_risk_features = ["fnlwgt"]
        for f in high_risk_features:
            if f not in self.leakage_features:
                self.leakage_features.append(f)

    # -----------------------------
    # Build schema object
    # -----------------------------
    def build_schema(self, df: pd.DataFrame):

        self.detect_feature_types(df)
        self.detect_leakage()

        self.schema = {
            "target": self.target_column,
            "num_features": self.numeric_features,
            "cat_features": self.categorical_features,
            "leakage_features": self.leakage_features,
            "num_samples": len(df)
        }

        return self.schema

    # -----------------------------
    # Save schema
    # -----------------------------
    def save_schema(self, path="dataset_schema.json"):
        with open(path, "w") as f:
            json.dump(self.schema, f, indent=4)

    # -----------------------------
    # Run full pipeline
    # -----------------------------
    def run(self, dataset_path: str):
        df = self.load_data(dataset_path)

        schema = self.build_schema(df)

        self.save_schema()

        print("\n✅ Schema Analysis Complete")
        print(json.dumps(schema, indent=4))

        return df, schema


def validate_schema(raw_df: pd.DataFrame = None, dataset_path: str = "data/raw/adult.csv",
                    target_column: str = "income", drop_leakage: bool = True):
    """
    Phase 1 entry point: load dataset, detect schema, optionally drop leakage columns.
    Returns (clean_df, target_col, schema_dict).
    """
    checker = SchemaChecker(target_column=target_column)

    if raw_df is None:
        df = checker.load_data(dataset_path)
    else:
        df = raw_df.copy()
        if df.columns.dtype == "int64" or all(isinstance(c, int) for c in df.columns):
            df.columns = [
                "age", "workclass", "fnlwgt", "education",
                "education-num", "marital-status", "occupation",
                "relationship", "race", "sex",
                "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"
            ]

    schema = checker.build_schema(df)
    checker.save_schema()

    clean_df = df.copy()
    if drop_leakage:
        cols_to_drop = [c for c in checker.leakage_features if c in clean_df.columns and c != target_column]
        if cols_to_drop:
            clean_df = clean_df.drop(columns=cols_to_drop)

    return clean_df, target_column, schema