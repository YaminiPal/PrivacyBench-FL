import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class LocalPreprocessor:
    def __init__(self, numerical_cols=None, categorical_cols=None, target_col=None,
                 categories=None):
        """
        Independent client-side preprocessing pipeline to prevent global data leakage.
        categories: optional fixed vocab per categorical column (from schema only, no stats).
        """
        self.numerical_cols = numerical_cols if numerical_cols else []
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.target_col = target_col
        self.categories = categories
        
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if categories:
            self.encoder.set_params(categories=categories)
        
        self.is_fitted = False

    # ====================================
    # FIT AND TRANSFORM (LOCAL ONLY)
    # ====================================
    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits the transformers and transforms the local dataframe entirely 
        within the client boundary space.
        Returns processed feature array X and target array y.
        """
        X_processed_parts = []
        
        # 1. Handle Numerical Streams
        if self.numerical_cols:
            num_data = df[self.numerical_cols].values
            num_imputed = self.num_imputer.fit_transform(num_data)
            num_scaled = self.scaler.fit_transform(num_imputed)
            X_processed_parts.append(num_scaled)
            
        # 2. Handle Categorical Streams
        if self.categorical_cols:
            cat_data = df[self.categorical_cols].values
            cat_imputed = self.cat_imputer.fit_transform(cat_data)
            cat_encoded = self.encoder.fit_transform(cat_imputed)
            X_processed_parts.append(cat_encoded)
            
        # Combine processed columns
        X = np.hstack(X_processed_parts) if X_processed_parts else np.array([])
        
        # 3. Extract Target Feature Label
        if self.target_col and self.target_col in df.columns:
            y = self._encode_target(df[self.target_col].values)
        else:
            y = np.array([])
        
        self.is_fitted = True
        return X, y

    # ====================================
    # TRANSFORM ONLY (FOR INCOMING VALIDATION)
    # ====================================
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms incoming local validation/test splits using previously fitted client parameters.
        """
        if not self.is_fitted:
            raise ValueError("The local preprocessor must be fit on local training data before calling transform().")
            
        X_processed_parts = []
        
        if self.numerical_cols:
            num_data = df[self.numerical_cols].values
            num_imputed = self.num_imputer.transform(num_data)
            num_scaled = self.scaler.transform(num_imputed)
            X_processed_parts.append(num_scaled)
            
        if self.categorical_cols:
            cat_data = df[self.categorical_cols].values
            cat_imputed = self.cat_imputer.transform(cat_data)
            cat_encoded = self.encoder.transform(cat_imputed)
            X_processed_parts.append(cat_encoded)
            
        X = np.hstack(X_processed_parts) if X_processed_parts else np.array([])
        y = self._encode_target(df[self.target_col].values) if self.target_col else np.array([])
        
        return X, y

    def _encode_target(self, y: np.ndarray) -> np.ndarray:
        """Encode binary income labels to 0/1 integers."""
        if y.dtype.kind in ("i", "u", "f"):
            return y.astype(np.int64)
        encoded = np.array([1 if ">50K" in str(v) else 0 for v in y], dtype=np.int64)
        return encoded

    # ====================================
    # SAVE LOCAL METADATA ARCHIVES
    # ====================================
    def save_state(self, export_dir: str):
        """
        Saves the fitted transformer states into the specific client directory cache.
        """
        os.makedirs(export_dir, exist_ok=True)
        meta_path = os.path.join(export_dir, "local_preprocessor.pkl")
        with open(meta_path, "wb") as f:
            state = {
                      "num_imputer": self.num_imputer,
                      "scaler": self.scaler,
                      "cat_imputer": self.cat_imputer,
                      "encoder": self.encoder,
                      "numerical_cols": self.numerical_cols,
                      "categorical_cols": self.categorical_cols,
                      "target_col": self.target_col
                    }

            pickle.dump(state, f)

    @classmethod
    def load_state(cls, import_dir: str):
        """
        Loads a pre-fitted state file from disk cache.
        """
        meta_path = os.path.join(import_dir, "local_preprocessor.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No preprocessing metadata found at: {meta_path}")
        with open(meta_path, "rb") as f:
            return pickle.load(f)