"""
preprocessing.py — Config-light Data Preprocessing
=====================================================

Handles missing values, scaling, and log transforms.
Designed to be driven by Analyzer output — not hardcoded.

Key design decisions:
- fit() on train, transform() on both (prevents leakage)
- Accepts Analyzer results OR manual config
- Never silently drops data

Usage:
    from kaggle_mentor import Analyzer, Preprocessor

    a = Analyzer(train, target="SalePrice").run()
    p = Preprocessor.from_analyzer(a)
    X_train = p.fit_transform(train.drop("SalePrice", axis=1))
    X_test  = p.transform(test)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings("ignore")


class Preprocessor:
    """
    Fits preprocessing on training data, applies consistently to test.

    Parameters
    ----------
    none_fill_cols : list
        Categorical columns where NaN means 'None' (structural absence).
    zero_fill_cols : list
        Numeric columns where NaN means 0 (structural absence).
    ordinal_mappings : dict
        {column: {category: int_value}} for ordinal encoding.
    target_encode_cols : list
        High-cardinality columns to target encode.
    log_transform_cols : list
        Numeric columns to apply log1p.
    scale : bool
        Whether to apply StandardScaler (use True for linear models).
    """

    def __init__(
        self,
        none_fill_cols:    List[str] = None,
        zero_fill_cols:    List[str] = None,
        ordinal_mappings:  Dict[str, Dict] = None,
        target_encode_cols: List[str] = None,
        log_transform_cols: List[str] = None,
        scale: bool = False,
    ):
        self.none_fill_cols     = none_fill_cols or []
        self.zero_fill_cols     = zero_fill_cols or []
        self.ordinal_mappings   = ordinal_mappings or {}
        self.target_encode_cols = target_encode_cols or []
        self.log_transform_cols = log_transform_cols or []
        self.scale              = scale

        # Fitted state (populated during fit_transform)
        self._medians:       Dict[str, float] = {}
        self._modes:         Dict[str, object] = {}
        self._target_means:  Dict[str, Dict[str, float]] = {}
        self._global_mean:   float = 0.0
        self._scaler:        Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._fitted = False

    @classmethod
    def from_analyzer(cls, analyzer, scale: bool = False) -> "Preprocessor":
        """
        Build a Preprocessor from Analyzer results.

        Translates Analyzer suggestions into preprocessing config.
        User can override any setting after construction.
        """
        results = analyzer.results

        # Missing value config from analyzer
        none_fill = []
        zero_fill = []
        for item in results.get("missing", []):
            if item["pattern"] in ("structural_absence", "co_missing"):
                for col in item["columns"]:
                    # Heuristic: if column name suggests numeric measure → zero
                    # otherwise → None string
                    if any(kw in col.lower() for kw in
                           ["area", "sf", "yr", "year", "cars", "bath"]):
                        zero_fill.append(col)
                    else:
                        none_fill.append(col)

        # Encoding config from analyzer
        ordinal_mappings = {}
        target_encode_cols = []
        for item in results.get("encoding", []):
            if item["strategy"] == "ordinal":
                # Default quality mapping — user should override for their dataset
                ordinal_mappings[item["feature"]] = {
                    "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5,
                    "No": 1, "Mn": 2, "Av": 3,
                    "Unf": 1, "RFn": 2, "Fin": 3,
                    "N": 0, "P": 1, "Y": 2,
                }
            elif item["strategy"] == "target":
                target_encode_cols.append(item["feature"])

        # Skewed features
        log_cols = [item["feature"] for item in results.get("skewed_features", [])]

        return cls(
            none_fill_cols=none_fill,
            zero_fill_cols=zero_fill,
            ordinal_mappings=ordinal_mappings,
            target_encode_cols=target_encode_cols,
            log_transform_cols=log_cols,
            scale=scale,
        )

    def fit_transform(self, X: pd.DataFrame,
                      y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit on training data and transform it.

        IMPORTANT: Always call fit_transform on TRAIN, then transform on TEST.
        Never fit on test data — that would be data leakage.
        """
        X = X.copy()

        # 1. Fill structural absences
        X = self._fill_structural(X)

        # 2. Fit and apply ordinal encoding
        X = self._apply_ordinal(X)

        # 3. Fit target encoding (requires y)
        if y is not None and self.target_encode_cols:
            X = self._fit_target_encode(X, y)

        # 4. Fit medians/modes for remaining missing values
        self._fit_imputation(X)
        X = self._apply_imputation(X)

        # 5. One-hot encode remaining categoricals
        X = self._one_hot_encode(X, fit=True)

        # 6. Log transform skewed features
        X = self._apply_log_transform(X)

        # 7. Scale (optional, for linear models)
        if self.scale:
            self._scaler = StandardScaler()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = self._scaler.fit_transform(X[numeric_cols])

        self._feature_names = X.columns.tolist()
        self._fitted = True
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using fitted parameters from training.
        Must call fit_transform first.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")

        X = X.copy()
        X = self._fill_structural(X)
        X = self._apply_ordinal(X)

        if self.target_encode_cols:
            X = self._apply_target_encode(X)

        X = self._apply_imputation(X)
        X = self._one_hot_encode(X, fit=False)
        X = self._apply_log_transform(X)

        if self.scale and self._scaler:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = self._scaler.transform(X[numeric_cols])

        # Align columns with training (add missing, drop extra)
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return X

    # -----------------------------------------------------------------------
    # Internal steps
    # -----------------------------------------------------------------------

    def _fill_structural(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.none_fill_cols:
            if col in X.columns:
                X[col] = X[col].fillna("None")
        for col in self.zero_fill_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        return X

    def _apply_ordinal(self, X: pd.DataFrame) -> pd.DataFrame:
        for col, mapping in self.ordinal_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(0).astype(int)
        return X

    def _fit_target_encode(self, X: pd.DataFrame,
                           y: pd.Series) -> pd.DataFrame:
        self._global_mean = float(y.mean())
        for col in self.target_encode_cols:
            if col not in X.columns:
                continue
            tmp = X[[col]].copy()
            tmp["_y"] = y.values
            means = tmp.groupby(col)["_y"].mean().to_dict()
            self._target_means[col] = means
            X[col] = X[col].map(means).fillna(self._global_mean)
        return X

    def _apply_target_encode(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.target_encode_cols:
            if col not in X.columns:
                continue
            means = self._target_means.get(col, {})
            X[col] = X[col].map(means).fillna(self._global_mean)
        return X

    def _fit_imputation(self, X: pd.DataFrame):
        for col in X.select_dtypes(include=[np.number]).columns:
            self._medians[col] = float(X[col].median())
        for col in X.select_dtypes(include="object").columns:
            mode = X[col].mode()
            self._modes[col] = mode[0] if len(mode) > 0 else "Unknown"

    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(self._medians.get(col, 0))
        for col in X.select_dtypes(include="object").columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(self._modes.get(col, "Unknown"))
        return X

    def _one_hot_encode(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        if not cat_cols:
            return X
        if fit:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            self._ohe_columns = X.columns.tolist()
        else:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        return X

    def _apply_log_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.log_transform_cols:
            if col in X.columns and X[col].dtype in [np.float64, np.int64, float, int]:
                X[col] = np.log1p(X[col].clip(lower=0))
        return X
