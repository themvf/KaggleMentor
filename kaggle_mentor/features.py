"""
features.py — Feature Hook System
===================================

Feature engineering is dataset-specific — no library can automate it.
Instead, we provide a clean hook system so users plug in their own logic.

Design:
- FeatureHook wraps a user-defined function
- Pipeline calls it at the right time
- Validates input/output to catch mistakes early

Usage:
    from kaggle_mentor import FeatureHook

    def my_features(df):
        df = df.copy()
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["QualityArea"] = df["OverallQual"] * df["GrLivArea"]
        return df

    hook = FeatureHook(my_features)
    df_engineered = hook.apply(df)

Also provides generic utilities that work across datasets:
- add_binary_indicators()
- add_ratio_feature()
- add_interaction()
- validate_features()
"""

import numpy as np
import pandas as pd
from typing import Callable, List, Optional
import warnings
warnings.filterwarnings("ignore")


class FeatureHook:
    """
    Wraps a user-defined feature engineering function.

    Parameters
    ----------
    fn : callable
        Function that takes a DataFrame and returns a DataFrame.
        Must not modify the input in-place (use df.copy()).
    name : str, optional
        Name for logging/debugging.
    validate : bool
        If True, checks that output has more columns than input
        and no new NaN values were introduced.
    """

    def __init__(self, fn: Callable, name: str = "custom_features",
                 validate: bool = True):
        self.fn = fn
        self.name = name
        self.validate = validate

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the feature function and optionally validate output."""
        n_cols_before = df.shape[1]
        n_rows_before = df.shape[0]

        result = self.fn(df)

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"FeatureHook '{self.name}': function must return a DataFrame, "
                f"got {type(result)}"
            )

        if result.shape[0] != n_rows_before:
            raise ValueError(
                f"FeatureHook '{self.name}': row count changed "
                f"({n_rows_before} → {result.shape[0]}). "
                "Feature functions must not add or remove rows."
            )

        if self.validate:
            n_new = result.shape[1] - n_cols_before
            new_nulls = result.isnull().sum().sum() - df.isnull().sum().sum()

            if n_new > 0:
                new_cols = [c for c in result.columns if c not in df.columns]
                print(f"✅ FeatureHook '{self.name}': added {n_new} features: "
                      f"{new_cols[:5]}" + (" ..." if n_new > 5 else ""))
            if new_nulls > 0:
                print(f"⚠️  FeatureHook '{self.name}': introduced {new_nulls} "
                      f"new NaN values — check your feature logic")

        return result

    def __repr__(self):
        return f"FeatureHook(name='{self.name}', fn={self.fn.__name__})"


# ---------------------------------------------------------------------------
# Generic feature utilities — work across datasets
# ---------------------------------------------------------------------------

def add_binary_indicators(df: pd.DataFrame,
                           columns: List[str],
                           prefix: str = "Has") -> pd.DataFrame:
    """
    Add binary (0/1) indicator for whether a numeric column is > 0.

    WHY: Presence/absence is often more predictive than the raw value.
    Example: HasPool is more useful than PoolArea for most houses.

    Parameters
    ----------
    df : DataFrame
    columns : list of column names to create indicators for
    prefix : str, default "Has"

    Returns
    -------
    DataFrame with new indicator columns added.

    Example:
        df = add_binary_indicators(df, ["PoolArea", "GarageArea", "Fireplaces"])
        # Creates: HasPoolArea, HasGarageArea, HasFireplaces
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{prefix}{col}"] = (df[col] > 0).astype(int)
        else:
            print(f"⚠️  add_binary_indicators: column '{col}' not found, skipping")
    return df


def add_interaction(df: pd.DataFrame,
                    col1: str, col2: str,
                    name: Optional[str] = None,
                    operation: str = "multiply") -> pd.DataFrame:
    """
    Create an interaction feature between two columns.

    WHY: Some features are more powerful in combination.
    Example: OverallQual × GrLivArea captures "large AND high quality".

    Parameters
    ----------
    df : DataFrame
    col1, col2 : column names
    name : output column name (default: col1_x_col2)
    operation : 'multiply', 'add', 'subtract', 'divide'

    Example:
        df = add_interaction(df, "OverallQual", "GrLivArea", "QualityArea")
    """
    df = df.copy()
    if col1 not in df.columns or col2 not in df.columns:
        missing = [c for c in [col1, col2] if c not in df.columns]
        print(f"⚠️  add_interaction: columns {missing} not found, skipping")
        return df

    out_name = name or f"{col1}_x_{col2}"
    if operation == "multiply":
        df[out_name] = df[col1] * df[col2]
    elif operation == "add":
        df[out_name] = df[col1] + df[col2]
    elif operation == "subtract":
        df[out_name] = df[col1] - df[col2]
    elif operation == "divide":
        df[out_name] = df[col1] / (df[col2].replace(0, np.nan))
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return df


def add_ratio_feature(df: pd.DataFrame,
                      numerator: str, denominator: str,
                      name: Optional[str] = None) -> pd.DataFrame:
    """
    Create a ratio feature (numerator / denominator).

    WHY: Ratios normalize for size. Price per sqft is more comparable
    than raw price. Quality per room is more informative than total quality.

    Example:
        df = add_ratio_feature(df, "GrLivArea", "TotRmsAbvGrd", "SFPerRoom")
    """
    return add_interaction(df, numerator, denominator, name, operation="divide")


def validate_features(df: pd.DataFrame,
                      expected_cols: Optional[List[str]] = None) -> dict:
    """
    Validate a feature DataFrame before modeling.

    Checks:
    - No NaN values
    - No infinite values
    - No zero-variance columns
    - All expected columns present (if provided)

    Returns a dict with 'passed' bool and 'issues' list.

    Example:
        report = validate_features(X_train)
        if not report['passed']:
            print(report['issues'])
    """
    issues = []

    # Check NaN
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        issues.append(f"NaN values in {len(nan_cols)} columns: {nan_cols[:5]}")

    # Check infinite
    numeric = df.select_dtypes(include=[np.number])
    inf_cols = numeric.columns[np.isinf(numeric).any()].tolist()
    if inf_cols:
        issues.append(f"Infinite values in {len(inf_cols)} columns: {inf_cols[:5]}")

    # Check zero variance
    zero_var = numeric.columns[numeric.std() == 0].tolist()
    if zero_var:
        issues.append(f"Zero-variance columns (consider dropping): {zero_var[:5]}")

    # Check expected columns
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            issues.append(f"Missing expected columns: {missing}")

    passed = len(issues) == 0
    if passed:
        print(f"✅ Feature validation passed ({df.shape[1]} features, {df.shape[0]} rows)")
    else:
        print(f"⚠️  Feature validation found {len(issues)} issue(s):")
        for issue in issues:
            print(f"   - {issue}")

    return {"passed": passed, "issues": issues}
