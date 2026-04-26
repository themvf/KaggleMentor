"""
pipeline.py — Lightweight Orchestration
=========================================

Thin glue layer. Does NOT make modeling decisions.
Connects Analyzer → Preprocessor → FeatureHook → user's model.

Design principle:
- User controls the model (we don't wrap sklearn/xgboost)
- Pipeline prepares data and gets out of the way
- Every step is optional and overridable

Usage:
    from kaggle_mentor import Pipeline, FeatureHook
    from sklearn.linear_model import Ridge

    def my_features(df):
        df = df.copy()
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        return df

    pipeline = Pipeline(
        train_path="train.csv",
        test_path="test.csv",
        target="SalePrice",
        feature_fn=my_features,
    )

    X_train, X_test, y = pipeline.prepare()

    # User controls modeling
    model = Ridge(alpha=10)
    model.fit(X_train, y)
    predictions = np.expm1(model.predict(X_test))
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional
import warnings
warnings.filterwarnings("ignore")

from .analyzer import Analyzer
from .preprocessing import Preprocessor
from .features import FeatureHook, validate_features
from .reporting import Reporter


class Pipeline:
    """
    Lightweight data preparation pipeline.

    Parameters
    ----------
    train_path : str
        Path to training CSV.
    test_path : str
        Path to test CSV.
    target : str
        Target column name.
    feature_fn : callable, optional
        User-defined feature engineering function.
        Signature: fn(df: DataFrame) -> DataFrame
    scale : bool
        Apply StandardScaler (True for linear models, False for trees).
    run_analysis : bool
        Run Analyzer automatically during prepare(). Default True.
    competition : str
        Competition name for reports.
    """

    def __init__(
        self,
        train_path: str,
        test_path: str,
        target: str,
        feature_fn: Optional[Callable] = None,
        scale: bool = False,
        run_analysis: bool = True,
        competition: str = "Kaggle Competition",
    ):
        self.train_path   = train_path
        self.test_path    = test_path
        self.target       = target
        self.feature_fn   = feature_fn
        self.scale        = scale
        self.run_analysis = run_analysis
        self.competition  = competition

        # Set after prepare()
        self.analyzer:     Optional[Analyzer]     = None
        self.preprocessor: Optional[Preprocessor] = None
        self._train_raw:   Optional[pd.DataFrame] = None
        self._test_raw:    Optional[pd.DataFrame] = None

    def prepare(self, save_report: bool = True):
        """
        Run the full data preparation pipeline.

        Steps:
        1. Load data
        2. Run Analyzer (EDA + recommendations)
        3. Apply feature engineering hook (if provided)
        4. Build Preprocessor from Analyzer results
        5. fit_transform train, transform test
        6. Validate output
        7. Optionally save EDA report

        Returns
        -------
        X_train : pd.DataFrame
        X_test  : pd.DataFrame
        y       : pd.Series (log-transformed if target is skewed)
        """
        print("=" * 60)
        print(f"🏠 kaggle_mentor Pipeline: {self.competition}")
        print("=" * 60)

        # Step 1: Load
        print("\n📂 Loading data...")
        self._train_raw = pd.read_csv(self.train_path)
        self._test_raw  = pd.read_csv(self.test_path)
        print(f"   Train: {self._train_raw.shape}  |  Test: {self._test_raw.shape}")

        # Step 2: Analyze
        if self.run_analysis:
            print("\n🔍 Running analysis...")
            self.analyzer = Analyzer(self._train_raw, target=self.target)
            self.analyzer.run()

            if save_report:
                reporter = Reporter(
                    self.analyzer.results,
                    target=self.target,
                    competition=self.competition
                )
                reporter.save("EDA_REPORT.md")

        # Step 3: Prepare target
        y_raw = self._train_raw[self.target]
        if self.analyzer:
            t = self.analyzer.results.get("target", {})
            if "log" in t.get("suggestion", ""):
                y = np.log1p(y_raw)
                print(f"\n✅ Target log-transformed (skewness={t['signal']['skewness']})")
            else:
                y = y_raw.copy()
                print(f"\n✅ Target used as-is (skewness={t['signal']['skewness']})")
        else:
            y = y_raw.copy()

        # Step 4: Combine train + test for consistent preprocessing
        n_train = len(self._train_raw)
        train_features = self._train_raw.drop(columns=[self.target])

        # Remove ID columns if present
        id_cols = [c for c in train_features.columns
                   if c.lower() in ("id", "passengerid", "customerid")]
        if id_cols:
            print(f"\n   Dropping ID columns: {id_cols}")
            train_features = train_features.drop(columns=id_cols)
            test_features  = self._test_raw.drop(
                columns=[c for c in id_cols if c in self._test_raw.columns])
        else:
            test_features = self._test_raw.copy()

        all_data = pd.concat([train_features, test_features],
                             axis=0).reset_index(drop=True)

        # Step 5: Feature engineering hook
        if self.feature_fn:
            print("\n🏗️  Applying feature engineering hook...")
            hook = FeatureHook(self.feature_fn, name="user_features")
            all_data = hook.apply(all_data)

        # Step 6: Build preprocessor and transform
        print("\n🔧 Preprocessing...")
        if self.analyzer:
            self.preprocessor = Preprocessor.from_analyzer(
                self.analyzer, scale=self.scale)
        else:
            self.preprocessor = Preprocessor(scale=self.scale)

        train_data = all_data.iloc[:n_train].copy()
        test_data  = all_data.iloc[n_train:].copy()

        X_train = self.preprocessor.fit_transform(train_data, y)
        X_test  = self.preprocessor.transform(test_data)

        # Step 7: Validate
        print("\n🔍 Validating features...")
        validate_features(X_train)

        print(f"\n✅ Pipeline complete!")
        print(f"   X_train: {X_train.shape}")
        print(f"   X_test:  {X_test.shape}")
        print(f"   y:       {y.shape}")
        print("\n👉 You control the model. Suggested next step:")
        print("   from sklearn.linear_model import Ridge")
        print("   model = Ridge(alpha=10).fit(X_train, y)")

        return X_train, X_test, y

    def generate_submission(self, predictions: np.ndarray,
                            id_col: str = "Id",
                            target_col: Optional[str] = None,
                            filename: str = "submission.csv",
                            inverse_transform: bool = True):
        """
        Generate a Kaggle submission CSV.

        Parameters
        ----------
        predictions : array of predicted values (log scale if inverse_transform=True)
        id_col : name of ID column in test data
        target_col : output column name (defaults to self.target)
        filename : output filename
        inverse_transform : if True, applies np.expm1 to reverse log transform
        """
        if self._test_raw is None:
            raise RuntimeError("Call prepare() first.")

        target_col = target_col or self.target
        final_preds = np.expm1(predictions) if inverse_transform else predictions

        submission = pd.DataFrame({
            id_col: self._test_raw[id_col],
            target_col: final_preds
        })
        submission.to_csv(filename, index=False)
        print(f"✅ Submission saved: {filename}")
        print(f"   Predictions: min=${final_preds.min():,.0f} | "
              f"mean=${final_preds.mean():,.0f} | max=${final_preds.max():,.0f}")
        return submission
