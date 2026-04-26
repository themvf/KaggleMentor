"""
analyzer.py — The Intelligence Layer
======================================

Converts a raw DataFrame into a structured, confidence-rated
modeling strategy. Separates SIGNAL (objective observations)
from SUGGESTION (heuristic recommendations).

Design principles:
- Signal is always objective and measurable
- Suggestions include confidence level + reasoning
- Never overconfident: MNAR/MAR is inferred, not certified
- Give options, not directives (especially for multicollinearity)
- Output is a structured dict — reporter and pipeline consume it

Usage:
    a = Analyzer(df, target="SalePrice")
    a.run()
    print(a.results)        # structured dict
    print(a.summary())      # human-readable text
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, pearsonr
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Confidence levels — used throughout to be explicit about certainty
# ---------------------------------------------------------------------------
HIGH   = "high"
MEDIUM = "medium"
LOW    = "low"


class Analyzer:
    """
    Analyzes a DataFrame and produces structured insights.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (with target column included).
    target : str
        Name of the target column.
    problem_type : str, optional
        'auto' (default), 'regression', or 'classification'.
        Auto-detects based on target cardinality and dtype.

    Attributes
    ----------
    results : dict
        Structured output after calling .run().
        Schema documented in _build_schema().
    """

    def __init__(self, df: pd.DataFrame, target: str,
                 problem_type: str = "auto"):
        self.df = df.copy()
        self.target = target
        self.problem_type = problem_type
        self.results = {}
        self._n = len(df)
        self._features = [c for c in df.columns if c != target]
        self._numeric = df[self._features].select_dtypes(
            include=[np.number]).columns.tolist()
        self._categorical = df[self._features].select_dtypes(
            include="object").columns.tolist()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> "Analyzer":
        """Run all analysis steps and populate self.results."""
        print("🔍 Running Analyzer...\n")

        self.results = {
            "meta":             self._analyze_meta(),
            "target":           self._analyze_target(),
            "missing":          self._analyze_missing(),
            "correlations":     self._analyze_correlations(),
            "multicollinearity": self._analyze_multicollinearity(),
            "skewed_features":  self._analyze_skewness(),
            "encoding":         self._analyze_encoding(),
            "outliers":         self._analyze_outliers(),
        }

        self._print_summary()
        return self

    def summary(self) -> str:
        """Return a human-readable text summary of results."""
        if not self.results:
            return "Run .run() first."
        lines = []
        lines.append("=" * 60)
        lines.append("ANALYZER SUMMARY")
        lines.append("=" * 60)
        lines.append(self._format_section("META",            self.results["meta"]))
        lines.append(self._format_section("TARGET",          self.results["target"]))
        lines.append(self._format_missing_summary())
        lines.append(self._format_correlation_summary())
        lines.append(self._format_multicollinearity_summary())
        lines.append(self._format_skewness_summary())
        lines.append(self._format_encoding_summary())
        lines.append(self._format_outlier_summary())
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Analysis steps — each returns a structured dict
    # -----------------------------------------------------------------------

    def _analyze_meta(self) -> dict:
        """Basic dataset characteristics."""
        n_train = self._n
        n_features = len(self._features)
        n_numeric = len(self._numeric)
        n_categorical = len(self._categorical)

        # Detect problem type
        if self.problem_type == "auto":
            target_series = self.df[self.target]
            if target_series.dtype == object:
                problem_type = "classification"
            elif target_series.nunique() <= 20:
                problem_type = "classification"
            else:
                problem_type = "regression"
        else:
            problem_type = self.problem_type

        return {
            "n_samples":      n_train,
            "n_features":     n_features,
            "n_numeric":      n_numeric,
            "n_categorical":  n_categorical,
            "problem_type":   problem_type,
            "target":         self.target,
        }

    def _analyze_target(self) -> dict:
        """
        Analyze target variable distribution.

        Signal:     skewness, kurtosis, range, distribution shape
        Suggestion: whether to apply log transform
        Confidence: high if skewness > 1.0 (strong evidence)
                    medium if 0.5 < skewness < 1.0
                    low if skewness < 0.5
        """
        y = self.df[self.target].dropna()
        skewness = float(skew(y))
        kurt = float(y.kurt())

        # Signal — objective
        signal = {
            "min":      float(y.min()),
            "max":      float(y.max()),
            "mean":     float(y.mean()),
            "median":   float(y.median()),
            "std":      float(y.std()),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
        }

        # Suggestion — heuristic with confidence
        if abs(skewness) > 1.0:
            suggestion = "apply log transform (log1p recommended)"
            confidence = HIGH
            reasoning = (
                f"Skewness of {skewness:.2f} is strongly non-normal. "
                "Log transform typically improves RMSE by 20-30% for "
                "regression targets with this level of skew."
            )
        elif abs(skewness) > 0.5:
            suggestion = "consider log transform — moderate skew detected"
            confidence = MEDIUM
            reasoning = (
                f"Skewness of {skewness:.2f} suggests moderate right-skew. "
                "Log transform may help but test both with cross-validation."
            )
        else:
            suggestion = "no transform likely needed — distribution is near-normal"
            confidence = HIGH
            reasoning = (
                f"Skewness of {skewness:.2f} is within acceptable range. "
                "Linear models should work well without transformation."
            )

        return {
            "signal":     signal,
            "suggestion": suggestion,
            "confidence": confidence,
            "reasoning":  reasoning,
        }

    def _analyze_missing(self) -> list:
        """
        Analyze missing value patterns.

        Separates MNAR inference from MAR inference.
        Never claims certainty — uses 'pattern suggests' language.

        Co-missing detection: if columns are missing together,
        it suggests structural absence (MNAR), not random error.
        """
        missing_counts = self.df[self._features].isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].sort_values(
            ascending=False)

        if len(missing_cols) == 0:
            return []

        results = []

        # Detect co-missing groups (columns that are always missing together)
        missing_df = self.df[missing_cols.index].isnull()
        processed = set()

        for col in missing_cols.index:
            if col in processed:
                continue

            col_missing_idx = set(missing_df.index[missing_df[col]])
            co_missing = [col]

            for other in missing_cols.index:
                if other == col or other in processed:
                    continue
                other_missing_idx = set(missing_df.index[missing_df[other]])
                if len(col_missing_idx) > 0 and len(other_missing_idx) > 0:
                    overlap = len(col_missing_idx & other_missing_idx)
                    union   = len(col_missing_idx | other_missing_idx)
                    jaccard = overlap / union if union > 0 else 0
                    if jaccard > 0.85:  # 85%+ overlap = co-missing
                        co_missing.append(other)

            processed.update(co_missing)
            pct = float(missing_counts[col] / self._n * 100)

            # Infer pattern — with explicit uncertainty
            if pct > 70:
                pattern = "structural_absence"
                suggestion = "fill with 'None' or 0 — absence is the information"
                confidence = HIGH
                reasoning = (
                    f"{pct:.0f}% missing strongly suggests this feature "
                    "doesn't apply to most houses (e.g., no pool, no alley). "
                    "Missingness is informative, not an error."
                )
            elif len(co_missing) > 1:
                pattern = "co_missing"
                suggestion = (
                    f"fill with 'None'/0 — {len(co_missing)} columns missing "
                    "together suggests structural absence"
                )
                confidence = MEDIUM
                reasoning = (
                    f"Columns {co_missing} are missing together "
                    f"(>85% overlap). Pattern suggests structural absence "
                    "(e.g., no garage → all garage columns missing). "
                    "Note: co-missing is inferred, not certified as MNAR."
                )
            else:
                pattern = "possibly_random"
                suggestion = (
                    "impute with median or neighborhood-grouped median — "
                    "or investigate further"
                )
                confidence = LOW
                reasoning = (
                    f"{pct:.1f}% missing with no clear co-missing pattern. "
                    "Could be MAR (depends on other features) or random. "
                    "Consider KNN imputation or group-based median."
                )

            results.append({
                "columns":    co_missing,
                "count":      int(missing_counts[col]),
                "pct":        round(pct, 1),
                "signal":     f"{pct:.1f}% missing" + (
                    f" (co-missing with {len(co_missing)-1} other columns)"
                    if len(co_missing) > 1 else ""
                ),
                "pattern":    pattern,
                "suggestion": suggestion,
                "confidence": confidence,
                "reasoning":  reasoning,
            })

        return results

    def _analyze_correlations(self) -> list:
        """
        Find features most correlated with target.

        Signal:     Pearson correlation coefficient
        Suggestion: feature importance ranking
        Note:       Pearson measures LINEAR correlation only.
                    Tree models may find non-linear relationships
                    in features with low Pearson correlation.
        """
        if self.results.get("meta", {}).get("problem_type") == "classification":
            return []  # Pearson not appropriate for classification

        y = self.df[self.target]
        results = []

        for col in self._numeric:
            col_data = self.df[col].dropna()
            common_idx = col_data.index.intersection(y.dropna().index)
            if len(common_idx) < 10:
                continue
            try:
                r, p = pearsonr(col_data[common_idx], y[common_idx])
                results.append({
                    "feature":     col,
                    "correlation": round(float(r), 3),
                    "p_value":     round(float(p), 4),
                    "signal":      f"r = {r:.3f}",
                    "strength":    self._correlation_strength(abs(r)),
                })
            except Exception:
                continue

        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Add suggestion to top features
        for item in results[:10]:
            item["suggestion"] = (
                "strong predictor — include in all models"
                if abs(item["correlation"]) > 0.5
                else "moderate predictor — include, monitor importance"
            )
            item["reasoning"] = (
                "Note: Pearson measures linear correlation only. "
                "Tree models may extract additional non-linear signal."
            )

        return results

    def _analyze_multicollinearity(self) -> list:
        """
        Detect highly correlated feature pairs.

        Signal:     Pearson r between feature pairs
        Suggestion: options (not directives) for handling
        Threshold:  r > 0.8 flagged (configurable)
        """
        threshold = 0.8
        results = []
        cols = self._numeric

        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                data = self.df[[col1, col2]].dropna()
                if len(data) < 10:
                    continue
                try:
                    r, _ = pearsonr(data[col1], data[col2])
                    if abs(r) >= threshold:
                        results.append({
                            "feature_1":  col1,
                            "feature_2":  col2,
                            "correlation": round(float(r), 3),
                            "signal":     f"r = {r:.3f} (high multicollinearity)",
                            "suggestion": (
                                f"Options: (1) drop '{col2}' if '{col1}' has "
                                f"higher target correlation, "
                                f"(2) combine into single feature, "
                                f"(3) use regularization (Ridge/Lasso) to handle automatically"
                            ),
                            "confidence": HIGH if abs(r) > 0.9 else MEDIUM,
                            "reasoning": (
                                f"Correlation of {r:.2f} between these features "
                                "causes instability in linear model coefficients. "
                                "Tree models are less affected but still benefit "
                                "from removing redundancy."
                            ),
                        })
                except Exception:
                    continue

        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return results

    def _analyze_skewness(self) -> list:
        """
        Identify numeric features with high skewness.

        Signal:     skewness value per feature
        Suggestion: apply log1p transform
        Threshold:  |skewness| > 0.75 (standard heuristic)
        """
        threshold = 0.75
        results = []

        for col in self._numeric:
            data = self.df[col].dropna()
            if len(data) < 10:
                continue
            s = float(skew(data))
            if abs(s) > threshold:
                results.append({
                    "feature":    col,
                    "skewness":   round(s, 3),
                    "signal":     f"skewness = {s:.3f}",
                    "suggestion": "apply log1p transform",
                    "confidence": HIGH if abs(s) > 2.0 else MEDIUM,
                    "reasoning": (
                        f"Skewness of {s:.2f} exceeds threshold of {threshold}. "
                        "log1p handles zero values safely. "
                        "Primarily benefits linear models; tree models are "
                        "less sensitive to feature skewness."
                    ),
                })

        results.sort(key=lambda x: abs(x["skewness"]), reverse=True)
        return results

    def _analyze_encoding(self) -> list:
        """
        Recommend encoding strategy for categorical features.

        Signal:     cardinality, value distribution
        Suggestion: one-hot / ordinal / target encoding
        """
        results = []

        # Common ordinal quality patterns
        quality_keywords = [
            "qual", "cond", "qc", "finish", "exposure",
            "slope", "shape", "drive", "functional"
        ]

        for col in self._categorical:
            n_cats = self.df[col].nunique()
            top_freq = self.df[col].value_counts(normalize=True).iloc[0]

            # Detect likely ordinal
            col_lower = col.lower()
            is_likely_ordinal = any(kw in col_lower for kw in quality_keywords)
            unique_vals = set(self.df[col].dropna().unique())
            quality_vals = {"Po", "Fa", "TA", "Gd", "Ex", "None",
                            "No", "Mn", "Av", "Unf", "RFn", "Fin"}
            has_quality_vals = len(unique_vals & quality_vals) >= 2

            if is_likely_ordinal or has_quality_vals:
                strategy = "ordinal"
                suggestion = (
                    "ordinal encode — natural quality ordering detected "
                    "(e.g., Po=1, Fa=2, TA=3, Gd=4, Ex=5)"
                )
                confidence = MEDIUM
                reasoning = (
                    "Column name or values suggest quality/condition ordering. "
                    "Verify the ordering makes sense for your domain."
                )
            elif n_cats > 15:
                strategy = "target"
                suggestion = (
                    f"target encode — {n_cats} categories is too many for one-hot. "
                    "Replace each category with mean target value "
                    "(use cross-fold means to reduce leakage risk)"
                )
                confidence = MEDIUM
                reasoning = (
                    f"{n_cats} categories would create {n_cats} one-hot columns. "
                    "Target encoding captures price signal in one column. "
                    "Risk: leakage if not done carefully with CV folds."
                )
            elif n_cats <= 2:
                strategy = "binary"
                suggestion = "label encode as 0/1 — binary feature"
                confidence = HIGH
                reasoning = "Only 2 categories — simple binary encoding is sufficient."
            else:
                strategy = "one_hot"
                suggestion = (
                    f"one-hot encode — {n_cats} categories is manageable "
                    f"({n_cats} new columns)"
                )
                confidence = HIGH
                reasoning = (
                    f"{n_cats} categories creates {n_cats} binary columns. "
                    "No ordinal assumption. Works well for nominal features."
                )

            results.append({
                "feature":    col,
                "n_categories": n_cats,
                "top_category_pct": round(float(top_freq * 100), 1),
                "signal":     f"{n_cats} unique categories",
                "strategy":   strategy,
                "suggestion": suggestion,
                "confidence": confidence,
                "reasoning":  reasoning,
            })

        return results

    def _analyze_outliers(self) -> list:
        """
        Detect potential outliers using IQR method.

        Signal:     count of values beyond 3×IQR
        Suggestion: investigate — not automatically remove
        Note:       Outliers may be legitimate data points.
                    Tree models are robust; linear models are sensitive.
        """
        results = []

        for col in self._numeric:
            data = self.df[col].dropna()
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            outlier_mask = (data < lower) | (data > upper)
            n_outliers = int(outlier_mask.sum())

            if n_outliers > 0:
                results.append({
                    "feature":   col,
                    "n_outliers": n_outliers,
                    "pct":       round(n_outliers / len(data) * 100, 2),
                    "signal":    f"{n_outliers} values beyond 3×IQR",
                    "suggestion": (
                        "investigate before removing — "
                        "outliers may be legitimate. "
                        "For linear models: consider removing. "
                        "For tree models: usually safe to keep."
                    ),
                    "confidence": MEDIUM,
                    "reasoning": (
                        "IQR method flags statistical outliers but cannot "
                        "determine if they are data errors or genuine edge cases. "
                        "Domain knowledge required."
                    ),
                })

        results.sort(key=lambda x: x["n_outliers"], reverse=True)
        return results

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _correlation_strength(self, r: float) -> str:
        if r >= 0.7:   return "very strong"
        if r >= 0.5:   return "strong"
        if r >= 0.3:   return "moderate"
        if r >= 0.1:   return "weak"
        return "very weak"

    def _print_summary(self):
        """Print a concise console summary after run()."""
        m = self.results["meta"]
        t = self.results["target"]
        print(f"📊 Dataset: {m['n_samples']} samples × {m['n_features']} features")
        print(f"   Numeric: {m['n_numeric']}  |  Categorical: {m['n_categorical']}")
        print(f"   Problem type: {m['problem_type']}")
        print()
        print(f"🎯 Target ({self.target}):")
        print(f"   {t['signal']['skewness']:.3f} skewness → [{t['confidence']}] {t['suggestion']}")
        print()

        missing = self.results["missing"]
        if missing:
            print(f"🔍 Missing values: {len(missing)} group(s) detected")
            for m_item in missing[:3]:
                cols_str = ", ".join(m_item["columns"][:3])
                if len(m_item["columns"]) > 3:
                    cols_str += f" +{len(m_item['columns'])-3} more"
                print(f"   [{m_item['confidence']}] {cols_str}: {m_item['suggestion'][:60]}...")
        else:
            print("✅ No missing values detected")
        print()

        top_corr = self.results["correlations"][:5]
        if top_corr:
            print(f"📈 Top correlations with {self.target}:")
            for c in top_corr:
                print(f"   {c['feature']:<20} r = {c['correlation']:>6.3f}  ({c['strength']})")
        print()

        multi = self.results["multicollinearity"]
        if multi:
            print(f"⚠️  Multicollinearity: {len(multi)} high-correlation pair(s)")
            for pair in multi[:3]:
                print(f"   {pair['feature_1']} ↔ {pair['feature_2']}: r={pair['correlation']}")
        print()

        skewed = self.results["skewed_features"]
        print(f"📐 Skewed features: {len(skewed)} need log1p transform")
        print()
        print("✅ Analysis complete. Call .summary() or pass to Reporter.")

    def _format_section(self, title: str, data: dict) -> str:
        lines = [f"\n--- {title} ---"]
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def _format_missing_summary(self) -> str:
        lines = ["\n--- MISSING VALUES ---"]
        for item in self.results["missing"]:
            lines.append(f"  [{item['confidence'].upper()}] {item['columns']}")
            lines.append(f"    Signal:     {item['signal']}")
            lines.append(f"    Suggestion: {item['suggestion']}")
            lines.append(f"    Reasoning:  {item['reasoning']}")
        return "\n".join(lines)

    def _format_correlation_summary(self) -> str:
        lines = ["\n--- TOP CORRELATIONS ---"]
        for item in self.results["correlations"][:10]:
            lines.append(f"  {item['feature']:<25} {item['signal']}  ({item['strength']})")
        return "\n".join(lines)

    def _format_multicollinearity_summary(self) -> str:
        lines = ["\n--- MULTICOLLINEARITY ---"]
        if not self.results["multicollinearity"]:
            lines.append("  None detected above threshold (r > 0.8)")
        for item in self.results["multicollinearity"]:
            lines.append(f"  [{item['confidence'].upper()}] {item['feature_1']} ↔ {item['feature_2']}")
            lines.append(f"    Signal:     {item['signal']}")
            lines.append(f"    Suggestion: {item['suggestion']}")
        return "\n".join(lines)

    def _format_skewness_summary(self) -> str:
        lines = ["\n--- SKEWED FEATURES ---"]
        for item in self.results["skewed_features"][:10]:
            lines.append(f"  [{item['confidence'].upper()}] {item['feature']:<25} {item['signal']}")
        if len(self.results["skewed_features"]) > 10:
            lines.append(f"  ... and {len(self.results['skewed_features'])-10} more")
        return "\n".join(lines)

    def _format_encoding_summary(self) -> str:
        lines = ["\n--- ENCODING RECOMMENDATIONS ---"]
        by_strategy = {}
        for item in self.results["encoding"]:
            s = item["strategy"]
            by_strategy.setdefault(s, []).append(item["feature"])
        for strategy, cols in by_strategy.items():
            lines.append(f"  {strategy.upper()} ({len(cols)}): {', '.join(cols[:5])}"
                         + (f" +{len(cols)-5} more" if len(cols) > 5 else ""))
        return "\n".join(lines)

    def _format_outlier_summary(self) -> str:
        lines = ["\n--- OUTLIERS ---"]
        if not self.results["outliers"]:
            lines.append("  None detected (3×IQR threshold)")
        for item in self.results["outliers"][:5]:
            lines.append(f"  {item['feature']:<25} {item['signal']} ({item['pct']}%)")
        return "\n".join(lines)
