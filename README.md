# kaggle_mentor

> The intelligence layer between raw data and modeling.

Not a modeling framework. Not a pipeline wrapper.  
The layer **before** modeling — EDA, insights, recommendations.

---

## What it does

```python
from kaggle_mentor import Analyzer, Reporter

a = Analyzer(df, target="SalePrice")
a.run()
```

```
📊 Dataset: 1,460 samples × 80 features
   Numeric: 36  |  Categorical: 43
   Problem type: regression

🎯 Target (SalePrice):
   1.881 skewness → [high] apply log transform (log1p recommended)

🔍 Missing values: 6 group(s) detected
   [high]   PoolQC, MiscFeature, Alley: fill with 'None'...
   [medium] GarageType, GarageQual, GarageCond +2 more: fill with 'None'/0...
   [low]    LotFrontage: impute with median or neighborhood-grouped median...

📈 Top correlations with SalePrice:
   OverallQual          r =  0.791  (very strong)
   GrLivArea            r =  0.709  (very strong)
   GarageCars           r =  0.640  (strong)

⚠️  Multicollinearity: 4 high-correlation pair(s)
   GarageArea ↔ GarageCars: r=0.882
   TotRmsAbvGrd ↔ GrLivArea: r=0.825

📐 Skewed features: 21 need log1p transform
```

---

## Design principles

- **Signal** = objective observation (skewness = 1.88)
- **Suggestion** = heuristic recommendation with confidence level
- **Never overconfident** — MNAR/MAR is inferred, not certified
- **Options, not directives** — especially for multicollinearity
- **You control the model** — we prepare data and get out of the way

---

## Install

```bash
pip install -e .
```

---

## Usage

### 1. Analyze only

```python
from kaggle_mentor import Analyzer, Reporter

a = Analyzer(train_df, target="SalePrice")
a.run()

r = Reporter(a.results, target="SalePrice", competition="House Prices")
r.save("EDA_REPORT.md")
```

### 2. Full pipeline with your own features

```python
from kaggle_mentor import Pipeline

def my_features(df):
    df = df.copy()
    df["TotalSF"]    = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"]   = df["YrSold"] - df["YearBuilt"]
    df["QualityArea"] = df["OverallQual"] * df["GrLivArea"]
    return df

pipeline = Pipeline(
    train_path="train.csv",
    test_path="test.csv",
    target="SalePrice",
    feature_fn=my_features,
    scale=False,          # True for linear models, False for XGBoost
)

X_train, X_test, y = pipeline.prepare()

# You control the model
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4)
model.fit(X_train, y)

pipeline.generate_submission(model.predict(X_test), filename="submission.csv")
```

### 3. Generic feature utilities

```python
from kaggle_mentor.features import add_binary_indicators, add_interaction, validate_features

df = add_binary_indicators(df, ["PoolArea", "GarageArea", "Fireplaces"])
df = add_interaction(df, "OverallQual", "GrLivArea", name="QualityArea")

report = validate_features(X_train)
```

---

## Architecture

```
kaggle_mentor/
├── analyzer.py       # EDA + structured insights (the core)
├── reporting.py      # Markdown report generator
├── preprocessing.py  # Missing values, encoding, scaling
├── features.py       # Hook system + generic utilities
└── pipeline.py       # Thin orchestration glue
```

**What's intentionally missing:**
- `models.py` — use sklearn/xgboost directly
- `ensemble.py` — use sklearn or build your own
- `tuning.py` — use GridSearchCV directly

---

## Output schema

```python
a.results = {
    "meta": { "n_samples": 1460, "problem_type": "regression", ... },
    "target": {
        "signal":     { "skewness": 1.881, "mean": 180921, ... },
        "suggestion": "apply log transform",
        "confidence": "high",
        "reasoning":  "Skewness of 1.88 is strongly non-normal..."
    },
    "missing": [
        {
            "columns":    ["PoolQC", "MiscFeature"],
            "pct":        99.5,
            "pattern":    "structural_absence",
            "suggestion": "fill with 'None'",
            "confidence": "high",
            "reasoning":  "99.5% missing strongly suggests..."
        },
        ...
    ],
    "correlations":      [...],
    "multicollinearity": [...],
    "skewed_features":   [...],
    "encoding":          [...],
    "outliers":          [...],
}
```

---

## Confidence levels

| Level | Meaning |
|-------|---------|
| 🟢 High | Strong statistical evidence, well-established heuristic |
| 🟡 Medium | Pattern detected but requires domain validation |
| 🔴 Low | Weak signal, investigate further before acting |

---

## Applying to a new competition

```python
# 1. Analyze
a = Analyzer(train, target="your_target").run()

# 2. Read the report
Reporter(a.results, target="your_target").save("EDA_REPORT.md")

# 3. Write your feature function based on domain knowledge
def my_features(df): ...

# 4. Prepare data
pipeline = Pipeline("train.csv", "test.csv", "your_target", feature_fn=my_features)
X_train, X_test, y = pipeline.prepare()

# 5. Train your model (you decide which one)
model = ...
model.fit(X_train, y)

# 6. Submit
pipeline.generate_submission(model.predict(X_test))
```
