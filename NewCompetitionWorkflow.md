# New Competition Workflow

A step-by-step guide for starting any Kaggle competition using `kaggle_mentor`.

---

## One-Time Setup

Install the library directly from GitHub (only needed once per machine):

```bash
pip install git+https://github.com/themvf/KaggleMentor.git
```

---

## Every New Competition — 5 Steps

### Step 1: Create a Folder and Download Data

```bash
mkdir MyCompetition
cd MyCompetition
# Download data from Kaggle and drop train.csv / test.csv in the folder
```

### Step 2: Run the Analyzer

```python
import pandas as pd
from kaggle_mentor import Analyzer, Reporter

train = pd.read_csv('train.csv')

a = Analyzer(train, target='SalePrice')  # replace with your target column
a.run()

Reporter(a.results, target='SalePrice', competition='My Competition').save('EDA_REPORT.md')
```

**What you get:**

- `EDA_REPORT.md` — full analysis with skewness, missing patterns, correlations, encoding recommendations
- Every finding includes a **confidence level** (🟢 High / 🟡 Medium / 🔴 Low) and **reasoning**
- **Signal** (objective) is always separated from **Suggestion** (heuristic)

### Step 3: Write Your Feature Function

This is the only competition-specific step. Use the `EDA_REPORT.md` findings as your guide:

```python
def my_features(df):
    df = df.copy()

    # Example: House Prices
    df['TotalSF']     = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge']    = df['YrSold'] - df['YearBuilt']
    df['QualityArea'] = df['OverallQual'] * df['GrLivArea']

    # Example: Titanic
    # df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    # df['Title']      = df['Name'].str.extract(r' ([A-Za-z]+)\.')

    return df
```

> Feature engineering is always domain-specific — the library cannot automate this for you.  
> The Analyzer tells you *what* to look at. You decide *what to build*.

You can also use the built-in generic utilities:

```python
from kaggle_mentor.features import add_binary_indicators, add_interaction

df = add_binary_indicators(df, ['PoolArea', 'GarageArea', 'Fireplaces'])
df = add_interaction(df, 'OverallQual', 'GrLivArea', name='QualityArea')
```

### Step 4: Prepare Data

```python
from kaggle_mentor import Pipeline

pipeline = Pipeline(
    train_path='train.csv',
    test_path='test.csv',
    target='SalePrice',
    feature_fn=my_features,
    scale=False,           # False for XGBoost/tree models
                           # True for Ridge/Lasso/linear models
    competition='My Competition'
)

X_train, X_test, y = pipeline.prepare()
```

**What happens automatically:**

| Step | What the library does |
|------|-----------------------|
| Target transform | Log1p if skewed (based on Analyzer) |
| Missing values | Fills structural absences with `None`/`0`, imputes the rest |
| Ordinal encoding | Quality/condition columns mapped to integers |
| Target encoding | High-cardinality columns replaced with mean target |
| One-hot encoding | Remaining categoricals |
| Log1p transform | Applied to skewed numeric features |
| Scaling | StandardScaler if `scale=True` |
| Train/test alignment | Test columns aligned to training columns |

### Step 5: Train Your Model and Submit

```python
import numpy as np
from xgboost import XGBRegressor   # or any model you choose

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y)

pipeline.generate_submission(
    model.predict(X_test),
    filename='submission.csv',
    inverse_transform=True   # True if target was log-transformed (regression)
                             # False for classification
)
```

---

## What the Library Handles vs What You Write

| Task | Handled by library | You write |
|------|--------------------|-----------|
| EDA + report | ✅ Automatic | — |
| Missing value strategy | ✅ Automatic | — |
| Encoding decisions | ✅ Automatic | — |
| Skew detection + transforms | ✅ Automatic | — |
| Train/test consistency | ✅ Automatic | — |
| Submission file + sanity check | ✅ Automatic | — |
| Feature engineering | — | ✅ Always custom |
| Model choice + hyperparameters | — | ✅ Always yours |
| Domain knowledge | — | ✅ Always yours |

---

## Time Saved vs Starting from Scratch

| Task | From scratch | With kaggle_mentor |
|------|--------------|--------------------|
| EDA + report | 2–3 hours | ~2 minutes |
| Missing value strategy | 30 min | Automatic |
| Encoding decisions | 30 min | Automatic |
| Preprocessing pipeline | 1–2 hours | ~10 lines |
| Feature engineering | Same | Same |
| Modeling | Same | Same |

---

## Updating the Library

When a new version is pushed to GitHub:

```bash
pip install --upgrade git+https://github.com/themvf/KaggleMentor.git
```

---

## Full Example (House Prices)

```python
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from kaggle_mentor import Analyzer, Reporter, Pipeline

# 1. Analyze
train = pd.read_csv('train.csv')
a = Analyzer(train, target='SalePrice').run()
Reporter(a.results, target='SalePrice', competition='House Prices').save('EDA_REPORT.md')

# 2. Feature function
def my_features(df):
    df = df.copy()
    df['TotalSF']      = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge']     = df['YrSold'] - df['YearBuilt']
    df['TotalBath']    = (df['FullBath'] + 0.5 * df['HalfBath'] +
                          df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    df['QualityArea']  = df['OverallQual'] * df['GrLivArea']
    df['WasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    return df

# 3. Prepare
pipeline = Pipeline('train.csv', 'test.csv', 'SalePrice',
                    feature_fn=my_features, competition='House Prices')
X_train, X_test, y = pipeline.prepare()

# 4. Train
model = XGBRegressor(n_estimators=1000, learning_rate=0.05,
                     max_depth=4, random_state=42)
model.fit(X_train, y)

# 5. Submit
pipeline.generate_submission(model.predict(X_test), filename='submission.csv')
```

---

*For full documentation see [README.md](README.md)*
