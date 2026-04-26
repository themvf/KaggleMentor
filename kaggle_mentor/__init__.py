"""
kaggle_mentor
=============
An intelligence layer that converts raw data into a structured,
confidence-rated modeling strategy.

Not a modeling framework. Not a pipeline wrapper.
The layer BEFORE modeling — EDA, insights, recommendations.

Usage:
    from kaggle_mentor import Analyzer, Reporter, Preprocessor

    a = Analyzer(df, target="SalePrice")
    a.run()

    r = Reporter(a.results)
    r.save("EDA_REPORT.md")
"""

from .analyzer import Analyzer
from .reporting import Reporter
from .preprocessing import Preprocessor
from .features import FeatureHook
from .pipeline import Pipeline

__version__ = "0.1.0"
__all__ = ["Analyzer", "Reporter", "Preprocessor", "FeatureHook", "Pipeline"]
