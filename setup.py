from setuptools import setup, find_packages

setup(
    name="kaggle_mentor",
    version="0.1.0",
    description="An intelligence layer that converts raw data into a modeling strategy",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "seaborn>=0.11",
    ],
    extras_require={
        "xgboost": ["xgboost>=1.5"],
        "lightgbm": ["lightgbm>=3.3"],
    },
)
