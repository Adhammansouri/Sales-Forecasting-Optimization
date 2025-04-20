from setuptools import setup, find_packages

setup(
    name="sales_forecasting",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "xgboost==2.0.2",
        "tensorflow==2.13.0",
        "prophet==1.1.4",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "jupyter==1.0.0",
        "statsmodels==0.14.0",
        "plotly==5.17.0",
        "python-dotenv==1.0.0",
        "pytest==7.4.2"
    ]
) 