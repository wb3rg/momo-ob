from setuptools import setup, find_packages

setup(
    name="tseries_patterns",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.29.0",
        "ccxt>=4.1.13",
        "requests>=2.31.0",
        "python-binance>=1.0.19",
    ],
    python_requires=">=3.8",
) 