from setuptools import setup, find_packages

setup(
    name="tseries_patterns",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
) 