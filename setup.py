"""Frakt setup - SMC trading engine."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="frakt",
    version="0.1.0",
    author="r464r64r",
    description="SMC-based trading engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/r464r64r/Frakt",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
    ],
    extras_require={
        "backtesting": [
            "vectorbt>=0.26",
        ],
        "exchanges": [
            "ccxt>=4.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
)
