from setuptools import find_packages, setup

setup(
    name="ALPHA",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pytest>=6.0.0",
        "pytest-cov>=2.0.0",
    ],
    python_requires=">=3.8",
)
