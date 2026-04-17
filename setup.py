"""Minimal setup for pip install -e . (editable mode)."""

from setuptools import setup, find_packages

setup(
    name="adattt",
    version="0.1.0",
    description="Efficient TTT: Adaptive Test-Time Training for VQA",
    author="Aishwarya Reddy Chinthalapudi, Yugesh Reddy Sappidi, Aryan Shetty",
    packages=find_packages(exclude=["tests*", "scripts*", "gpu*", "demo*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "transformers>=4.36",
        "datasets",
        "pillow",
        "pyyaml",
        "numpy",
        "matplotlib",
        "scipy",
        "tqdm",
    ],
)
