from setuptools import setup, find_packages

setup(
    name="hydra-trading",
    version="0.1.0",
    description="Multi-Agent Reinforcement Learning Trading System",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.1.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
    ],
    extras_require={
        "directml": ["torch-directml>=0.2.0"],
        "test": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "hydra-train=scripts.train:main",
            "hydra-eval=scripts.evaluate:main",
            "hydra-export=scripts.export_to_vectorbt:main",
        ],
    },
)
