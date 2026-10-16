"""
Entrypoint for cortexhr_features build:
    python -m cortexhr_features.build
"""
from src.features.build_features import run_all

if __name__ == "__main__":
    run_all()

