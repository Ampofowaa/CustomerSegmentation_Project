#!/usr/bin/env python
"""
CLI entrypoint for training.

Usage:
    python scripts/train.py
    python scripts/train.py --n-clusters 6

Run from the project root (so `config` and `src` resolve on the path), e.g.:
    cd customer_segmentation && python scripts/train.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from the project root without
# installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.pipeline import run_training_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the customer segmentation model")
    parser.add_argument("--n-clusters", type=int, default=None, help="Override config.N_CLUSTERS")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    n_clusters = args.n_clusters or config.N_CLUSTERS
    metrics = run_training_pipeline(n_clusters=n_clusters)

    print("\nTraining complete.")
    print(f"  Rows trained on : {metrics['n_rows_trained']}")
    print(f"  Clusters (k)    : {metrics['n_clusters']}")
    print(f"  Silhouette      : {metrics['silhouette_score']:.4f}")
    print(f"  Cluster sizes   : {metrics['cluster_sizes']}")


if __name__ == "__main__":
    main()
