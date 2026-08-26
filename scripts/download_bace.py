#!/usr/bin/env python3
"""Download the BACE-1 dataset from MoleculeNet.

BACE-1 (beta-secretase 1) is a protease implicated in the production of
amyloid-beta plaques, and one of the most heavily pursued Alzheimer's disease
targets. The dataset contains 1,513 inhibitors with measured binding affinity,
reported here as pIC50 (higher = more potent).

Unlike a physicochemical benchmark such as ESOL, this is a real medicinal
chemistry SAR series, which makes it a realistic testbed for active learning
under an assay budget.

Reference:
    Subramanian, G., Ramsundar, B., Pande, V., & Denny, R. A. (2016).
    Computational Modeling of Beta-Secretase 1 (BACE-1) Inhibitors Using
    Ligand Based Approaches. Journal of Chemical Information and Modeling,
    56(10), 1936-1949.

Usage:
    python scripts/download_bace.py
"""

import urllib.request
from pathlib import Path

import pandas as pd

BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "bace.csv"


def download_bace():
    """Download and process the BACE-1 dataset."""
    print(f"Downloading BACE-1 dataset from {BACE_URL}...")

    urllib.request.urlretrieve(BACE_URL, OUTPUT_FILE.with_suffix(".raw.csv"))

    df = pd.read_csv(OUTPUT_FILE.with_suffix(".raw.csv"))

    # The original file carries SMILES in 'mol' plus ~590 precomputed
    # descriptor columns. molax featurizes from SMILES, so keep only the
    # structure and the endpoint.
    processed_df = pd.DataFrame(
        {
            "smiles": df["mol"],
            "property": df["pIC50"],
        }
    ).dropna()

    processed_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(processed_df)} molecules to {OUTPUT_FILE}")

    OUTPUT_FILE.with_suffix(".raw.csv").unlink()

    print("\nDataset Statistics:")
    print(f"  Number of molecules: {len(processed_df)}")
    print(
        f"  pIC50 range: [{processed_df['property'].min():.2f}, "
        f"{processed_df['property'].max():.2f}]"
    )
    print(f"  pIC50 mean: {processed_df['property'].mean():.2f}")
    print(f"  pIC50 std: {processed_df['property'].std():.2f}")


if __name__ == "__main__":
    download_bace()
