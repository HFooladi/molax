#!/usr/bin/env python3
"""Download the ESOL (Delaney) dataset from MoleculeNet.

The ESOL dataset contains 1,128 molecules with measured aqueous solubility values.
This is a standard benchmark dataset for molecular property prediction.

Reference:
    Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility Directly from
    Molecular Structure. Journal of Chemical Information and Computer Sciences,
    44(3), 1000-1005.

Usage:
    python scripts/download_esol.py
"""

import urllib.request
from pathlib import Path

import pandas as pd

ESOL_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
)
OUTPUT_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "esol.csv"


def download_esol():
    """Download and process the ESOL dataset."""
    print(f"Downloading ESOL dataset from {ESOL_URL}...")

    # Download the file
    urllib.request.urlretrieve(ESOL_URL, OUTPUT_FILE.with_suffix(".raw.csv"))

    # Load and process the data
    df = pd.read_csv(OUTPUT_FILE.with_suffix(".raw.csv"))

    # The original file has columns: Compound ID, ESOL predicted log solubility,
    # Minimum Degree, Molecular Weight, Number of H-Bond Donors,
    # Number of Rings, Number of Rotatable Bonds, Polar Surface Area,
    # measured log solubility in mols per litre, smiles

    # Rename columns for consistency with molax conventions
    processed_df = pd.DataFrame(
        {
            "smiles": df["smiles"],
            "property": df["measured log solubility in mols per litre"],
        }
    )

    # Save processed data
    processed_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(processed_df)} molecules to {OUTPUT_FILE}")

    # Clean up raw file
    OUTPUT_FILE.with_suffix(".raw.csv").unlink()

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"  Number of molecules: {len(processed_df)}")
    print(
        f"  Property range: [{processed_df['property'].min():.2f}, "
        f"{processed_df['property'].max():.2f}]"
    )
    print(f"  Property mean: {processed_df['property'].mean():.2f}")
    print(f"  Property std: {processed_df['property'].std():.2f}")


if __name__ == "__main__":
    download_esol()
