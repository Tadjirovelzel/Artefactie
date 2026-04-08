"""
Loads ABP and CVP pressure signals from Excel files
Written for KT3401 - Assignment Artefact Detection.
"""

import numpy as np
import pandas as pd

def read_artefacts(filepath, fs):
    """
    Load ABP and CVP signals from an Excel file.

    The file is expected to have ABP in column 2 and CVP in column 3
    (0-indexed), with the first two rows containing metadata.
    """
    try:
        raw = pd.read_excel(filepath, sheet_name=0, header=None)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

    data = raw.iloc[2:, :].to_numpy()
    ABP  = pd.to_numeric(data[:, 1], errors="coerce")
    CVP  = pd.to_numeric(data[:, 2], errors="coerce")
    t    = np.arange(1 / fs, len(ABP) / fs + 1 / fs, 1 / fs)
    return t, ABP, CVP
