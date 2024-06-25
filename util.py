"""Utility functions"""

from pathlib import Path

import numpy as np


def augmented(arr):
    """Return array of points in augmented form

    Add ones at the end of axis 1.

    """
    return np.concatenate([arr, np.ones((1, *arr.shape[1:]))], axis=0)


def nii_name(fname):
    """Return NIfTI filename without .nii.gz"""
    return Path(fname).name.removesuffix(".gz").removesuffix(".nii")
