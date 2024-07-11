"""Utility functions"""

from pathlib import Path

import numpy as np
import pandas as pd


def augmented(arr):
    """Return array of points in augmented form

    Add ones at the end of axis 0.

    """
    return np.concatenate([arr, np.ones((1, *arr.shape[1:]))], axis=0)


def nii_name(fname):
    """Return NIfTI filename without .nii.gz"""
    return Path(fname).name.removesuffix(".gz").removesuffix(".nii")


def transform_markers(markers: pd.DataFrame, affine, colnames=("x", "y", "z")):
    """Apply an affine transform to a table of fiducial markers

    :param markers: A table of fiducial markers.

    :param affine: 4x4 affine transformation matrix

    :param colnames: Column names that correspond to the x, y, and z values, in that
    order.

    The row index of `markers` will be preserved.

    """
    return pd.DataFrame(
        data=(affine @ augmented(markers[list(colnames)].values.T))[:3].T,
        index=markers.index,
        columns=colnames,
    )
