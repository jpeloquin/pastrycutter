"""Utility functions"""

from pathlib import Path

import numpy as np
import pandas as pd


def augmented(arr):
    """Return array of points in augmented form

    Add ones at the end of axis 0.

    """
    return np.concatenate([arr, np.ones((1, *arr.shape[1:]))], axis=0)


def itk_directions_from_affine(affine):
    """Return ITK (LPS+) direction cosines, spacing, and origin from RAS+ affine"""
    A = lps_header_affine(affine)
    itk_spacing = np.linalg.norm(A[:3, :3], axis=0)
    itk_origin = A[:3, 3]
    itk_direction = A[:3, :3] / itk_spacing
    return itk_direction, itk_spacing, itk_origin


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


def lps_header_affine(A):
    """Convert RAS+ header affine to LPS+

    Converting a header affine (ijk → RAS x) is *not* the same as converting a
    transformation affine (RAS x → RAS y).

    """
    return np.array(
        [
            [-A[0, 0], -A[0, 1], -A[0, 2], -A[0, 3]],
            [-A[1, 0], -A[1, 1], -A[1, 2], -A[1, 3]],
            [A[2, 0], A[2, 1], A[2, 2], A[2, 3]],
            [0, 0, 0, 1],
        ]
    )


def lps_transformation_affine(A):
    """Convert RAS+ affine to LPS+

    This function will also convert an RAS+ affine to LPS+

    """
    # Slower but clearer way to convert LPS → RAS:
    #
    # ras_A_lps = np.array([[-1, 0, 0, 0],
    #                       [0, -1, 0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]])
    # itk_affine = np.array(
    #     [
    #         [p[0], p[1], p[2], p[9]],
    #         [p[3], p[4], p[5], p[10]],
    #         [p[6], p[7], p[8], p[11]],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # A = ras_A_lps @ itk_affine @ ras_A_lps.T
    return np.array(
        [
            [A[0, 0], A[0, 1], -A[0, 2], -A[0, 3]],
            [A[1, 0], A[1, 1], -A[1, 2], -A[1, 3]],
            [-A[2, 0], -A[2, 1], A[2, 2], A[2, 3]],
            [0, 0, 0, 1],
        ]
    )


def ras_transformation_affine(A):
    """Convert LPS+ affine to RAS+

    Since A = lps_affine(lps_affine(A)), ras_affine just calls lps_affine(A).  The
    two functions are separate only to ensure the caller's intent is clear.

    """
    return lps_transformation_affine(A)
