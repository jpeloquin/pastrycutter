"""Utility functions"""
import numpy as np


def augmented(arr):
    """Return array of points in augmented form

    Add ones at the end of axis 1.

    """
    return np.vstack([arr, np.ones((1, arr.shape[1]))])
