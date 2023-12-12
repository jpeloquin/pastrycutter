"""Functions to manipulate bounding boxes and other selections"""
from copy import copy

import numpy as np


def bb_corners(bbox):
    """Return bounding box corners

    :param bbox: Bounding box input as an array with axis 0 → x/y/z and axis 1 →
    min/max.

    """
    corners = np.array(
        [
            [bbox[0][0], bbox[1][0], bbox[2][0]],
            [bbox[0][1], bbox[1][0], bbox[2][0]],
            [bbox[0][1], bbox[1][1], bbox[2][0]],
            [bbox[0][0], bbox[1][1], bbox[2][0]],
            [bbox[0][0], bbox[1][0], bbox[2][1]],
            [bbox[0][1], bbox[1][0], bbox[2][1]],
            [bbox[0][1], bbox[1][1], bbox[2][1]],
            [bbox[0][0], bbox[1][1], bbox[2][1]],
        ]
    ).T
    return corners


def corner_indices(shape):
    """Return array of indices for corner voxels"""
    corners = np.array(
        [
            [0, 0, 0],
            [shape[0] - 1, 0, 0],
            [shape[0] - 1, shape[1] - 1, 0],
            [0, shape[1] - 1, 0],
            [0, 0, shape[2] - 1],
            [shape[0] - 1, 0, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, shape[2] - 1],
            [0, shape[1] - 1, shape[2] - 1],
        ]
    ).T
    return corners


def pad_bb(bb, pad=(0, 0, 0)):
    """Return bounding box expanded by pad"""
    if type(pad) is int:
        pad = [pad] * 3
    bb = np.array(copy(bb))
    bb[0][0] -= pad[0]
    bb[1][0] -= pad[1]
    bb[2][0] -= pad[2]
    bb[0][1] += pad[0]
    bb[1][1] += pad[1]
    bb[2][1] += pad[2]
    return bb


def trim_bb_px_to_nii(nii, bb):
    """Trim bb (px) to not exceed image boundaries"""
    bb = copy(bb)
    bb[0][0] = max([0, bb[0][0]])
    bb[1][0] = max([0, bb[1][0]])
    bb[2][0] = max([0, bb[2][0]])
    bb[0][1] = min([nii.shape[0], bb[0][1]])
    bb[1][1] = min([nii.shape[1], bb[1][1]])
    bb[2][1] = min([nii.shape[2], bb[2][1]])
    return bb
