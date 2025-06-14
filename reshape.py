from copy import copy

import nibabel as nib
import numpy as np
from nibabel import Nifti1Image

from pastrycutter.select import trim_bb_px_to_nii


def crop_to_bb_mm(nii: Nifti1Image, bb):
    """Crop a NIfTI image to a physical bounding box

    Since this is a crop, not a reslice, the actual crop will be to the pixel grid
    bounding box that minimally encloses the provided physical bounding box.

    """
    bb = np.vstack(
        [
            bb,
            [
                1,
                1,
            ],
        ]
    )
    i_aff_w = np.linalg.inv(nii.affine)
    bb_px = i_aff_w @ bb
    bb_px = np.vstack(
        [
            np.floor(np.min(bb_px[:-1], axis=1)).astype(int),
            np.ceil(np.max(bb_px[:-1], axis=1)).astype(int),
        ]
    ).T
    bb_px = trim_bb_px_to_nii(nii, bb_px)
    cropped = crop_to_bb_px(nii, bb_px)
    return cropped


def crop_to_bb_px(nii: Nifti1Image, bb):
    """Crop a NIfTI image to a pixel grid bounding box

    :param nii: Image to crop.

    :param bb: Array of (min, max) integer pixel positions in order of i, j,
    k specifying the bounding box for the crop.

    """
    # Get uncropped image info
    affine = nii.affine.copy()
    img = nii.get_fdata().copy()
    # Adjusting bounding box by padding
    bb = [list(x) for x in bb]
    bb = trim_bb_px_to_nii(nii, bb)
    # Crop the image
    img = img[
        bb[0][0] : (bb[0][1] + 1), bb[1][0] : (bb[1][1] + 1), bb[2][0] : (bb[2][1] + 1)
    ]
    # Shift the origin
    origin = np.dot(nii.affine, np.array([bb[0][0], bb[1][0], bb[2][0], 1]))
    affine[:, 3] = origin
    # Create new NIfTI image
    cropped = nib.Nifti1Image(img, affine)
    return cropped
