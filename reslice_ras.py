"""Reslice NIfTI image to be isotropic and aligned with world axes."""

from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import map_coordinates


def reslice_ras(nii, sz, interp="spline"):
    """Reslice NIfTI image to be isotropic with RAS+ axes.

    sz := voxel size in mm

    The returned image has the same RAS-aligned bounding box as the
    original image data.

    """
    # Define original image data as im0 and resliced data as im1.
    #
    # Define the following coordinate systems:
    # w := World axes [mm]
    # i := Voxel indices, 0-indexed, of original image array
    # j := Voxel indices, 0-indexed, of RAS-aligned array
    i_im0_corners = np.array(
        [
            [0, 0, 0, 1],
            [nii.shape[0], 0, 0, 1],
            [nii.shape[0], nii.shape[1], 0, 1],
            [0, nii.shape[1], 0, 1],
            [0, 0, nii.shape[2], 1],
            [nii.shape[0], 0, nii.shape[2], 1],
            [nii.shape[0], nii.shape[1], nii.shape[2], 1],
            [0, nii.shape[1], nii.shape[2], 1],
        ]
    ).T
    w_im0_corners = nii.affine @ i_im0_corners
    w_im0_bbmax = np.max(w_im0_corners, axis=1)[:3]
    w_im0_bbmin = np.min(w_im0_corners, axis=1)[:3]
    im1_shape = np.ceil((w_im0_bbmax - w_im0_bbmin) / sz).astype(int)
    i_im0 = np.vstack(
        [np.reshape(np.indices(nii.shape), (3, -1)), np.ones([1, np.prod(nii.shape)])]
    )
    # Compute the affine matrix for the resliced image.  Define the
    # offset so that the image centers coincide.
    i_im0_center = (np.array(nii.shape) - 1) / 2
    w_im0_center = (nii.affine @ np.hstack([i_im0_center, [1]]))[:3]
    j_im1_center = (np.array(im1_shape) - 1) / 2
    o = w_im0_center - sz * np.eye(3) @ j_im1_center
    w_aff_j = np.array(
        [[sz, 0, 0, o[0]], [0, sz, 0, o[1]], [0, 0, sz, o[2]], [0, 0, 0, 1]]
    )
    # Reslice the image
    j_im1 = np.vstack(
        [
            np.reshape(np.indices(im1_shape), (3, -1)),
            np.ones([1, np.prod(im1_shape)]),
        ]
    )
    i_aff_w = np.linalg.inv(nii.affine)
    i_im1 = i_aff_w @ w_aff_j @ j_im1
    im0 = np.asarray(nii.dataobj)
    if interp == "spline":
        order = 3
    elif interp == "nearest":
        order = 0
    im1 = map_coordinates(
        im0, i_im1[:3, :].reshape([3] + im1_shape.tolist()), mode="constant",
        order=order,
    )
    resliced = nib.Nifti1Image(im1, w_aff_j)
    return resliced


if __name__ == "__main__":
    desc = "Reslice NIfTI image to be isotropic and aligned with world axes."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("infile", help="Path of NIfTI file to reslice")
    parser.add_argument("outfile", help="Path to save resliced NIfTI file")
    parser.add_argument(
        "--voxel_size",
        type=float,
        dest="voxel_size",
        default=None,
        help="Desired voxel size in mm",
    )
    parser.add_argument(
        "--interp",
        choices=["spline", "nearest"],
        dest="interp",
        default="spline",
        help="Interpolation mode.  Supports cubic spline (default) and nearest neighbor, which is useful for resampling segmentation (label) images.",
    )
    args = parser.parse_args()
    original = nib.load(args.infile)
    resliced = reslice_ras(original, args.voxel_size, interp=args.interp)
    nib.save(resliced, args.outfile)
