"""Reslice NIfTI image to be isotropic and aligned with world axes."""

from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

# Debugging aids:
# np.set_printoptions(suppress=True, precision=3)


def read_itk_transform_txt(pth):
    """Read an Insight Transform File V1.0 text file

    Returns an augmented affine matrix representing the transform in RAS+ coordinates.
    Note that ITK uses LPS+ so this requires conversion.

    """
    with open(pth, "r") as f:
        fmt = f.readline().lstrip("#").rstrip()
        if not fmt == "Insight Transform File V1.0":
            raise ValueError(
                f"File {pth} does not declare 'Insight Transform File V1.0' on its first line; is it really an ITK Transform File?"
            )
        order = int(f.readline().removeprefix("#Transform ").rstrip())
        transform_type = f.readline().removeprefix("Transform: ").rstrip()
        if not transform_type == "MatrixOffsetTransformBase_double_3_3":
            raise ValueError(
                f"This function does not currently support reading '{transform_type}' trasnforms."
            )
        p = [
            float(v) for v in f.readline().removeprefix("Parameters: ").rstrip().split()
        ]
        fixed = [
            float(v)
            for v in f.readline().removeprefix("FixedParameters: ").rstrip().split()
        ]
    # Slower but clearer way:
    #
    # ras_A_lps = np.array([[-1, 0, 0, 0],
    #                       [0, -1, 0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]])
    # itk_affine = np.array(
    #     [
    #         [p[0], p[1], p[2], p[11]],
    #         [p[3], p[4], p[5], p[10]],
    #         [p[6], p[7], p[8], p[9]],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # nifti_affine = ras_A_lps @ itk_affine @ ras_A_lps.T
    nifti_affine = np.array(
        [
            [p[0], p[1], -p[2], -p[11]],
            [p[3], p[4], -p[5], -p[10]],
            [-p[6], -p[7], p[8], p[9]],
            [0, 0, 0, 1],
        ]
    )
    return nifti_affine


# x translations are handled correctly when the rotation is about x


def reslice_ras(nii, sz, interp="spline", affine=np.eye(4)):
    """Reslice NIfTI image to be isotropic with RAS+ axes.

    :param nii: Nibabel NIfTI image object.

    :param sz: Voxel size of output image, in mm.

    :param interp: "spline" (default), "linear", or "nearest".  Interpolation method to
    use when reslicing the input image.

    :param affine: (Optional) Augmented affine matrix defining the output image's RAS
    axes in the coordinate system of the input image.

    Reslice the input image such that the voxels are isotropic with the input voxel size
    and the array axes are RAS-aligned.  The domain (bounding box) of the output image
    is set to fully cover the image data, without clipping.  If a transform file is
    provided, it is used to calculate the voxel coordinates prior to reslicing.

    """

    def corner_indices(shape):
        """Return array of indices for corner voxels"""
        corners = np.array(
            [
                [0, 0, 0, 1],
                [shape[0] - 1, 0, 0, 1],
                [shape[0] - 1, shape[1] - 1, 0, 1],
                [0, shape[1] - 1, 0, 1],
                [0, 0, shape[2] - 1, 1],
                [shape[0] - 1, 0, shape[2] - 1, 1],
                [shape[0] - 1, shape[1] - 1, shape[2] - 1, 1],
                [0, shape[1] - 1, shape[2] - 1, 1],
            ]
        ).T
        return corners

    # Define original image data as im0 and resliced data as im1.
    #
    # Define the following coordinate system prefixes:
    # w := World axes in input image [mm]
    # x := World axes in output image, after transform [mm]
    # i := Voxel indices, 0-indexed, of original image array
    # j := Voxel indices, 0-indexed, of RAS-aligned array
    x_aff_w = affine
    # Figure out how big the output image voxel grid needs to be
    i_im0_corners = corner_indices(nii.shape)
    w_im0_corners = nii.affine @ i_im0_corners
    x_im0_corners = x_aff_w @ w_im0_corners
    x_im0_bbmax = np.max(x_im0_corners, axis=1)[:3]
    x_im0_bbmin = np.min(x_im0_corners, axis=1)[:3]
    im1_shape = (np.ceil(x_im0_bbmax - x_im0_bbmin) / sz).astype(int)
    i_im0 = np.vstack(
        [np.reshape(np.indices(nii.shape), (3, -1)), np.ones([1, np.prod(nii.shape)])]
    )
    # Compute the affine matrix for the resliced image.
    w_aff_x = np.linalg.inv(x_aff_w)
    # Define the output image's voxel grid
    j_im1 = np.vstack(
        [
            np.reshape(np.indices(im1_shape), (3, -1)),
            np.ones([1, np.prod(im1_shape)]),
        ]
    )
    # Define the voxel → physical transform for the output image.  The affine will
    # always be a scaling matrix, since by design the output image's voxel grid is
    # supposed to be parellel to the anatomic axes.  Since the trasnform between the
    # input image's physical coordinates and the output image's physical coordinates is
    # fully determined elsewhere (as a function parameter), the offset determines the
    # viewport that the output image has over the input image.  We choose to make the
    # image centers coincide, and the voxel grid size was chosen above so that this will
    # produce full coverage of the input image.
    i_im0_center = np.hstack([(np.array(nii.shape) - 1) / 2, 1])
    j_im1_center = np.hstack([(np.array(im1_shape) - 1) / 2, 1])
    x_im0_center = x_aff_w @ nii.affine @ i_im0_center
    w_im0_center = w_aff_x @ nii.affine @ nii.affine @ i_im0_center
    x_aff_j_scaling = np.array(
        [[sz, 0, 0, 0], [0, sz, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]]
    )
    o = x_im0_center - x_aff_j_scaling @ j_im1_center
    x_aff_j = np.array(
        [[sz, 0, 0, o[0]], [0, sz, 0, o[1]], [0, 0, sz, o[2]], [0, 0, 0, 1]]
    )
    # Reslice the image
    i_aff_w = np.linalg.inv(nii.affine)
    i_im1 = i_aff_w @ w_aff_x @ x_aff_j @ j_im1
    im0 = np.asarray(nii.dataobj)
    order = {"spline": 3, "linear": 1, "nearest": 0}[interp]
    im1 = map_coordinates(
        im0,
        i_im1[:3, :].reshape([3] + im1_shape.tolist()),
        mode="constant",
        order=order,
    )
    resliced = nib.Nifti1Image(im1, x_aff_j)
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
