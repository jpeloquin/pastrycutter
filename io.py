from copy import copy
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from pandas import DataFrame
from scipy.io import loadmat, savemat


def _affine_from_itk(p, cor=None):
    """Return affine matrix from ITK parameters list

    :param p: List-like of parameters, 12 values.  The first 9 define the upper left
    3×3 part of the affine matrix and the last 3 define the translation vector.

    :param cor: Center of rotation, 3 values.


    """
    A = np.array(
        [
            [p[0], p[1], p[2], p[9]],
            [p[3], p[4], p[5], p[10]],
            [p[6], p[7], p[8], p[11]],
            [0, 0, 0, 1],
        ]
    )
    if cor is not None:
        return _affine_with_cor(A, cor)


def _affine_convert_ras_lps(A):
    """Convert LPS+ affine to RAS+ or vice versa"""
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


def _affine_with_cor(A, cor):
    """Update affine matrix with center of rotation

    :param A: 4×4 affine matrix (augmented form).

    :param cor: Center of rotation, 3 values.

    """
    A = copy(A)
    R = A[:3, :3]
    cor_Δ = cor - R @ cor
    A[:3, -1] = A[:3, -1] + cor_Δ
    return A


def read_itk_affine(pth: Union[str, Path]):
    """Read ITK or ANTs .mat affine in RAS+ coords"""
    pth = Path(pth)
    if pth.suffix == ".mat":
        return read_itk_affine_mat(pth)
    else:
        return read_itk_transform_txt(pth)


def read_itk_affine_mat(pth: Union[str, Path]):
    """Read ANTs affine in RAS+ coords

    ANTs uses LPS+; the returned affine is in RAS+.

    """
    pth = str(pth)
    ants_affine = loadmat(pth)
    if "AffineTransform_double_3_3" in ants_affine:
        return read_itk_affine_mat_3d(pth)
    elif "AffineTransform_double_2_2" in ants_affine:
        return read_itk_affine_mat_2d(pth)
    else:
        raise NotImplementedError(f"Affine transform format not supported.")


def read_itk_affine_mat_2d(pth: Union[str, Path]):
    """Read 2D ANTs affine

    ANTs uses LPS+, but that doesn't mean anything in 2D, so the coordinates are
    passed through as-is.

    """
    ants_affine = loadmat(pth)
    a = ants_affine["AffineTransform_double_2_2"]
    # fmt: off
    affine = np.array(
        [[a[0][0], a[1][0], a[-2][0]],
         [a[2][0], a[3][0], a[-1][0]],
         [0, 0, 1]]
    )
    # fmt: on
    return affine


def read_itk_affine_mat_3d(pth: Union[str, Path]):
    """Read 3D ANTs affine in RAS coords

    ANTs uses LPS+; the returned affine is in RAS+.

    """
    data = loadmat(pth)
    p = data["AffineTransform_double_3_3"][:, 0]
    c = data["fixed"][:, 0]
    A = _affine_from_itk(p, c)
    A = _affine_convert_ras_lps(A)
    return A


def read_fcsv(infile):
    """Return DataFrame from Slicer .fcsv"""
    # Markups fiducial file version = 5.2 has two extra unlabled columns, so we can't
    # use the header row directly.

    # Get column names
    with open(infile, "r") as f:
        for i in range(3):
            ln = f.readline()
    colnames = ln.removeprefix("# columns = ").split(",")
    # Read data
    df = pd.read_csv(infile, skiprows=3, index_col=None, header=None)
    df = df[[i for i in range(len(colnames))]]
    df.columns = colnames
    # Switch from LPS to RAS
    df["x"] = -df["x"]
    df["y"] = -df["y"]
    return df


def read_landmark_json(pth):
    """Return landmark coordinates (RAS+) from Slicer JSON markups (.mrk.json) file"""
    with open(pth, "rb") as f:
        data = json.loads(f.read())
    if len(data["markups"]) > 1:
        raise NotImplementedError("Multiple markups sections not yet supported.")
    out = {"x": [], "y": [], "z": [], "label": []}
    for pt_info in data["markups"][0]["controlPoints"]:
        orient = np.array(pt_info["orientation"]).reshape(3, 3)
        x = np.array(pt_info["position"]) @ orient
        out["x"].append(x[0])
        out["y"].append(x[1])
        out["z"].append(x[2])
        out["label"].append(pt_info["label"])
    return pd.DataFrame(out)


def read_itk_transform_txt(pth):
    """Read an Insight Transform File V1.0 text file

    Returns an augmented affine matrix representing the transform in RAS+ coordinates.
    Note that ITK uses LPS+ so this requires conversion.

    """
    supported_transforms = (
        "MatrixOffsetTransformBase_double_3_3",
        "AffineTransform_double_3_3",
    )
    with open(pth, "r") as f:
        fmt = f.readline().lstrip("#").rstrip()
        if not fmt == "Insight Transform File V1.0":
            raise ValueError(
                f"File {pth} does not declare 'Insight Transform File V1.0' on its first line; is it really an ITK Transform File?"
            )
        order = int(f.readline().removeprefix("#Transform ").rstrip())
        transform_type = f.readline().removeprefix("Transform: ").rstrip()
        if not transform_type in supported_transforms:
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
    A = _affine_from_itk(p, fixed)
    A = _affine_convert_ras_lps(A)
    return A


def write_fcsv(markers: DataFrame, pth):
    """Write fiducial markers to Slicer .fcsv file

    :param markers: DataFrame with required columns "x", "y", and "z" and optional
    columns "label".  Slicer uses LPS and the coordinates will be converted accordingly.

    Returns RAS+ coordinates.

    """
    columns = [
        "id",
        "x",
        "y",
        "z",
        "ow",
        "ox",
        "oy",
        "oz",
        "vis",
        "sel",
        "lock",
        "label",
        "desc",
        "associatedNodeID",
    ]
    markers = markers.reset_index().copy()
    markers["id"] = markers.index + 1
    markers["x"] = -markers["x"]
    markers["y"] = -markers["y"]
    markers["ow"] = 0
    markers["ox"] = 0
    markers["oy"] = 0
    markers["oz"] = 1
    markers["vis"] = 1
    markers["sel"] = 1
    markers["lock"] = 0
    if "label" not in markers:
        markers["label"] = [str(i + 1) for i in markers.index]
    markers["desc"] = ""
    markers["associatedNodeID"] = None
    markers["extra1"] = 2
    markers["extra2"] = 0
    with open(pth, "w") as f:
        f.writelines(
            [
                "# Markups fiducial file version = 5.4\n",
                "# CoordinateSystem = LPS\n",
                f"# columns = {','.join(columns)}\n",
            ]
        )
        markers[columns + ["extra1", "extra2"]].to_csv(f, header=False, index=False)


def write_itk_affine_mat(affine: NDArray, pth: Union[str, Path]):
    """Write an RAS+ affine transform to an ITK-compatible .mat file

    Since ITK uses LPS+, the input affine is converted to LPS+ coordinates before
    being written to the output file.

    """
    dim = affine.shape[0] - 1
    A = _affine_convert_ras_lps(affine)
    serialized = np.hstack([A[:-1, :-1].ravel(), A[:-1, -1]])
    mat = {
        f"AffineTransform_double_{dim}_{dim}": np.atleast_2d(serialized).T,
        "fixed": np.zeros((A.shape[0] - 1, 1)),
    }
    savemat(pth, mat, format="4")  # ANTs requires format 4


def write_itk_affine_txt(affine, pth):
    """Write an RAS+ affine transform to an Insight Transform File V1.0 text file

    :param affine: 4x4 augmented affine matrix in RAS+ coordinates

    :param pth: Output file path.

    Note that ITK uses LPS+, so the output text file's parameters are in LPS+
    coordinates.

    """
    A = affine
    # writelines doesn't write lines; it writes the concatenation of the list
    lines = [
        "#Insight Transform File V1.0\n",
        "#Transform 0\n",
        "Transform: MatrixOffsetTransformBase_double_3_3\n",
        f"Parameters: {A[0, 0]} {A[0, 1]} {-A[0, 2]}",
        f" {A[1, 0]} {A[1, 1]} {-A[1, 2]}",
        f" {-A[2, 0]} {-A[2, 1]} {A[2, 2]}",
        f" {-A[0, 3]} {-A[1, 3]} {A[2, 3]}\n",
        "FixedParameters: 0 0 0\n",
    ]
    with open(pth, "w") as f:
        f.writelines(lines)
