from copy import copy
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from pandas import DataFrame
from scipy.io import loadmat, savemat


def _affine_with_cor(A, cor):
    """Update affine matrix with center of rotation"""
    A = copy(A)
    R = A[:3, :3]
    cor_Δ = cor - R @ cor
    A[:3, -1] = A[:3, -1] + cor_Δ
    return A


def read_affine_mat(pth: Union[str, Path]):
    """Read ANTs affine in RAS coords

    ANTs uses LPS+; the returned affine is in RAS+.

    """
    ants_affine = loadmat(pth)
    if "AffineTransform_double_3_3" in ants_affine:
        return read_affine_mat_3d(pth)
    elif "AffineTransform_double_2_2" in ants_affine:
        return read_affine_mat_2d(pth)
    else:
        raise NotImplementedError(f"Affine transform format not supported.")


def read_affine_mat_2d(pth: Union[str, Path]):
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


def read_affine_mat_3d(pth: Union[str, Path]):
    """Read 3D ANTs affine in RAS coords

    ANTs uses LPS+; the returned affine is in RAS+.

    """
    ants_affine = loadmat(pth)
    a = ants_affine["AffineTransform_double_3_3"]
    # fmt: off
    affine = np.array([
         [a[0][0], a[1][0], -a[2][0], -a[-3][0]],
         [a[3][0], a[4][0], -a[5][0], -a[-2][0]],
         [-a[6][0], -a[7][0], a[8][0], a[-1][0]],
         [0, 0, 0, 1],
    ])
    # fmt: on
    return affine


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
    #         [p[0], p[1], p[2], p[9]],
    #         [p[3], p[4], p[5], p[10]],
    #         [p[6], p[7], p[8], p[11]],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # A = ras_A_lps @ itk_affine @ ras_A_lps.T
    # Have to apply rotations for center of rotation offset in LPS+ coords
    R_lps = np.array(
        [
            [p[0], p[1], p[2]],
            [p[3], p[4], p[5]],
            [-p[6], -p[7], p[8]],
        ]
    )
    cor_delta = fixed - R_lps @ fixed
    cor_delta[0:2] = -cor_delta[0:2]
    # Construct the RAS+ affine
    A = np.array(
        [
            [p[0], p[1], -p[2], -p[9]],
            [p[3], p[4], -p[5], -p[10]],
            [-p[6], -p[7], p[8], p[11]],
            [0, 0, 0, 1],
        ]
    )
    A[:3, -1] = A[:3, -1] + cor_delta
    return A


def write_fcsv(markers: DataFrame, pth):
    """Write fiducial markers to Slicer .fcsv file

    :param markers: DataFrame with required columns "x", "y", and "z" and optional
    columns "label".  RAS (mm) coordinates assumed.  Slicer uses LPS and the
    coordinates will be converted accordingly..

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


def write_affine_mat(affine: NDArray, pth: Union[str, Path]):
    dim = affine.shape[0] - 1
    serialized = np.hstack([affine[:-1, :-1].ravel(), affine[:-1, -1]])
    mat = {
        f"AffineTransform_double_{dim}_{dim}": np.atleast_2d(serialized).T,
        "fixed": np.zeros((affine.shape[0] - 1, 1)),
    }
    savemat(pth, mat, format="4")  # ANTs requires format 4


def write_itk_transform_txt(affine, pth):
    """Write an augmented affine matrix to an Insight Transform File V1.0 text file

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
