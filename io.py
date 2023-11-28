import json

import numpy as np
import pandas as pd


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
    out = {"x": [],
            "y": [],
            "z": [],
            "label": []}
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
    # nifti_affine = ras_A_lps @ itk_affine @ ras_A_lps.T
    nifti_affine = np.array(
        [
            [p[0], p[1], -p[2], -p[9]],
            [p[3], p[4], -p[5], -p[10]],
            [-p[6], -p[7], p[8], p[11]],
            [0, 0, 0, 1],
        ]
    )
    return nifti_affine


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
