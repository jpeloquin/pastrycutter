import numpy as np


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