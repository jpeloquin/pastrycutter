import ants
import numpy as np
import numpy.testing as npt

from pastrycutter.io import _affine_with_cor


def _apply_affine(A, pt):
    return (A @ np.concatenate([pt, [1]]))[:3]


def _affine_from_params(p):
    return np.array(
        [
            [p[0], p[1], p[2], p[9]],
            [p[3], p[4], p[5], p[10]],
            [p[6], p[7], p[8], p[11]],
            [0, 0, 0, 1],
        ]
    )


def test_xrot():
    # 45° rotation about x
    p = [
        1,
        0,
        0,
        0,
        0.7071067811865482,
        -0.7071067811865481,
        0,
        0.7071067811865479,
        0.707106781186548,
        3,
        7,
        9,
    ]
    A = _affine_from_params(p)
    for c in ([10, 0, 0], [0, 10, 0], [0, 0, 10]):
        B = _affine_with_cor(A, c)
        t = ants.create_ants_transform(parameters=p, fixed_parameters=c)
        for pt in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            actual = _apply_affine(B, pt)
            desired = t.apply_to_point(pt)
            npt.assert_allclose(actual, desired, atol=1e-6)


def test_yrot():
    # 30° rotation about y
    p = [
        0.8660254037844399,
        0.5,
        0,
        0,
        1,
        0,
        -0.5,
        0,
        0.866025403784439,
        3,
        7,
        9,
    ]
    A = _affine_from_params(p)
    for c in ([10, 0, 0], [0, 10, 0], [0, 0, 10]):
        B = _affine_with_cor(A, c)
        t = ants.create_ants_transform(parameters=p, fixed_parameters=c)
        for pt in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            actual = _apply_affine(B, pt)
            desired = t.apply_to_point(pt)
            npt.assert_allclose(actual, desired, atol=1e-6)


def test_zrot():
    # -30° rotation about z
    p = [
        0.8660254037844399,
        0.5,
        0,
        -0.5,
        0.8660254037844394,
        0,
        0,
        0,
        1,
        3,
        7,
        9,
    ]
    A = _affine_from_params(p)
    for c in ([10, 0, 0], [0, 10, 0], [0, 0, 10]):
        B = _affine_with_cor(A, c)
        t = ants.create_ants_transform(parameters=p, fixed_parameters=c)
        for pt in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            actual = _apply_affine(B, pt)
            desired = t.apply_to_point(pt)
            npt.assert_allclose(actual, desired, atol=1e-6)


def test_combined():
    p = [
        0.6629458982677394,
        -0.6208851530148471,
        -0.4183352277010815,
        0.6718340441253725,
        0.739942111693849,
        -0.033536375716476774,
        0.3303660895493516,
        -0.25881904510252,
        0.9076733711903693,
        44.179872630636496,
        17.391967218157433,
        13.352369341586469,
    ]
    A = _affine_from_params(p)
    for c in ([10, 0, 0], [0, 10, 0], [0, 0, 10]):
        B = _affine_with_cor(A, c)
        t = ants.create_ants_transform(parameters=p, fixed_parameters=c)
        for pt in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]):
            actual = _apply_affine(B, pt)
            desired = t.apply_to_point(pt)
            npt.assert_allclose(actual, desired, atol=1e-6)
