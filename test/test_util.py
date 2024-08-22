import numpy as np
import numpy.testing as npt

from pastrycutter.util import itk_directions_from_affine


def test_itk_directions_from_affine():
    # RAS+ affine from Nibabel
    # fmt: off
    nii_affine = np.array(
        [[0.009250608355532, 0.033524328990694, 2.993227724788234, -36.931278228759766],
         [-0.517010726852538, 0.022959245534753, 0.045046186633166, 34.55845260620117],
         [-0.022398590212721, -0.516106401083607, 0.196432706969583, 198.7960662841797],
         [0.0, 0.0, 0.0, 1.0]])
    # fmt: on
    # ITK directions, spacing, and origin (from ANTSPy reading the same image)
    itk_direction = np.array(
        [
            [-0.017872864365285, -0.064755858547126, -0.997741068283379],
            [0.998903283843942, -0.044348257345331, -0.015015373327727],
            [-0.04327574661596, -0.996915197423661, 0.065477471122289],
        ]
    )
    itk_spacing = np.array([0.5175783634185791, 0.5177034139633179, 3.000004529953003])
    itk_origin = np.array([36.931278228759766, -34.55845260620117, 198.7960662841797])
    # Calculated ITK directions, spacing, and origin
    direction, spacing, origin = itk_directions_from_affine(nii_affine)
    npt.assert_allclose(direction, itk_direction, strict=True)
    npt.assert_allclose(spacing, itk_spacing, strict=True)
    npt.assert_allclose(origin, itk_origin, strict=True)
