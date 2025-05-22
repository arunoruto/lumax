from typing import Callable

import numpy as np
import numpy.typing as npt

from refmod.hapke.functions import h_function
from refmod.hapke.roughness import microscopic_roughness


def imsa(
    incidence_direction: npt.NDArray,
    emission_direction: npt.NDArray,
    surface_orientation: npt.NDArray,
    single_scattering_albedo: npt.NDArray,
    phase_function: Callable[[npt.NDArray], npt.NDArray],
    opposition_effect_h: float = 0,
    oppoistion_effect_b0: float = 0,
    roughness: float = 0,
) -> npt.NDArray:
    """
    Calculates the reflectance using the IMSA (Inversion of Multiple Scattering and Absorption) model.

    Args:
        incidence_direction: Array of shape (..., 3) representing the incidence direction vectors.
        emission_direction: Array of shape (..., 3) representing the emission direction vectors.
        surface_orientation: Array of shape (..., 3) representing the surface orientation vectors.
        single_scattering_albedo: Array of shape (...,) representing the single scattering albedo values.
        phase_function: Callable function that takes the cosine of the phase angle and returns the phase function values.
        opposition_effect_h: Opposition effect parameter h.
        oppoistion_effect_b0: Opposition effect parameter b0.
        roughness: Surface roughness parameter.

    Returns:
        Array of shape (...,) representing the reflectance values.

    Raises:
        Exception: If at least one reflectance value is not real.

    Notes:
        - The input arrays should have compatible shapes for broadcasting.
        - The phase function should be a callable function that takes the cosine of the phase angle as input and returns
          the phase function values.
        - The reflectance values are calculated using the IMSA model, which accounts for multiple scattering and absorption.

    References:
        - TODO: Add references to the IMSA model.

    """
    # Allocate memory
    if np.ndim(single_scattering_albedo) == 0:
        single_scattering_albedo *= np.ones(surface_orientation.shape[:2])

    # Angles
    incidence_direction /= np.linalg.norm(
        incidence_direction, axis=2, keepdims=True
    )
    emission_direction /= np.linalg.norm(
        emission_direction, axis=2, keepdims=True
    )
    surface_orientation /= np.linalg.norm(
        surface_orientation, axis=2, keepdims=True
    )

    # Phase angle
    cos_alpha = np.sum(incidence_direction * emission_direction, axis=2)
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1
    sin_alpha = np.sqrt(1 - cos_alpha**2)

    [s, mu_0, mu] = microscopic_roughness(
        roughness, incidence_direction, emission_direction, surface_orientation
    )

    # Phase function values
    p_g = phase_function(cos_alpha)

    # H-Function
    # TODO: implement derivation of the H-function
    h_mu_0 = h_function(mu_0, single_scattering_albedo)
    h_mu = h_function(mu, single_scattering_albedo)

    # Opposition effect
    b_g = 1.0
    if oppoistion_effect_b0 != 0.0:
        b_g += oppoistion_effect_b0 / (
            1 + sin_alpha / (1 + cos_alpha) / opposition_effect_h
        )

    # Reflectance
    refl = mu_0 / (mu_0 + mu) * s / (4 * np.pi)
    refl *= single_scattering_albedo * (b_g * p_g + h_mu_0 * h_mu - 1)
    refl /= 4 * np.pi
    refl[(mu <= 0) | (mu_0 <= 0)] = np.nan
    refl[refl < 1e-6] = np.nan

    # Final result
    threshold_imag = 0.1
    threshold_error = 1e-4
    arg_rh = np.divide(
        np.imag(refl),
        np.real(refl),
        out=np.zeros_like(refl, dtype=float),
        where=np.real(refl) != 0,
    )
    refl[arg_rh > threshold_imag] /= np.nan

    if np.any(arg_rh >= threshold_error):
        raise Exception("At least one reflectance value is not real!")

    return refl
