"""
??? info "References"

    1. Hapke, B. (1984). Bidirectional reflectance spectroscopy: 3.
    Correction for macroscopic roughness. Icarus, 59(1), 41-59.
    <https://doi.org/10.1016/0019-1035(84)90054-X>
"""

# import numpy as np
# import numpy.typing as npt
import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from refmod.hapke.functions import normalize_vec


@jax.jit
def __f_exp(x: ArrayLike, y: float):
    """
    Helper for the micoscopic roughness:
    calculates the exponential function for the given inputs.

    Args:
        x (numpy.ndarray): The input array.
        y (float): The exponential factor.

    Returns:
        (np.ndarray): The result of the exponential function.

    """
    return np.exp(-2 / np.pi * y * x)


@jax.jit
def __f_exp_2(x: ArrayLike, y: float):
    """
    Helper for the micoscopic roughness:
    calculates the exponential function with a squared term.

    Args:
        x (numpy.ndarray): The input array.
        y (float): The value to be squared.

    Returns:
        (np.ndarray): The result of the exponential function.

    """
    return np.exp(-(y**2) * x**2 / np.pi)


@jax.jit
def __prime_term(
    cos_x: ArrayLike,
    sin_x: ArrayLike,
    cot_r: float,
    cos_psi: ArrayLike,
    sin_psi_div_2_sq: ArrayLike,
    psi: ArrayLike,
    cot_a: ArrayLike,
    cot_b: ArrayLike,
    cot_c: ArrayLike,
    cot_d: ArrayLike,
    index: ArrayLike,
):
    # num = cos_psi[index] * __f_exp_2(cot_a[index], cot_r) + sin_psi_div_2_sq[
    #     index
    # ] * __f_exp_2(cot_b[index], cot_r)
    # den = (
    #     2
    #     - __f_exp(cot_c[index], cot_r)
    #     - psi[index] / np.pi * __f_exp(cot_d[index], cot_r)
    # )
    return np.where(
        index,
        cos_x
        + sin_x
        / cot_r
        * (
            cos_psi * __f_exp_2(cot_a, cot_r)
            + sin_psi_div_2_sq * __f_exp_2(cot_b, cot_r)
        )
        / (2 - __f_exp(cot_c, cot_r) - psi / np.pi * __f_exp(cot_d, cot_r)),
        0,
    )


@jax.jit
def microscopic_roughness(
    roughness: float,
    incidence_direction: Array,
    emission_direction: Array,
    surface_orientation: Array,
):
    r"""
    Calculates the microscopic roughness factor for the Hapke reflectance model.

    Args:
        roughness (float): The roughness parameter.
        incidence_direction (numpy.ndarray): Array of incidence directions.
        emission_direction (numpy.ndarray): Array of emission directions.
        surface_orientation (numpy.ndarray): Array of surface orientations.

    Returns:
        s (numpy.ndarray): The microscopic roughness factor.
        cos_i (numpy.ndarray): The modified incidence-normal cosine value ($\cos_i^{\prime}$).
        cos_e (numpy.ndarray): The modified emission-normal cosine value ($\cos_e^{\prime}$).

    Note:
        - prime-zero terms: equations 48 and 49 in Hapke (1984).
        - $i  < e$: equations 46 and 47 in Hapke (1984).
        - $i >= e$: equations 50 and 51 in Hapke (1984).
    """

    # Angles
    # incidence_direction /= np.linalg.norm(
    #     incidence_direction, axis=-1, keepdims=True
    # )
    # emission_direction /= np.linalg.norm(
    #     emission_direction, axis=-1, keepdims=True
    # )
    # surface_orientation /= np.linalg.norm(
    #     surface_orientation, axis=-1, keepdims=True
    # )
    incidence_direction = normalize_vec(incidence_direction)
    emission_direction = normalize_vec(emission_direction)
    surface_orientation = normalize_vec(surface_orientation)

    # Incidence angle
    cos_i = np.sum(incidence_direction * surface_orientation, axis=-1)
    cos_i = np.array([cos_i]) if isinstance(cos_i, float) else cos_i
    cos_i = np.clip(cos_i, -1, 1)
    sin_i = np.sqrt(1 - cos_i**2)
    # tan_i = sin_i / cos_i
    # cot_i = np.divide(1, tan_i, out=np.ones_like(tan_i) * np.inf, where=tan_i != 0)
    # cot_i = np.divide(cos_i, sin_i, out=np.ones_like(cos_i) * np.inf, where=sin_i != 0)
    cot_i = np.where(sin_i == 0, np.inf, cos_i / sin_i)
    i = np.arccos(cos_i)

    # Emission angle
    cos_e = np.sum(emission_direction * surface_orientation, axis=-1)
    cos_e = np.array([cos_e]) if isinstance(cos_e, float) else cos_e
    cos_e = np.clip(cos_e, -1, 1)
    sin_e = np.sqrt(1 - cos_e**2)
    cot_e = np.where(sin_e == 0, np.inf, cos_e / sin_e)
    e = np.arccos(cos_e)

    # if roughness == 0:
    #     print("Roughness is zero, returning default values")
    #     return np.ones_like(cos_e), cos_i, cos_e

    # Projections
    projection_incidence = normalize_vec(
        incidence_direction
        - np.expand_dims(cos_i, axis=-1) * surface_orientation
    )
    projection_emission = normalize_vec(
        emission_direction
        - np.expand_dims(cos_e, axis=-1) * surface_orientation
    )

    # Azicos_eth angle
    cos_psi = np.sum(projection_incidence * projection_emission, axis=-1)
    cos_psi = np.clip(cos_psi, -1, 1)
    sin_psi = np.sqrt(1 - cos_psi**2)
    sin_psi_div_2_sq = np.abs(0.5 - cos_psi / 2)
    psi = np.arccos(cos_psi)

    # Macroscopic Roughness
    tan_rough = np.tan(roughness)
    cot_rough = 1 / tan_rough

    # Check for cases
    ile = i < e
    ige = i >= e
    # Check for singularities
    mask = (cos_i == 1) | (cos_e == 1)

    factor = 1 / np.sqrt(1 + np.pi * tan_rough**2)
    # f_psi = np.exp(
    #     -2
    #     * np.divide(
    #         sin_psi,
    #         1 + cos_psi,
    #         out=np.ones_like(sin_psi) * -1,
    #         where=cos_psi != -1,
    #     )
    # )
    f_psi = np.exp(-2 * sin_psi / (1 + cos_psi))

    cos_i_s0 = factor * (
        cos_i
        + sin_i
        * tan_rough
        * __f_exp_2(cot_i, cot_rough)
        / (2.0 - __f_exp(cot_i, cot_rough))
    )
    cos_e_s0 = factor * (
        cos_e
        + sin_e
        * tan_rough
        * __f_exp_2(cot_e, cot_rough)
        / (2.0 - __f_exp(cot_e, cot_rough))
    )

    cos_i_s = np.zeros_like(cos_i)
    # cos_i_s = cos_i_s.at[ile].set(
    cos_i_s += factor * __prime_term(
        cos_i,
        sin_i,
        cot_rough,
        cos_psi,
        sin_psi_div_2_sq,
        psi,
        cot_e,
        cot_i,
        cot_e,
        cot_i,
        ile,
    )
    cos_i_s += factor * __prime_term(
        cos_i,
        sin_i,
        cot_rough,
        np.ones_like(cos_psi),
        -sin_psi_div_2_sq,
        psi,
        cot_i,
        cot_e,
        cot_i,
        cot_e,
        ige,
    )
    # cos_i_s = cos_i_s.at[mask].set(cos_i[mask])
    cos_i_s = np.where(mask, cos_i, cos_i_s)

    cos_e_s = np.zeros_like(cos_e)
    cos_e_s += factor * __prime_term(
        cos_e,
        sin_e,
        cot_rough,
        np.ones_like(cos_psi),
        -sin_psi_div_2_sq,
        psi,
        cot_e,
        cot_i,
        cot_e,
        cot_i,
        ile,
    )
    cos_e_s += factor * __prime_term(
        cos_e,
        sin_e,
        cot_rough,
        cos_psi,
        sin_psi_div_2_sq,
        psi,
        cot_i,
        cot_e,
        cot_i,
        cot_e,
        ige,
    )
    # cos_e_s = cos_e_s.at[mask].set(cos_e[mask])
    cos_e_s = np.where(mask, cos_e, cos_e_s)

    s = factor * (cos_e_s / cos_e_s0) * (cos_i / cos_i_s0)
    # s = s.at[ile].set(
    #     s[ile] / (1 + f_psi[ile] * (factor * (cos_i[ile] / cos_i_s0[ile]) - 1))
    # )
    # s = s.at[ige].set(
    #     s[ige] / (1 + f_psi[ige] * (factor * (cos_e[ige] / cos_e_s0[ige]) - 1))
    # )
    # s = s.at[mask].set(1)
    s /= np.where(
        ile,
        (1 + f_psi * (factor * (cos_i / cos_i_s0) - 1)),
        (1 + f_psi * (factor * (cos_e / cos_e_s0) - 1)),
    )
    s = np.where(mask, 1, s)

    return np.squeeze(s), np.squeeze(cos_i_s), np.squeeze(cos_e_s)
