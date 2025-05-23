"""
??? info "References"

    1. Cornette, J. J., & Shanks, R. E. (1992). Bidirectional reflectance
    of flat, optically thick particulate systems. Applied Optics, 31(15),
    3152-3160. <https://doi.org/10.1364/AO.31.003152>
"""

from functools import partial

# import numpy as np
# from numpy.typing import ArrayLike
import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

# import jax.numpy as np


@jax.jit
# @partial(np.vectorize, signature="(n),(n)->(n)")
def h_function_1(x: ArrayLike, w: ArrayLike) -> ArrayLike:
    gamma = np.sqrt(1 - w)
    return (1 + 2 * x) / (1 + 2 * x * gamma)


@jax.jit
# @partial(np.vectorize, signature="(n),(n)->(n)")
def h_function_2(x: ArrayLike, w: ArrayLike) -> ArrayLike:
    gamma = np.sqrt(1 - w)
    r0 = (1 - gamma) / (1 + gamma)
    h_inv = 1 - w * x * (r0 + (1 - 2 * r0 * x) / 2 * np.log(1 + 1 / x))
    return 1 / h_inv


@jax.jit
# @partial(np.vectorize, signature="(n),(n)->(n)")
def h_function_2_derivative(x: ArrayLike, w: ArrayLike) -> ArrayLike:
    gamma = np.sqrt(1 - w)
    r0 = (1 - gamma) / (1 + gamma)
    x_log_term = np.log(1 + 1 / x)

    dr0_dw = 1 / (gamma * (1 + gamma) ** 2)
    h = h_function_2(x, w)
    return (
        h**2
        * x
        * (r0 + (1 - 2 * r0 * x) / 2 * x_log_term + w * dr0_dw * (1 - x * x_log_term))
    )


def h_function(x: ArrayLike, w: ArrayLike, level: int = 1) -> ArrayLike:
    """
    Calculates the Hapke function for a given set of parameters.

    Args:
        x (float): The input parameter.
        w (numpy.ndarray): The weight array.
        level (int, optional): The level of the Hapke function to calculate. Defaults to 1.

    Returns:
        (float): The calculated Hapke function value.

    Raises:
        Exception: If an invalid level is provided.
    """

    match level:
        case 1:
            h = h_function_1(x, w)
        case 2:
            h = h_function_2(x, w)
        case _:
            raise Exception("Please provide a level between 1 and 2!")

    return h


def h_function_derivative(x: ArrayLike, w: ArrayLike, level: int = 1) -> ArrayLike:
    """
    Calculates the Hapke function for a given set of parameters.

    Args:
        x (float): The input parameter.
        w (numpy.ndarray): The weight array.
        level (int, optional): The level of the Hapke function to calculate. Defaults to 1.

    Returns:
        (float): The calculated Hapke function value.

    Raises:
        Exception: If an invalid level is provided.
    """

    match level:
        case 1:
            dh_dw = np.zeros_like(x)
        case 2:
            dh_dw = h_function_2_derivative(x, w)
        case _:
            raise Exception("Please provide a level between 1 and 2!")

    return dh_dw


@jax.jit
def double_henyey_greenstein(cos_g: ArrayLike, b: float = 0.21, c: float = 0.7):
    """
    Calculates the phase function for the double Henyey-Greenstein model.

    Args:
        cos_g (float): The cosine of the scattering angle.
        b (float, optional): The asymmetry parameter. Defaults to 0.21.
        c (float, optional): The backscatter fraction. Defaults to 0.7.

    Returns:
        (float): The phase function value.

    """
    p_g = (1 + c) / 2 * (1 - b**2) / (1 - 2 * b * cos_g + b**2) ** 1.5
    p_g += (1 - c) / 2 * (1 - b**2) / (1 + 2 * b * cos_g + b**2) ** 1.5
    return p_g


@jax.jit
def cornette_shanks(cos_g: ArrayLike, xi: float):
    """
    Calculates the Cornette-Shanks function.

    Args:
        cos_g (float): The cosine of the incidence angle.
        xi (float): The single scattering albedo.

    Returns:
        (float): The value of the Cornette-Shanks function.

    Note:
        Equation 8 from Cornette and Shanks (1992).

    """
    p_g = (
        1.5
        * (1 - xi**2)
        / (2 + xi**2)
        * (1 + cos_g**2)
        / np.power(1 + xi**2 - 2 * xi * cos_g, 1.5)
    )
    return p_g
