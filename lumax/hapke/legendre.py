"""
??? info "References"

    1. Hapke, B. (2002). Bidirectional Reflectance Spectroscopy: 5.
    The Coherent Backscatter Opposition Effect and Anisotropic Scattering.
    Icarus, 157(2), 523–534. <https://doi.org/10.1006/icar.2002.6853>
"""

from functools import partial

# import numpy as np
# import numpy.typing as npt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@partial(jax.jit, static_argnums=(0,))
def coef_a(n: int = 15):
    """
    Calculates the coefficients 'a_n' for the Legendre polynomial series.

    Args:
        n (int): The number of coefficients to calculate. Default is 15.

    Returns:
        (numpy.ndarray): An array of coefficients 'a_n' for the Legendre polynomial series.

    Note:
        Equation 27 in Hapke (2002).
    """
    a_n = jnp.zeros(n + 1)
    # range = jnp.arange(n)
    # a_n = a_n.at[1:].set(-1 * eval_legendre(range, 0) / (range + 2))
    # # a_n[1:] = -1 * eval_legendre(range, 0) / (range + 2)
    # return a_n
    a_n = a_n.at[1].set(-0.5)
    for i in range(3, n + 1, 2):
        a_n = a_n.at[i].set((2 - i) / (i + 1) * a_n[i - 2])
    return a_n


# @partial(jax.jit, static_argnames=["n"])
@partial(jax.jit, static_argnums=(2,))
def coef_b_single(b: float, c: float, n: int):
    # n = 15
    # range = np.arange(1, n + 2)
    range = jax.lax.iota(int, n + 1) + 1
    # range = jnp.arange(n + 1) + 1
    return (2 * range + 1) * jnp.power(-b, range)


# @partial(jax.jit, static_argnames=["n"])
@partial(jax.jit, static_argnums=(2,))
def coef_b_double(b: float, c: float, n: int):
    # range = np.arange(n + 1)
    # print(n[0])
    # n = 15
    range = jax.lax.iota(int, n + 1)
    b_n = c * (2 * range + 1) * jnp.power(b, range)
    # TODO: why is the first element one and not c?
    b_n = b_n.at[0].set(1)
    return b_n


# @partial(jax.jit, static_argnums=(2,))
def coef_b(b: float, c: float, n: int = 15):
    """
    Calculates the coefficients for the Hapke reflectance model Legendre polynomial expansion.

    Args:
        b (float, optional): The single scattering albedo. Defaults to 0.21.
        c (float, optional): The asymmetry factor. Defaults to 0.7.
        n (int, optional): The number of coefficients to calculate. Defaults to 15.

    Returns:
        (numpy.ndarray): The calculated coefficients for the Legendre polynomial expansion.

    Note:
        Equation on page 530 in Hapke (2002).
    """
    return coef_b_single(b, n) if jnp.isnan(c) else coef_b_double(b, c, n)
    # args = (b, c, n)
    # return jax.lax.cond(
    # jnp.isnan(c),
    # coef_b_single,
    # coef_b_double,
    # *args,
    # )


@jax.jit
def function_p(x: ArrayLike, b_n: ArrayLike, a_n: ArrayLike):
    """
    Calculates the P function using the Hapke reflectance model.

    Args:
        x (numpy.ndarray): The input array.
        b_n (numpy.ndarray): The B_n coefficients.
        a_n (ArrayLike, optional): The A_n coefficients. Defaults to jnp.empty(1) * jnp.nan.

    Returns:
        (numpy.ndarray): The calculated P function.

    Note:
        Equations 23 and 24 in Hapke (2002).
    """
    n = b_n.size
    res = jnp.ones(x.shape + (n,))
    res = res.at[..., 1].set(x)
    for i in range(2, n):
        res = res.at[..., i].set(
            (2 - 1 / i) * x * res[..., i - 1] - (1 - 1 / i) * res[..., i - 2]
        )
    return 1 + jnp.sum(a_n * b_n * res, axis=-1)

    # n = jnp.arange(b_n.size)
    # if jnp.any(jnp.isnan(a_n)):
    # a_n = coef_a(b_n.size)
    # match x.ndim:
    #     case 1:
    #         x = jnp.expand_dims(x, axis=1)
    #     case 2:
    #         x = jnp.expand_dims(x, axis=2)
    # x = jnp.expand_dims(x, axis=-1)
    # # NOTE: Maybe use: from jax.scipy.special import lpmn_values
    # return 1 + jnp.sum(a_n * b_n * eval_legendre(n, x), axis=-1)
    # return 1 + jnp.sum(a_n * b_n * eval_legendre(n, x), axis=2)


@jax.jit
def value_p(b_n: ArrayLike, a_n: ArrayLike):
    """
    Calculates the value of the P function.

    Args:
        b_n (numpy.ndarray): Array of coefficients.
        a_n (ArrayLike, optional): Array of coefficients. Defaults to jnp.empty(1) * jnp.nan.

    Returns:
        (float): The calculated value of the P function.

    Note:
        Equations 25 in Hapke (2002).
    """
    return 1 + jnp.sum(a_n**2 * b_n)
