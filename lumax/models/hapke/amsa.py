from functools import partial
from typing import Callable, Tuple, Union

# import numpy as np
# import numpy.typing as npt
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from lumax.core.geometry import normalize_vec
from lumax.models.hapke.functions import (
    double_henyey_greenstein,
    h_function_2,
    h_function_2_derivative,
)
from lumax.models.hapke.legendre import coef_a, function_p, value_p
from lumax.models.hapke.roughness import microscopic_roughness


# @partial(jax.jit, static_argnames=["phase_function"])
@partial(jax.profiler.annotate_function, name="amsa_main")
@jax.jit
def __amsa_main(
    single_scattering_albedo: float,
    incidence_direction: Array,
    emission_direction: Array,
    surface_orientation: Array,
    # phase_function: Callable[[ArrayLike], ArrayLike],
    phase_function: dict,
    b_n: Array,
    a_n: Array,
    roughness: float = 0,
    hs: float = 0,
    bs0: float = 0,
    hc: float = 0,
    bc0: float = 0,
):
    """
    Calculates the reflectance using the AMSA (Advanced Modified Shadowing and Coherent Backscattering) model.

    Args:
        incidence_direction: Array of shape (number_u, number_v, 3) representing the incidence direction vectors.
        emission_direction: Array of shape (number_u, number_v, 3) representing the emission direction vectors.
        surface_orientation: Array of shape (number_u, number_v, 3) representing the surface orientation vectors.
        single_scattering_albedo: Array of shape (number_u, number_v) representing the single scattering albedo values.
        phase_function: Callable function that takes the cosine of the scattering angle and returns the phase function values.
        b_n: Array of shape (n,) representing the coefficients of the Legendre expansion.
        a_n: Array of shape (n,) representing the coefficients of the Legendre expansion. Defaults to jnp.empty(1) * jnp.nan.
        hs: Float representing the shadowing parameter. Defaults to 0.
        bs0: Float representing the shadowing parameter. Defaults to 0.
        roughness: Float representing the surface roughness. Defaults to 0.
        hc: Float representing the coherent backscattering parameter. Defaults to 0.
        bc0: Float representing the coherent backscattering parameter. Defaults to 0.

    Returns:
        Array of shape (number_u, number_v) representing the reflectance values.

    Raises:
        Exception: If at least one reflectance value is not real.

    """
    # Angles
    # incidence_direction /= jnp.linalg.norm(
    #     incidence_direction, axis=-1, keepdims=True
    # )
    # emission_direction /= jnp.linalg.norm(
    #     emission_direction, axis=-1, keepdims=True
    # )
    # surface_orientation /= jnp.linalg.norm(
    #     surface_orientation, axis=-1, keepdims=True
    # )
    incidence_direction = normalize_vec(incidence_direction)
    emission_direction = normalize_vec(emission_direction)
    surface_orientation = normalize_vec(surface_orientation)

    # Roughness
    [s, mu_0, mu] = microscopic_roughness(
        roughness, incidence_direction, emission_direction, surface_orientation
    )

    # Legendre
    p_mu_0 = function_p(mu_0, b_n, a_n)
    p_mu = function_p(mu, b_n, a_n)
    p = value_p(b_n, a_n)

    # Alpha angle
    cos_alpha = jnp.sum(incidence_direction * emission_direction, axis=-1)
    cos_alpha = jnp.array([cos_alpha]) if isinstance(cos_alpha, float) else cos_alpha
    cos_alpha = jnp.clip(cos_alpha, -1, 1)
    sin_alpha = jnp.sqrt(1 - cos_alpha**2)
    tan_alpha_2 = sin_alpha / (1 + cos_alpha)

    # Phase function values
    # p_g = phase_function(cos_alpha)
    # p_g = double_henyey_greenstein(cos_alpha, **phase_function)
    p_g = double_henyey_greenstein(cos_alpha, phase_function["b"], phase_function["c"])

    # H-Function
    h_mu_0 = h_function_2(mu_0, single_scattering_albedo)
    h_mu = h_function_2(mu, single_scattering_albedo)

    # M term
    m = p_mu_0 * (h_mu - 1) + p_mu * (h_mu_0 - 1) + p * (h_mu_0 - 1) * (h_mu - 1)

    # Shadow-hiding effect
    b_sh = 1 + bs0 / (1 + tan_alpha_2 / hs)

    # Coherent backscattering effect
    hc_2 = tan_alpha_2 / hc
    b_cb = 1 + bc0 * 0.5 * (1 + (1 - jnp.exp(-hc_2)) / hc_2) / (1 + hc_2) ** 2

    # Reflectance
    albedo_independent = mu_0 / (mu_0 + mu) * s / (4 * jnp.pi) * b_cb
    p_g *= b_sh

    return (
        albedo_independent,
        p_g,
        m,
        mu_0,
        mu,
        p_mu_0,
        p_mu,
        p,
        h_mu_0,
        h_mu,
    )


# @partial(jax.jit, static_argnames=["phase_function"])
@jax.jit
def amsa_scalar(
    single_scattering_albedo: float,
    incidence_direction: ArrayLike,
    emission_direction: ArrayLike,
    surface_orientation: ArrayLike,
    phase_function: dict,
    b_n: ArrayLike,
    a_n: ArrayLike = jnp.nan,
    roughness: float = 0,
    hs: float = 0,
    bs0: float = 0,
    hc: float = 0,
    bc0: float = 0,
    refl_optimization: float = 0.0,
) -> ArrayLike:
    """
    Calculates the reflectance using the AMSA (Advanced Modified Shadowing and Coherent Backscattering) model.

    Args:
        incidence_direction: Array of shape (number_u, number_v, 3) representing the incidence direction vectors.
        emission_direction: Array of shape (number_u, number_v, 3) representing the emission direction vectors.
        surface_orientation: Array of shape (number_u, number_v, 3) representing the surface orientation vectors.
        single_scattering_albedo: Array of shape (number_u, number_v) representing the single scattering albedo values.
        phase_function: Callable function that takes the cosine of the scattering angle and returns the phase function values.
        b_n: Array of shape (n,) representing the coefficients of the Legendre expansion.
        a_n: Array of shape (n,) representing the coefficients of the Legendre expansion. Defaults to jnp.empty(1) * jnp.nan.
        hs: Float representing the shadowing parameter. Defaults to 0.
        bs0: Float representing the shadowing parameter. Defaults to 0.
        roughness: Float representing the surface roughness. Defaults to 0.
        hc: Float representing the coherent backscattering parameter. Defaults to 0.
        bc0: Float representing the coherent backscattering parameter. Defaults to 0.

    Returns:
        Array of shape (number_u, number_v) representing the reflectance values.

    Raises:
        Exception: If at least one reflectance value is not real.

    """
    with jax.profiler.TraceAnnotation("amsa_scalar"):
        (albedo_independent, p_g, m, _, _, _, _, _, _, _) = __amsa_main(
            single_scattering_albedo,
            incidence_direction,
            emission_direction,
            surface_orientation,
            phase_function,
            b_n,
            a_n,
            roughness,
            hs,
            bs0,
            hc,
            bc0,
        )

        # Reflectance
        refl = albedo_independent * single_scattering_albedo * (p_g + m)
        # refl[(mu <= 0) | (mu_0 <= 0)] = jnp.nan
        # refl[refl < 1e-6] = jnp.nan

        # Final result
        threshold_imag = 0.1
        threshold_error = 1e-4
        # arg_rh = jnp.divide(
        #     jnp.imag(refl),
        #     jnp.real(refl),
        #     out=jnp.zeros_like(refl, dtype=float),
        #     where=jnp.real(refl) != 0,
        # )
        arg_rh = jnp.where(jnp.real(refl) == 0, 0, jnp.imag(refl) / jnp.real(refl))
        refl = jnp.where(arg_rh > threshold_imag, jnp.nan, refl)

        # if jnp.any(arg_rh >= threshold_error):
        #     raise Exception("At least one reflectance value is not real!")

        refl -= refl_optimization

        return refl


amsa_vector = jax.vmap(
    amsa_scalar,
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None),
    out_axes=0,
)
amsa_image = jax.vmap(
    amsa_vector,
    in_axes=(1, 1, 1, 1, None, None, None, None, None, None, None, None),
    out_axes=1,
)


amsa_vector_opt = jax.vmap(
    amsa_scalar,
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None, 0),
    out_axes=0,
)
amsa_image_opt = jax.vmap(
    amsa_vector_opt,
    in_axes=(1, 1, 1, 1, None, None, None, None, None, None, None, None, 1),
    out_axes=1,
)


# @partial(jax.jit, static_argnames=["phase_function"])
@jax.jit
def amsa_scalar_derivative(
    single_scattering_albedo: ArrayLike,
    incidence_direction: ArrayLike,
    emission_direction: ArrayLike,
    surface_orientation: ArrayLike,
    phase_function: Callable[[ArrayLike], ArrayLike],
    b_n: ArrayLike,
    a_n: ArrayLike,
    roughness: float = 0,
    hs: float = 0,
    bs0: float = 0,
    hc: float = 0,
    bc0: float = 0,
    refl_optimization: ArrayLike = 0.0,
) -> ArrayLike:
    (
        albedo_independent,
        p_g,
        m,
        mu_0,
        mu,
        p_mu_0,
        p_mu,
        p,
        h_mu_0,
        h_mu,
    ) = __amsa_main(
        single_scattering_albedo,
        incidence_direction,
        emission_direction,
        surface_orientation,
        phase_function,
        b_n,
        a_n,
        roughness,
        hs,
        bs0,
        hc,
        bc0,
    )

    dh0_dw = h_function_2_derivative(mu_0, single_scattering_albedo)
    dh_dw = h_function_2_derivative(mu, single_scattering_albedo)

    # derivative of M term
    dm_dw = (
        p_mu_0 * dh_dw
        + p_mu * dh0_dw
        + p * (dh_dw * (h_mu_0 - 1) + dh0_dw * (h_mu - 1))
    )

    dr_dw = (p_g + m + single_scattering_albedo * dm_dw) * albedo_independent

    return dr_dw


amsa_vector_derivative = jax.vmap(
    amsa_scalar_derivative,
    in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None),
    out_axes=0,
)
amsa_image_derivative = jax.vmap(
    amsa_vector_derivative,
    in_axes=(1, 1, 1, 1, None, None, None, None, None, None, None, None),
    out_axes=1,
)
