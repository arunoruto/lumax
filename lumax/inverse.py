import jax
import jax.numpy as np
import optimistix as optx
from jax.typing import ArrayLike

from refmod.hapke import amsa_image_opt

# from refmod.hapke import amsa_image_derivative


@jax.jit
def amsa_optx_wrapper(x, args):
    return amsa_image_opt(x, *args)


def inverse_model(
    refl: ArrayLike,
    incidence_direction: ArrayLike,
    emission_direction: ArrayLike,
    surface_orientation: ArrayLike,
    # phase_function: Callable[[ArrayLike], ArrayLike],
    phase_function: dict,
    b_n: ArrayLike,
    a_n: ArrayLike,
    roughness: float = 0,
    hs: float = 0,
    bs0: float = 0,
    hc: float = 0,
    bc0: float = 0,
) -> ArrayLike:
    x0 = np.ones_like(refl) / 3

    args = (
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
        refl,
    )
    solver = optx.LevenbergMarquardt(
        rtol=1e-8,
        atol=1e-8,
        # norm=optx.rms_norm,
    )
    sol = optx.least_squares(amsa_optx_wrapper, solver, x0, args)
    return sol.value
