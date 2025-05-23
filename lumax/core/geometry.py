import jax.numpy as jnp
from jaxtyping import Array, Float


def normalize_vec(x: Float[Array, "3 dim"]) -> Float[Array, "3 dim"]:
    return x / jnp.expand_dims(
        jnp.sqrt(x[0, ...] ** 2 + x[1, ...] ** 2 + x[2, ...] ** 2), axis=0
    )
