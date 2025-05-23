import os

import jax
import jax.numpy as jnp
import jax.profiler as profiler

LOG_DIR = "./tensorboard_logs_test_scope"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


@jax.jit
def foo(x):
    with jax.named_scope("my_inner_math"):
        y = jnp.sin(x) * 2.0
        z = jnp.cos(y) + x
    return z


@jax.jit
def bar(x):
    with jax.named_scope("my_outer_computation"):
        res1 = foo(x * 0.5)
        with jax.named_scope("another_section"):
            res2 = jnp.tanh(res1)
    return res2


print("Running minimal test with named_scope...")
key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (1000, 1000))

# Warm-up
_ = bar(data).block_until_ready()

profiler.start_trace(LOG_DIR)
result = bar(data)
result.block_until_ready()
profiler.stop_trace()

print(f"Profiling done. Check TensorBoard/Perfetto with logs in {LOG_DIR}")
