#! /usr/bin/env python

import jax
from jax.extend import backend

print(jax.devices())
print(backend.get_backend().platform)
