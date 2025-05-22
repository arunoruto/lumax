# python -m scripts.benchmark.amsa --copy-env -o bench.json
import warnings

import jax
import numpy as np
import pyperf
from astropy.io import fits

from refmod.dtm_helper import dtm2grad
from refmod.hapke import double_henyey_greenstein
from refmod.hapke.amsa import amsa_image
from refmod.hapke.legendre import coef_a, coef_b

warnings.simplefilter("ignore")

DATA_DIR = "test/data"

file_name = f"{DATA_DIR}/hopper_amsa.fits"
f = fits.open(file_name)

result = f["result"].data.astype(float)
i = np.deg2rad(f["result"].header["i"])
e = np.deg2rad(f["result"].header["e"])
b = f["result"].header["b"]
c = f["result"].header["c"]
hs = f["result"].header["hs"]
bs0 = f["result"].header["bs0"]
tb = f["result"].header["tb"]
hc = f["result"].header["hc"]
bc0 = f["result"].header["bc0"]
albedo = f["albedo"].data.astype(float)
dtm = f["dtm"].data.astype(float)
resolution = f["dtm"].header["res"]

n = dtm2grad(dtm, resolution, normalize=False)

u, v = result.shape

i = np.reshape([np.sin(i), 0, np.cos(i)], [1, 1, -1])
e = np.reshape([np.sin(e), 0, np.cos(e)], [1, 1, -1])
i = np.tile(i, (u, v, 1))
e = np.tile(e, (u, v, 1))

N = 15
a_n = coef_a(N)
b_n = coef_b(b, c, N)


# @jax.jit
# def phase_function(x):
#     return double_henyey_greenstein(x, b, c)
phase_function = dict(b=b, c=c)

# amsa1 = jax.vmap(
#     amsa, (0, 0, 0, 0, None, None, None, None, None, None, None, None), 0
# )

# amsa2 = jax.vmap(
#     amsa1, (1, 1, 1, 1, None, None, None, None, None, None, None, None), 1
# )

runner = pyperf.Runner(values=1, processes=5)
runner.bench_func(
    "AMSA",
    lambda: (
        amsa_image(
            albedo,
            i,
            e,
            n,
            phase_function,
            b_n,
            a_n,
            tb,
            hs,
            bs0,
            hc,
            bc0,
        )
    ),
)
