import os

import jax
import numpy as np
from astropy.io import fits
from lumax.dtm_helper import dtm2grad
from lumax.models.hapke import double_henyey_greenstein
from lumax.models.hapke.amsa import amsa_image, amsa_image_derivative
from lumax.models.hapke.imsa import imsa
from lumax.models.hapke.legendre import coef_a, coef_b

# jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

DATA_DIR = "test/data"
EXTENSION = "fits"
# EXTENSION = "mat"
RTOL = 1e-3 if os.getenv("JAX_ENABLE_X64") else 1e-12

# def test_imsa_hopper():
#     file_name = f"{DATA_DIR}/hopper_imsa.fits"
#     f = fits.open(file_name)

#     result = f["result"].data.astype(float)
#     i = np.deg2rad(f["result"].header["i"])
#     e = np.deg2rad(f["result"].header["e"])
#     b = f["result"].header["b"]
#     c = f["result"].header["c"]
#     h = f["result"].header["hs"]
#     b0 = f["result"].header["bs0"]
#     tb = f["result"].header["tb"]
#     albedo = f["albedo"].data.astype(float)
#     dtm = f["dtm"].data.astype(float)
#     resolution = f["dtm"].header["res"]

#     n = dtm2grad(dtm, resolution, normalize=False)

#     u = result.shape[0]
#     v = result.shape[1]

#     i = np.reshape([np.sin(i), 0, np.cos(i)], [1, 1, -1])
#     e = np.reshape([np.sin(e), 0, np.cos(e)], [1, 1, -1])
#     i = np.tile(i, (u, v, 1))
#     e = np.tile(e, (u, v, 1))

#     refl = imsa(
#         i, e, n, albedo, lambda x: double_henyey_greenstein(x, b, c), h, b0, tb
#     )
#     result[np.isnan(refl)] = np.nan
#     np.testing.assert_allclose(refl, result)
#     # np.testing.assert_allclose(refl, result, rtol=1e-20)


def test_amsa_hopper():
    file_name = f"{DATA_DIR}/hopper_amsa.fits"
    f = fits.open(file_name)

    result = f["result"].data.astype(float)
    derivative = f["derivative"].data
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

    a_n = coef_a()
    b_n = coef_b(b, c)

    # @jax.jit
    # def phase_function(x):
    #     return double_henyey_greenstein(x, b, c)
    phase_function = dict(b=b, c=c)

    refl = amsa_image(
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
    result[np.isnan(refl)] = np.nan
    # np.testing.assert_allclose(refl, result)
    np.testing.assert_allclose(refl, result, rtol=RTOL)

    dr_dw = amsa_image_derivative(
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
    derivative[np.isnan(dr_dw)] = np.nan
    np.testing.assert_allclose(dr_dw, derivative, rtol=RTOL)
