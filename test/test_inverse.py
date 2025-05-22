import numpy as np
from astropy.io import fits
from refmod.dtm_helper import dtm2grad
from refmod.hapke import double_henyey_greenstein

# from refmod.hapke.amsa import amsa
from refmod.hapke.amsa import amsa_image
from refmod.hapke.imsa import imsa
from refmod.hapke.legendre import coef_a, coef_b
from refmod.inverse import inverse_model
from scipy.optimize import least_squares

DATA_DIR = "test/data"
RTOL = 1e-12


def test_inverse_amsa():
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
    r = 20
    uc = u // 2 + np.arange(-r, r + 1)
    vc = v // 2 + np.arange(-r, r + 1)
    # uc, vc = np.meshgrid(uc, vc, indexing="ij")

    i = np.reshape([np.sin(i), 0, np.cos(i)], [1, 1, -1])
    e = np.reshape([np.sin(e), 0, np.cos(e)], [1, 1, -1])
    i = np.tile(i, (u, v, 1))
    e = np.tile(e, (u, v, 1))

    a_n = coef_a()
    b_n = coef_b(b, c)
    refl = amsa_image(
        albedo[uc, :][:, vc],
        i[uc, :, :][:, vc, :],
        e[uc, :, :][:, vc, :],
        n[uc, :, :][:, vc, :],
        # lambda x: double_henyey_greenstein(x, b, c),
        dict(b=b, c=c),
        b_n,
        a_n,
        tb,
        hs,
        bs0,
        hc,
        bc0,
    )
    albedo_recon = inverse_model(
        refl,
        i[uc, :, :][:, vc, :],
        e[uc, :, :][:, vc, :],
        n[uc, :, :][:, vc, :],
        # lambda x: double_henyey_greenstein(x, b, c),
        dict(b=b, c=c),
        b_n,
        a_n,
        tb,
        hs,
        bs0,
        hc,
        bc0,
    )

    # np.testing.assert_allclose(albedo_recon, albedo[uc, :][:, vc], rtol=1e-20)
    np.testing.assert_allclose(albedo_recon, albedo[uc, :][:, vc], rtol=RTOL)
