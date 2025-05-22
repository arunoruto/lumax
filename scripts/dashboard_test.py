from glob import glob

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from astropy.io import fits
from plotly.subplots import make_subplots

from refmod.dtm_helper import dtm2grad
from refmod.hapke import amsa, double_henyey_greenstein, imsa
from refmod.hapke.legendre import coef_a, coef_b

# from refmod.hapke.helper import microscopic_roughness

TEST_FITS_FILES = "test/data/*.fits"

st.set_page_config(layout="wide")

with st.sidebar:
    file = st.selectbox(
        "File", glob(TEST_FITS_FILES), format_func=lambda x: x.split("/")[-1]
    )

file = file or ""

f = fits.open(file)
result = f["result"].data
result[result < 1e-5] = np.nan
dtm = f["dtm"].data
dtm_resolution = f["dtm"].header["res"]
albedo = f["albedo"].data
# python calculations
u = result.shape[0]
v = result.shape[1]

i = np.deg2rad(f["result"].header["i"])
e = np.deg2rad(f["result"].header["e"])
i = np.reshape([np.sin(i), 0, np.cos(i)], [1, 1, -1])
e = np.reshape([np.sin(e), 0, np.cos(e)], [1, 1, -1])
i = np.tile(i, (u, v, 1))
e = np.tile(e, (u, v, 1))

b = f["result"].header["b"]
c = f["result"].header["c"]

h = f["result"].header["hs"]
b0 = f["result"].header["bs0"]

tb = f["result"].header["tb"]

n = dtm2grad(dtm, dtm_resolution, normalize=False)
# normal = np.moveaxis(f["normal"].data, 0, -1)

# refl = imsa(i, e, n, albedo, lambda x: double_henyey_greenstein(x, b, c), h, b0, tb)
# s, mu0, mu = microscopic_roughness(tb, i, e, normal)
if "imsa" in file:
    refl = imsa(i, e, n, albedo, lambda x: double_henyey_greenstein(x, b, c), h, b0, tb)
elif "amsa" in file:
    a_n = coef_a()
    b_n = coef_b(b, c)
    hc = f["result"].header["hc"]
    bc0 = f["result"].header["bc0"]
    refl = amsa(
        i,
        e,
        n,
        albedo,
        lambda x: double_henyey_greenstein(x, b, c),
        b_n,
        a_n,
        h,
        b0,
        tb,
        hc,
        bc0,
    )
else:
    raise Exception("File needs to be imsa or amsa")

st.write("Result")
fig = make_subplots(1, 2)
fig.add_trace(go.Heatmap(z=result, colorscale="Jet"), row=1, col=1)
fig.add_trace(go.Heatmap(z=refl, colorscale="Jet"), row=1, col=2)
fig.update_layout(
    yaxis=dict(
        scaleanchor="x",
        scaleratio=1,
    ),
    yaxis2=dict(
        scaleanchor="x2",
        scaleratio=1,
    ),
)
st.plotly_chart(fig, use_container_width=True)

st.write("Ratio")
fig = go.Figure()
fig.add_trace(go.Heatmap(z=(refl / result - 1), colorscale="Jet"))
fig.update_layout(
    height=900,
    yaxis=dict(
        scaleanchor="x",
        scaleratio=1,
    ),
)
st.plotly_chart(fig, use_container_width=True)

# st.write("Albedo")
# fig = go.Figure()
# fig.add_trace(go.Heatmap(z=albedo, colorscale="Jet"))
# fig.update_layout(
#     yaxis=dict(
#         scaleanchor="x",
#         scaleratio=1,
#     )
# )
# st.plotly_chart(fig, use_container_width=True)

# st.write("DTM")
# fig = go.Figure()
# fig.add_trace(go.Surface(z=dtm))
# fig.update_layout(
#     height=900,
#     scene=dict(
#         aspectratio=dict(
#             x=dtm_resolution,
#             y=dtm_resolution,
#             z=1,
#         ),
#     ),
#     scene_camera=dict(eye=dict(x=dtm_resolution, y=dtm_resolution, z=dtm_resolution)),
# )
# st.plotly_chart(fig, use_container_width=True)
