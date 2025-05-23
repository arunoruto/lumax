import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.profiler as profiler
import numpy as np  # For initial data loading and some non-JAX ops
from astropy.io import fits

# Lumax imports
from lumax.dtm_helper import dtm2grad
from lumax.inverse import inverse_model
from lumax.models.hapke.amsa import amsa_image  # Needed to generate 'refl'
from lumax.models.hapke.legendre import coef_a, coef_b

# Ensure the lumax package is findable if running script from a subdirectory
# This adds the project root to the Python path. Adjust if your script is elsewhere.
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# if PROJECT_ROOT not in sys.path:
#    sys.path.insert(0, PROJECT_ROOT)


# --- Configuration ---
# Adjust this path if your script is not in the project root or a similar level
# For example, if script is in scripts/profiling/, DATA_DIR might be "../../test/data"
DATA_DIR = "test/data"
FITS_FILE_NAME = "hopper_amsa.fits"
PROFILE_LOG_DIR = "./tensorboard_logs_inverse_amsa"  # Directory for profiler output
# Cropping parameter from your test
CROP_RADIUS = 20


# --- Helper function to load and prepare data (adapted from your test) ---
def setup_data_and_params():
    """Loads data from FITS, prepares parameters, and generates input reflectance."""
    print(f"Loading data from: {os.path.join(DATA_DIR, FITS_FILE_NAME)}")
    fits_path = os.path.join(DATA_DIR, FITS_FILE_NAME)
    if not os.path.exists(fits_path):
        print(f"ERROR: FITS file not found at {fits_path}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)

    with fits.open(fits_path) as f:
        result_np = f["result"].data.astype(float)
        i_angle_rad = np.deg2rad(f["result"].header["i"])
        e_angle_rad = np.deg2rad(f["result"].header["e"])
        b_param = f["result"].header["b"]
        c_param = f["result"].header["c"]
        hs_param = f["result"].header["hs"]
        bs0_param = f["result"].header["bs0"]
        tb_param = f["result"].header["tb"]
        hc_param = f["result"].header["hc"]
        bc0_param = f["result"].header["bc0"]
        albedo_np = f["albedo"].data.astype(float)
        dtm_np = f["dtm"].data.astype(float)
        resolution_val = f["dtm"].header["res"]

    print("Data loaded. Preparing parameters...")
    # dtm2grad uses numpy.gradient, returns numpy arrays
    n_full_np = dtm2grad(dtm_np, resolution_val, normalize=False)

    u_full, v_full = result_np.shape

    # Define slices for cropping
    uc_center, vc_center = u_full // 2, v_full // 2
    uc_slice = slice(uc_center - CROP_RADIUS, uc_center + CROP_RADIUS + 1)
    vc_slice = slice(vc_center - CROP_RADIUS, vc_center + CROP_RADIUS + 1)

    num_rows_cropped = uc_slice.stop - uc_slice.start
    num_cols_cropped = vc_slice.stop - vc_slice.start

    # Prepare i and e vectors (constant for the cropped region)
    i_vec_np = np.reshape([np.sin(i_angle_rad), 0, np.cos(i_angle_rad)], [1, 1, 3])
    e_vec_np = np.reshape([np.sin(e_angle_rad), 0, np.cos(e_angle_rad)], [1, 1, 3])

    i_tiled_np = np.tile(i_vec_np, (num_rows_cropped, num_cols_cropped, 1))
    e_tiled_np = np.tile(e_vec_np, (num_rows_cropped, num_cols_cropped, 1))

    # Crop the main arrays using NumPy slicing
    albedo_cropped_np = albedo_np[uc_slice, vc_slice]
    n_cropped_np = n_full_np[uc_slice, vc_slice, :]

    # Convert to JAX arrays before passing to JAX-based Lumax functions
    print("Converting relevant arrays to JAX arrays...")
    albedo_cropped_jnp = jnp.asarray(albedo_cropped_np)
    i_cropped_jnp = jnp.asarray(i_tiled_np)
    e_cropped_jnp = jnp.asarray(e_tiled_np)
    n_cropped_jnp = jnp.asarray(n_cropped_np)

    # Hapke parameters (coef_a and coef_b should return JAX arrays)
    a_n_val = coef_a()
    b_n_val = coef_b(b_param, c_param)
    phase_params_dict = dict(b=b_param, c=c_param)  # Python dict with floats is fine

    print("Calculating reference reflectance (refl) using amsa_image...")
    # amsa_image is JITted and expects JAX arrays for its main inputs
    # This step is important as 'refl_jnp' is the input to 'inverse_model'
    start_refl_calc = time.time()
    refl_jnp = amsa_image(
        albedo_cropped_jnp,
        i_cropped_jnp,
        e_cropped_jnp,
        n_cropped_jnp,
        phase_params_dict,
        b_n_val,
        a_n_val,
        tb_param,
        hs_param,
        bs0_param,
        hc_param,
        bc0_param,
    )
    refl_jnp.block_until_ready()  # Ensure this calculation is done before timing/profiling inversion
    end_refl_calc = time.time()
    print(
        f"Reference reflectance calculation complete. Took {end_refl_calc - start_refl_calc:.2f}s"
    )

    return (
        refl_jnp,
        i_cropped_jnp,
        e_cropped_jnp,
        n_cropped_jnp,
        phase_params_dict,
        b_n_val,
        a_n_val,
        tb_param,
        hs_param,
        bs0_param,
        hc_param,
        bc0_param,
        albedo_cropped_jnp,  # True albedo for optional verification
    )


# --- Main profiling function ---
def run_profiling():
    # Ensure JAX is using the desired device (optional, good for consistency)
    try:
        print(f"JAX default backend: {jax.default_backend()}")
        print(f"JAX available devices: {jax.devices()}")
        # You can force a device, e.g., jax.config.update("jax_platform_name", "cpu")
    except Exception as e:
        print(f"Could not query JAX devices: {e}")

    # 1. Setup data and parameters
    (
        refl_jnp,
        i_jnp,
        e_jnp,
        n_jnp,
        phase_dict,
        b_n,
        a_n,
        tb,
        hs,
        bs0,
        hc,
        bc0,
        albedo_true_jnp,  # For verification
    ) = setup_data_and_params()

    print(f"\nStarting JAX profiler. Log directory: {PROFILE_LOG_DIR}")
    if not os.path.exists(PROFILE_LOG_DIR):
        os.makedirs(PROFILE_LOG_DIR)

    # It's good practice to JIT compile the function to be profiled once outside the profiler
    # This separates compilation time from execution time in the profile.
    print("JIT compiling inverse_model (first run)...")
    try:
        # Create a dummy x0 of the correct shape and type.
        # inverse_model creates its x0 based on refl, so this is more about warming up the args structure.
        _ = inverse_model(
            refl_jnp,
            i_jnp,
            e_jnp,
            n_jnp,
            phase_dict,
            b_n,
            a_n,
            tb,
            hs,
            bs0,
            hc,
            bc0,
        ).block_until_ready()  # pyright: ignore
        print("JIT compilation run complete.")
    except Exception as e:
        print(f"Error during JIT compilation run: {e}")
        # Depending on the error, you might want to stop or continue.
        # For now, we'll continue to try profiling execution.

    # 2. Start JAX Profiler
    print("Running inverse_model for profiling...")
    # profiler.start_trace(PROFILE_LOG_DIR)
    start_inversion_time = time.time()
    with profiler.trace(
        PROFILE_LOG_DIR,
        # create_perfetto_link=True,
        # create_perfetto_trace=True,
    ):
        # 3. Run the function to be profiled

        albedo_recon_jnp = inverse_model(
            refl_jnp,
            i_jnp,
            e_jnp,
            n_jnp,
            phase_dict,
            b_n,
            a_n,
            tb,
            hs,
            bs0,
            hc,
            bc0,
        )

        # 4. CRUCIAL: Block until JAX computations are complete
        albedo_recon_jnp.block_until_ready()  # pyright: ignore
    end_inversion_time = time.time()

    # 5. Stop JAX Profiler
    # profiler.stop_trace()
    print(
        f"inverse_model execution complete. Took {end_inversion_time - start_inversion_time:.2f}s (wall time)."
    )

    print("\n--- Profiling Summary ---")
    print(f"Profile data saved to: {PROFILE_LOG_DIR}")
    print("To view the trace, run the following command in your terminal:")
    print(f"  tensorboard --logdir {PROFILE_LOG_DIR}")
    print(
        "Then open the URL provided by TensorBoard (usually http://localhost:6006/) in your browser."
    )

    # Optional: Quick verification of the result (can be commented out)
    print("\nVerifying inversion result (optional check)...")
    try:
        # Convert JAX arrays to NumPy for assertion if needed, or use jnp directly for checks
        np.testing.assert_allclose(
            np.asarray(albedo_recon_jnp),
            np.asarray(albedo_true_jnp),  # albedo_true_jnp is already cropped
            rtol=1e-5,  # Relaxed tolerance for quick check
        )
        print(
            "SUCCESS: Inversion result matches expected albedo (within relaxed tolerance)."
        )
    except AssertionError as e_assert:
        print(f"WARNING: Inversion result mismatch: {e_assert}")
        # For debugging, print some values:
        # print("Reconstructed (sample):", np.asarray(albedo_recon_jnp).flatten()[:5])
        # print("True (sample):       ", np.asarray(albedo_true_jnp).flatten()[:5])
        # diff = np.abs(np.asarray(albedo_recon_jnp) - np.asarray(albedo_true_jnp))
        # print("Max difference:", np.max(diff))


if __name__ == "__main__":
    run_profiling()
