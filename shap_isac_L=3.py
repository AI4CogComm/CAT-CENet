"""
SHAP Analysis for XL-MIMO Near-Field ISAC Channel Estimation

Scenario
--------
L = 3 : number of communication multipath components
K = 3 : number of sensing targets
X = 3 : number of shared communication-sensing scatterers

Test dataset
------------
Pnear_overlap3_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_2000.mat

MAT variables
-------------
Channel_mat
Channel_near_Radar_overlap3_mat

SHAP procedure
--------------
1. Load the trained dual-input ISAC model.
2. Aggregate the original channel output into one scalar per sample.
3. Use SHAP GradientExplainer.
4. Calculate mean absolute SHAP contributions for:
   - communication input
   - sensing target 1
   - sensing target 2
   - sensing target 3
5. Visualize the four contribution heatmaps with a unified color scale.
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import shap
import tensorflow as tf
from tensorflow.keras.models import Model, load_model


# ============================================================================
# GPU Configuration
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        print(error)


# ============================================================================
# TensorFlow / SHAP Compatibility
# ============================================================================

# Keep the original graph-mode setting used by this SHAP implementation.
tf.compat.v1.disable_eager_execution()


# ============================================================================
# ISAC Scenario Configuration
# ============================================================================

N = 256
NX = 16
NY = 16

L = 3
K = 3
X = 3

assert X <= min(L, K), "X must satisfy X <= min(L, K)."

# Each complex sensing target is represented by real and imaginary parts.
# K=3 -> [Re1, Im1, Re2, Im2, Re3, Im3]
SENSING_CHANNELS = 2 * K


# ============================================================================
# SHAP Configuration
# ============================================================================

SNR_DB = -5
TEST_SAMPLES = 2000

BACKGROUND_SIZE = 100
NUM_EXPLAIN_SAMPLES = 20

MODEL_PATH = Path(
    "checkpoints/isac_L3_K3_X3_-5dB.hdf5"
)

TEST_DATASET_PATH = Path(
    "data/Pnear_overlap3_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_2000.mat"
)

SENSING_KEY = "Channel_near_Radar_overlap3_mat"

FIGURE_SAVE_PATH = Path(
    "shap_L3_K3_X3_breakdown.png"
)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(filepath, data_num, nx, ny, snr_db):
    """
    Load the independent test dataset and generate noisy communication input.

    Parameters
    ----------
    filepath : str or Path
        Path to the MATLAB test dataset.
    data_num : int
        Number of test samples.
    nx, ny : int
        2-D antenna reshape dimensions.
    snr_db : float
        Communication input SNR in dB.

    Returns
    -------
    noisy_input : ndarray
        Noisy communication channel, shape (data_num, 16, 16, 2).
    true_output : ndarray
        Clean communication channel, shape (data_num, 16, 16, 2).
    sensing_input : ndarray
        Sensing input, shape (data_num, 16, 16, 6).
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    data = sio.loadmat(filepath)

    if "Channel_mat" not in data:
        raise KeyError(
            "Variable 'Channel_mat' was not found in the dataset."
        )

    if SENSING_KEY not in data:
        raise KeyError(
            f"Variable '{SENSING_KEY}' was not found in the dataset."
        )

    channel = data["Channel_mat"]
    sensing = data[SENSING_KEY]

    if data_num > channel.shape[0]:
        raise ValueError(
            f"Requested {data_num} samples, "
            f"but dataset contains {channel.shape[0]} samples."
        )

    channel = channel[:data_num]
    sensing = sensing[:data_num, :, :K]

    # ------------------------------------------------------------------------
    # Sensing input:
    # [Re(target1), Im(target1),
    #  Re(target2), Im(target2),
    #  Re(target3), Im(target3)]
    # ------------------------------------------------------------------------

    real_part = np.real(sensing)
    imaginary_part = np.imag(sensing)

    sensing_input = np.stack(
        [
            real_part[:, :, 0],
            imaginary_part[:, :, 0],
            real_part[:, :, 1],
            imaginary_part[:, :, 1],
            real_part[:, :, 2],
            imaginary_part[:, :, 2],
        ],
        axis=-1,
    )

    sensing_input = np.reshape(
        sensing_input,
        (data_num, nx, ny, SENSING_CHANNELS),
    )

    # ------------------------------------------------------------------------
    # Communication input with AWGN
    # ------------------------------------------------------------------------

    snr_linear = 10 ** (snr_db / 10.0)

    noisy_input = np.zeros(
        (data_num, nx, ny, 2),
        dtype=float,
    )

    true_output = np.zeros(
        (data_num, nx, ny, 2),
        dtype=float,
    )

    for i in range(data_num):
        h = channel[i]
        H = np.reshape(h, (nx, ny))

        true_output[i, :, :, 0] = np.real(H)
        true_output[i, :, :, 1] = np.imag(H)

        noise = (
            np.random.randn(nx, ny)
            + 1j * np.random.randn(nx, ny)
        ) / np.sqrt(2)

        H_noisy = H + noise / np.sqrt(snr_linear)

        noisy_input[i, :, :, 0] = np.real(H_noisy)
        noisy_input[i, :, :, 1] = np.imag(H_noisy)

    print("\n================ Dataset Information ================")
    print(f"L                : {L}")
    print(f"K                : {K}")
    print(f"X                : {X}")
    print(f"SNR              : {snr_db} dB")
    print(f"Samples          : {data_num}")
    print(f"Communication    : {noisy_input.shape}")
    print(f"Sensing          : {sensing_input.shape}")
    print(f"Ground truth     : {true_output.shape}")
    print("=====================================================\n")

    return noisy_input, true_output, sensing_input


# ============================================================================
# Scalar-Output Wrapper
# ============================================================================

def build_scalar_output_model(base_model):
    """
    Aggregate the original output (batch, 16, 16, 2)
    into one scalar value per sample.

    This follows the original SHAP implementation:
        scalar_output = sum(H_est)
    """

    original_output = base_model.outputs[0]

    scalar_output = tf.reduce_sum(
        original_output,
        axis=[1, 2, 3],
    )

    scalar_model = Model(
        inputs=base_model.inputs,
        outputs=scalar_output,
    )

    return scalar_model


# ============================================================================
# SHAP Analysis
# ============================================================================

def shap_explain():
    """
    Run SHAP GradientExplainer for the L=3, K=3, X=3 model.
    """

    # ------------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------------

    print("\n===== 1. Load trained model =====")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    base_model = load_model(
        MODEL_PATH,
        compile=False,
    )

    # ------------------------------------------------------------------------
    # 2. Build scalar-output model
    # ------------------------------------------------------------------------

    print("===== 2. Build scalar-output model =====")

    model = build_scalar_output_model(
        base_model
    )

    # ------------------------------------------------------------------------
    # 3. Load independent test dataset
    # ------------------------------------------------------------------------

    print("===== 3. Load ISAC test dataset =====")

    noisy_input, _, sensing_input = load_dataset(
        filepath=TEST_DATASET_PATH,
        data_num=TEST_SAMPLES,
        nx=NX,
        ny=NY,
        snr_db=SNR_DB,
    )

    # ------------------------------------------------------------------------
    # 4. Prepare SHAP background data
    # ------------------------------------------------------------------------

    print(
        f"===== 4. Initialize GradientExplainer "
        f"(background={BACKGROUND_SIZE}) ====="
    )

    background_ids = np.random.choice(
        noisy_input.shape[0],
        BACKGROUND_SIZE,
        replace=False,
    )

    background_noisy = noisy_input[background_ids]
    background_sensing = sensing_input[background_ids]

    explainer = shap.GradientExplainer(
        model,
        [
            background_noisy,
            background_sensing,
        ],
    )

    # ------------------------------------------------------------------------
    # 5. Calculate SHAP values
    # ------------------------------------------------------------------------

    explain_noisy = noisy_input[
        :NUM_EXPLAIN_SAMPLES
    ]

    explain_sensing = sensing_input[
        :NUM_EXPLAIN_SAMPLES
    ]

    print(
        f"===== 5. Compute SHAP values "
        f"(samples={NUM_EXPLAIN_SAMPLES}) ====="
    )

    start_time = time.time()

    shap_values = explainer.shap_values(
        [
            explain_noisy,
            explain_sensing,
        ]
    )

    elapsed_time = time.time() - start_time

    print(
        f"SHAP completed in {elapsed_time:.2f} seconds."
    )

    # Keep the original SHAP return-value handling.
    shap_noisy = shap_values[0]
    shap_sensing = shap_values[1]

    # ------------------------------------------------------------------------
    # 6. Separate the three sensing targets
    # ------------------------------------------------------------------------

    # sensing channels:
    # [Re1, Im1, Re2, Im2, Re3, Im3]

    shap_target1 = shap_sensing[:, :, :, 0:2]
    shap_target2 = shap_sensing[:, :, :, 2:4]
    shap_target3 = shap_sensing[:, :, :, 4:6]

    # ------------------------------------------------------------------------
    # 7. Global contribution analysis
    # ------------------------------------------------------------------------

    importance_comm = np.mean(
        np.abs(shap_noisy)
    )

    importance_t1 = np.mean(
        np.abs(shap_target1)
    )

    importance_t2 = np.mean(
        np.abs(shap_target2)
    )

    importance_t3 = np.mean(
        np.abs(shap_target3)
    )

    total_importance = (
        importance_comm
        + importance_t1
        + importance_t2
        + importance_t3
        + 1e-12
    )

    contribution_comm = (
        importance_comm / total_importance * 100
    )

    contribution_t1 = (
        importance_t1 / total_importance * 100
    )

    contribution_t2 = (
        importance_t2 / total_importance * 100
    )

    contribution_t3 = (
        importance_t3 / total_importance * 100
    )

    print(
        "\n================ SHAP Contribution Analysis ================"
    )

    print(
        f"Communication input : {importance_comm:.6e} "
        f"({contribution_comm:.2f}%)"
    )

    print(
        f"Sensing target 1    : {importance_t1:.6e} "
        f"({contribution_t1:.2f}%)"
    )

    print(
        f"Sensing target 2    : {importance_t2:.6e} "
        f"({contribution_t2:.2f}%)"
    )

    print(
        f"Sensing target 3    : {importance_t3:.6e} "
        f"({contribution_t3:.2f}%)"
    )

    print(
        "============================================================\n"
    )

    # ------------------------------------------------------------------------
    # 8. SHAP heatmaps
    # ------------------------------------------------------------------------

    sample_idx = 0

    heatmap_comm = np.sum(
        np.abs(shap_noisy[sample_idx]),
        axis=-1,
    )

    heatmap_t1 = np.sum(
        np.abs(shap_target1[sample_idx]),
        axis=-1,
    )

    heatmap_t2 = np.sum(
        np.abs(shap_target2[sample_idx]),
        axis=-1,
    )

    heatmap_t3 = np.sum(
        np.abs(shap_target3[sample_idx]),
        axis=-1,
    )

    heatmaps = [
        heatmap_comm,
        heatmap_t1,
        heatmap_t2,
        heatmap_t3,
    ]

    titles = [
        f"Communication Input\n"
        f"({contribution_comm:.1f}%)",

        f"Sensing Target 1\n"
        f"({contribution_t1:.1f}%)",

        f"Sensing Target 2\n"
        f"({contribution_t2:.1f}%)",

        f"Sensing Target 3\n"
        f"({contribution_t3:.1f}%)",
    ]

    # Unified color scale for all four heatmaps.
    global_vmin = min(
        image.min() for image in heatmaps
    )

    global_vmax = max(
        image.max() for image in heatmaps
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(28, 6),
    )

    fig.suptitle(
        "SHAP Value Heatmaps with Unified Color Scale",
        fontsize=16,
        y=0.98,
    )

    image_handle = None

    for ax, heatmap, title in zip(
        axes,
        heatmaps,
        titles,
    ):
        image_handle = ax.imshow(
            heatmap,
            cmap="jet",
            aspect="auto",
            vmin=global_vmin,
            vmax=global_vmax,
        )

        ax.set_title(
            title,
            fontsize=14,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    colorbar = fig.colorbar(
        image_handle,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        pad=0.02,
    )

    colorbar.set_label(
        "SHAP Value Magnitude",
        fontsize=12,
    )

    plt.savefig(
        FIGURE_SAVE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"SHAP figure saved to: {FIGURE_SAVE_PATH}"
    )

    return (
        shap_noisy,
        shap_target1,
        shap_target2,
        shap_target3,
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    shap_explain()
