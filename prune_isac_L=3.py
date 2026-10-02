"""
Pruning for XL-MIMO Near-Field ISAC Channel Estimation

Scenario:
    L = 3  : number of communication multipath components
    K = 3  : number of sensing/radar targets
    X = 3  : number of shared communication-sensing scatterers

Datasets:
    Training:
        Pnear_overlap3_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_50000.mat
    Testing:
        Pnear_overlap3_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_2000.mat

Pruning strategy:
    - Prune Conv2D and Dense layers only.
    - Polynomial sparsity schedule.
    - Final target sparsity: 80%.
    - Fine-tune the pruned model once.
    - Use 50,000 training samples.
    - Evaluate on 2,000 independent test samples.
"""

import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ============================================================================
# GPU Configuration
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        print(f"GPU configuration error: {error}")


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

# K=3 complex sensing targets
# -> [Re1, Im1, Re2, Im2, Re3, Im3]
SENSING_CHANNELS = 2 * K


# ============================================================================
# Pruning Configuration
# ============================================================================

TRAIN_SNR_DB = -5
TEST_SNR_DB = -5

TRAIN_SAMPLES = 50000
TEST_SAMPLES = 2000

BATCH_SIZE = 128
EPOCHS = 35
VALIDATION_SPLIT = 0.1

PRUNING_LEARNING_RATE = 1e-4

INITIAL_SPARSITY = 0.0
FINAL_SPARSITY = 0.80

# 90% of the 50,000 samples are used for pruning fine-tuning,
# because validation_split = 0.1.
NUM_TRAIN_SAMPLES = int(
    TRAIN_SAMPLES * (1.0 - VALIDATION_SPLIT)
)

STEPS_PER_EPOCH = int(
    np.ceil(NUM_TRAIN_SAMPLES / BATCH_SIZE)
)

END_STEP = STEPS_PER_EPOCH * EPOCHS


# ============================================================================
# Paths
# ============================================================================

TRAIN_DATASET_PATH = Path(
    "data/Pnear_overlap3_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_50000.mat"
)

TEST_DATASET_PATH = Path(
    "data/Pnear_overlap3_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_2000.mat"
)

SENSING_KEY = "Channel_near_Radar_overlap3_mat"

# Dense model trained before pruning.
ORIGINAL_MODEL_PATH = Path(
    "checkpoints/isac_L3_K3_X3_-5dB.hdf5"
)

# Best checkpoint during pruning fine-tuning.
PRUNED_CHECKPOINT_PATH = Path(
    "checkpoints/isac_L3_K3_X3_-5dB_pruning_80pct.hdf5"
)

# Final exported sparse model after strip_pruning.
PRUNED_FINAL_PATH = Path(
    "checkpoints/isac_L3_K3_X3_-5dB_pruned_final_80pct.hdf5"
)

PRUNING_LOG_DIR = Path(
    "pruning_logs/L3_K3_X3"
)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(filepath, data_num, nx, ny, snr_db):
    """
    Load ISAC data and generate noisy communication observations.

    Returns
    -------
    noisy_input : ndarray
        Shape: (data_num, 16, 16, 2)

    true_output : ndarray
        Shape: (data_num, 16, 16, 2)

    sensing_input : ndarray
        Shape: (data_num, 16, 16, 6)
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

    if channel.shape[1] != N:
        raise ValueError(
            f"Expected Channel_mat shape (*, {N}), "
            f"but got {channel.shape}."
        )

    if sensing.shape[1] != N or sensing.shape[2] != K:
        raise ValueError(
            f"Expected sensing shape (*, {N}, {K}), "
            f"but got {sensing.shape}."
        )

    if data_num > channel.shape[0]:
        raise ValueError(
            f"Requested {data_num} samples, "
            f"but dataset contains {channel.shape[0]} samples."
        )

    channel = channel[:data_num]
    sensing = sensing[:data_num]

    # ------------------------------------------------------------------------
    # Convert K=3 complex sensing targets into 6 real-valued channels:
    #
    # [Re1, Im1, Re2, Im2, Re3, Im3]
    # ------------------------------------------------------------------------

    sensing_real = np.real(sensing)
    sensing_imag = np.imag(sensing)

    sensing_ri = np.stack(
        [sensing_real, sensing_imag],
        axis=-1,
    ).reshape(
        data_num,
        N,
        SENSING_CHANNELS,
    )

    sensing_input = sensing_ri.reshape(
        data_num,
        nx,
        ny,
        SENSING_CHANNELS,
    ).astype(np.float32)

    # ------------------------------------------------------------------------
    # Generate noisy communication observations
    # ------------------------------------------------------------------------

    snr_linear = 10 ** (snr_db / 10.0)

    noisy_input = np.zeros(
        (data_num, nx, ny, 2),
        dtype=np.float32,
    )

    true_output = np.zeros(
        (data_num, nx, ny, 2),
        dtype=np.float32,
    )

    for i in range(data_num):
        H = channel[i].reshape(nx, ny)

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
    print(f"Dataset          : {filepath}")
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
# Pruning Configuration
# ============================================================================

PRUNING_PARAMS = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=INITIAL_SPARSITY,
        final_sparsity=FINAL_SPARSITY,
        begin_step=0,
        end_step=END_STEP,
    )
}


def apply_pruning_to_layer(layer):
    """
    Apply magnitude pruning only to Conv2D and Dense layers.
    """

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.Dense,
        ),
    ):
        return tfmot.sparsity.keras.prune_low_magnitude(
            layer,
            **PRUNING_PARAMS,
        )

    return layer


def build_pruned_model(original_model):
    """
    Clone the trained dense model and attach pruning wrappers.
    """

    return tf.keras.models.clone_model(
        original_model,
        clone_function=apply_pruning_to_layer,
    )


# ============================================================================
# Pruning Fine-Tuning
# ============================================================================

def train_pruned_model(
    model,
    train_data,
    save_path,
):
    """
    Fine-tune the pruned model once using 50,000 training samples.
    """

    noisy_input, true_output, sensing_input = train_data

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PRUNING_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.compile(
        optimizer=Adam(
            learning_rate=PRUNING_LEARNING_RATE
        ),
        loss="mse",
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),

        tfmot.sparsity.keras.PruningSummaries(
            log_dir=str(PRUNING_LOG_DIR)
        ),

        ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
    ]

    model.fit(
        x=[
            noisy_input,
            sensing_input,
        ],
        y=true_output,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    return model


# ============================================================================
# Export Final Pruned Model
# ============================================================================

def export_pruned_model(
    model,
    save_path,
):
    """
    Remove pruning wrappers and export the final sparse model.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stripped_model = tfmot.sparsity.keras.strip_pruning(
        model
    )

    print("\n================ Final Pruned Model ================")
    stripped_model.summary()
    print("====================================================")

    stripped_model.save(
        str(save_path),
        include_optimizer=False,
    )

    return stripped_model


# ============================================================================
# Sparsity Statistics
# ============================================================================

def report_model_sparsity(model):
    """
    Report non-zero parameters and sparsity of each weighted layer.
    """

    total_weight_params = 0
    total_nonzero_params = 0

    print("\n================ Sparsity Statistics =================")

    for layer in model.layers:

        if not layer.weights:
            continue

        layer_total = 0
        layer_nonzero = 0

        for weight in layer.weights:

            weight_array = weight.numpy()

            layer_total += weight_array.size
            layer_nonzero += np.count_nonzero(
                weight_array
            )

        layer_sparsity = 100.0 * (
            1.0
            - layer_nonzero
            / max(layer_total, 1)
        )

        total_weight_params += layer_total
        total_nonzero_params += layer_nonzero

        print(
            f"{layer.name:35s} "
            f"non-zero={layer_nonzero:10d}  "
            f"total={layer_total:10d}  "
            f"sparsity={layer_sparsity:6.2f}%"
        )

    overall_sparsity = 100.0 * (
        1.0
        - total_nonzero_params
        / max(total_weight_params, 1)
    )

    print("------------------------------------------------------")
    print(
        f"Total weighted parameters : {total_weight_params}"
    )
    print(
        f"Total non-zero parameters : {total_nonzero_params}"
    )
    print(
        f"Overall sparsity          : {overall_sparsity:.2f}%"
    )
    print("======================================================\n")


# ============================================================================
# Independent Test Evaluation
# ============================================================================

def evaluate_model(
    model,
    test_data,
):
    """
    Evaluate the final pruned model on 2,000 independent test samples.
    """

    noisy_input, true_output, sensing_input = test_data

    estimated_channel = model.predict(
        [
            noisy_input,
            sensing_input,
        ],
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    signal_power = np.mean(
        true_output ** 2
    )

    input_mse = np.mean(
        (true_output - noisy_input) ** 2
    )

    output_mse = np.mean(
        (true_output - estimated_channel) ** 2
    )

    input_nmse = input_mse / signal_power
    output_nmse = output_mse / signal_power

    print("\n================ Test Results =================")
    print(f"Test SNR          : {TEST_SNR_DB} dB")
    print(f"Test samples      : {TEST_SAMPLES}")
    print(f"Input NMSE        : {input_nmse:.6e}")
    print(f"Pruned model NMSE : {output_nmse:.6e}")
    print("===============================================\n")

    return input_nmse, output_nmse


# ============================================================================
# Main
# ============================================================================

def main():

    print("================ ISAC Pruning Experiment =================")
    print(f"L = {L}, K = {K}, X = {X}")
    print(f"Training SNR           : {TRAIN_SNR_DB} dB")
    print(f"Training samples       : {TRAIN_SAMPLES}")
    print(f"Test SNR               : {TEST_SNR_DB} dB")
    print(f"Test samples           : {TEST_SAMPLES}")
    print(f"Target sparsity        : {FINAL_SPARSITY:.0%}")
    print(f"Pruning epochs         : {EPOCHS}")
    print(f"Steps per epoch        : {STEPS_PER_EPOCH}")
    print(f"Pruning end step       : {END_STEP}")
    print("===========================================================")

    # ------------------------------------------------------------------------
    # 1. Load the already-trained dense model.
    # ------------------------------------------------------------------------

    if not ORIGINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Original model not found: {ORIGINAL_MODEL_PATH}"
        )

    original_model = tf.keras.models.load_model(
        ORIGINAL_MODEL_PATH,
        compile=False,
    )

    # ------------------------------------------------------------------------
    # 2. Add pruning wrappers.
    # ------------------------------------------------------------------------

    pruned_model = build_pruned_model(
        original_model
    )

    # ------------------------------------------------------------------------
    # 3. Load 50,000 training samples.
    # ------------------------------------------------------------------------

    train_data = load_dataset(
        filepath=TRAIN_DATASET_PATH,
        data_num=TRAIN_SAMPLES,
        nx=NX,
        ny=NY,
        snr_db=TRAIN_SNR_DB,
    )

    # ------------------------------------------------------------------------
    # 4. Fine-tune the pruned model once.
    # ------------------------------------------------------------------------

    pruned_model = train_pruned_model(
        model=pruned_model,
        train_data=train_data,
        save_path=PRUNED_CHECKPOINT_PATH,
    )

    # ------------------------------------------------------------------------
    # 5. Export the final sparse model.
    # ------------------------------------------------------------------------

    final_model = export_pruned_model(
        model=pruned_model,
        save_path=PRUNED_FINAL_PATH,
    )

    # ------------------------------------------------------------------------
    # 6. Report sparsity.
    # ------------------------------------------------------------------------

    report_model_sparsity(
        final_model
    )

    # Release training data before testing.
    del train_data

    # ------------------------------------------------------------------------
    # 7. Load 2,000 independent test samples.
    # ------------------------------------------------------------------------

    test_data = load_dataset(
        filepath=TEST_DATASET_PATH,
        data_num=TEST_SAMPLES,
        nx=NX,
        ny=NY,
        snr_db=TEST_SNR_DB,
    )

    # ------------------------------------------------------------------------
    # 8. Evaluate on the independent test dataset.
    # ------------------------------------------------------------------------

    evaluate_model(
        model=final_model,
        test_data=test_data,
    )

    print("Pruning completed.")
    print(f"Final model: {PRUNED_FINAL_PATH}")


if __name__ == "__main__":
    main()
