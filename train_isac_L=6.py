"""
XL-MIMO Near-Field ISAC Channel Estimation Network

Scenario:
    L = 6  : number of communication multipath components
    K = 6  : number of sensing/radar targets
    X = 6  : number of shared communication-sensing scatterers

Dataset:
    Pnear_overlap6_AOAerr_1e-4_RangeErr_1e-2_Ln6_N256_2000.mat

The .mat file contains:
    Channel_mat
    Channel_near_Radar_overlap6_mat

This script performs ONE training run.
"""
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Permute,
    Reshape,
    Subtract,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


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

L = 6
K = 6
X = 6

assert X <= min(L, K), "X must satisfy X <= min(L, K)."

SENSING_CHANNELS = 2 * K
OUTPUT_CHANNELS = 2


# ============================================================================
# Experiment Configuration
# ============================================================================

TRAIN_SNR_DB = -5
TEST_SNR_DB = -5

TRAIN_SAMPLES = 50000
TEST_SAMPLES = 2000

BATCH_SIZE = 128
EPOCHS = 200
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 1e-3


# ============================================================================
# Dataset Paths
# ============================================================================

# Training dataset:
# num_sta = 5000, num_ffading = 10 -> 50,000 samples
TRAIN_DATASET_PATH = Path(
    "data/Pnear_overlap6_AOAerr_1e-4_RangeErr_1e-2_Ln6_N256_50000.mat"
)

# Independent test dataset:
# num_sta = 200, num_ffading = 10 -> 2,000 samples
TEST_DATASET_PATH = Path(
    "data/Pnear_overlap6_AOAerr_1e-4_RangeErr_1e-2_Ln6_N256_2000.mat"
)

SENSING_KEY = "Channel_near_Radar_overlap6_mat"

MODEL_SAVE_PATH = Path(
    "checkpoints/isac_L6_K6_X6_-5dB.hdf5"
)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(filepath, data_num, nx, ny, snr_db):
    """Load one ISAC dataset and add AWGN to the communication channel."""

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    data = sio.loadmat(filepath)

    if "Channel_mat" not in data:
        raise KeyError("Variable 'Channel_mat' was not found in the dataset.")

    if SENSING_KEY not in data:
        raise KeyError(
            f"Variable '{SENSING_KEY}' was not found in the dataset."
        )

    channel = data["Channel_mat"]
    sensing = data[SENSING_KEY]

    if data_num > channel.shape[0]:
        raise ValueError(
            f"Requested {data_num} samples, but dataset contains "
            f"{channel.shape[0]} samples."
        )

    channel = channel[:data_num]
    sensing = sensing[:data_num]

    # K complex sensing targets -> 2K real-valued channels
    sensing_real = np.real(sensing)
    sensing_imag = np.imag(sensing)

    sensing_ri = np.stack(
        [sensing_real, sensing_imag],
        axis=-1,
    ).reshape(data_num, N, 2 * K)

    sensing_input = sensing_ri.reshape(
        data_num,
        nx,
        ny,
        2 * K,
    ).astype(np.float32)

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
    print(f"L/K/X            : {L}/{K}/{X}")
    print(f"SNR              : {snr_db} dB")
    print(f"Samples          : {data_num}")
    print(f"Channel input    : {noisy_input.shape}")
    print(f"Sensing input    : {sensing_input.shape}")
    print(f"Channel target   : {true_output.shape}")
    print("=====================================================\n")

    return noisy_input, true_output, sensing_input


# ============================================================================
# Fusion Attention Block
# ============================================================================

def fusion_attention_block(
    inputs,
    query_data,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.1,
):
    # Cross-attention
    cross_attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=int(inputs.shape[-1]),
    )(
        query=query_data,
        key=inputs,
        value=inputs,
    )

    cross_attention = Dropout(dropout_rate)(cross_attention)
    x = Add()([inputs, cross_attention])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Channel-domain self-attention
    channel_attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=int(x.shape[-1]),
    )(
        query=x,
        key=x,
        value=x,
    )

    channel_attention = Dropout(dropout_rate)(channel_attention)
    x = Add()([x, channel_attention])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Spatial-domain self-attention
    spatial_input = Permute((2, 1))(x)

    spatial_attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=int(spatial_input.shape[-1]),
    )(
        query=spatial_input,
        key=spatial_input,
        value=spatial_input,
    )

    spatial_attention = Permute((2, 1))(spatial_attention)
    spatial_attention = Dropout(dropout_rate)(spatial_attention)

    x = Add()([x, spatial_attention])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward network
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(int(x.shape[-1]))(ffn)
    ffn = Dropout(dropout_rate)(ffn)

    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


# ============================================================================
# Network
# ============================================================================

def build_model():
    noisy_input = Input(
        shape=(NX, NY, 2),
        name="noisy_communication_channel",
    )

    sensing_input = Input(
        shape=(NX, NY, SENSING_CHANNELS),
        name="sensing_information",
    )

    x_comm = Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
    )(noisy_input)
    x_comm = BatchNormalization()(x_comm)
    x_comm = Dropout(0.1)(x_comm)

    x_sensing = Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
    )(sensing_input)
    x_sensing = BatchNormalization()(x_sensing)
    x_sensing = Dropout(0.1)(x_sensing)

    x_comm = Reshape((NX * NY, 64))(x_comm)
    x_sensing = Reshape((NX * NY, 64))(x_sensing)

    x = fusion_attention_block(
        x_comm,
        x_sensing,
        num_heads=4,
        ff_dim=128,
    )

    x = fusion_attention_block(
        x,
        x_sensing,
        num_heads=4,
        ff_dim=128,
    )

    x = Reshape((NX, NY, 64))(x)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    residual = Conv2D(
        OUTPUT_CHANNELS,
        kernel_size=(3, 3),
        padding="same",
        activation="linear",
        name="estimated_residual",
    )(x)

    output = Subtract(name="estimated_channel")(
        [noisy_input, residual]
    )

    return Model(
        inputs=[noisy_input, sensing_input],
        outputs=output,
        name="NearField_ISAC_Channel_Estimator",
    )


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_data):
    noisy_input, true_output, sensing_input = train_data

    MODEL_SAVE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        x=[noisy_input, sensing_input],
        y=true_output,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )


# ============================================================================
# Testing
# ============================================================================

def test_model(model_path, test_data):
    noisy_input, true_output, sensing_input = test_data

    model = load_model(
        model_path,
        compile=False,
    )

    estimated_channel = model.predict(
        [noisy_input, sensing_input],
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    signal_power = np.mean(true_output ** 2)

    input_nmse = np.mean(
        (true_output - noisy_input) ** 2
    ) / signal_power

    output_nmse = np.mean(
        (true_output - estimated_channel) ** 2
    ) / signal_power

    print("\n================ Test Results =================")
    print(f"Test SNR          : {TEST_SNR_DB} dB")
    print(f"Test samples      : {TEST_SAMPLES}")
    print(f"Input NMSE        : {input_nmse:.6e}")
    print(f"Estimated NMSE    : {output_nmse:.6e}")
    print("===============================================\n")

    return input_nmse, output_nmse


# ============================================================================
# Main
# ============================================================================

def main():

    print("================ ISAC Experiment =================")
    print(f"L = {L}, K = {K}, X = {X}")
    print(f"Training SNR     : {TRAIN_SNR_DB} dB")
    print(f"Training samples : {TRAIN_SAMPLES}")
    print(f"Test SNR         : {TEST_SNR_DB} dB")
    print(f"Test samples     : {TEST_SAMPLES}")
    print("==================================================")

    # 1. Load 50,000 training samples
    train_data = load_dataset(
        filepath=TRAIN_DATASET_PATH,
        data_num=TRAIN_SAMPLES,
        nx=NX,
        ny=NY,
        snr_db=TRAIN_SNR_DB,
    )

    # 2. Build and train one model at this SNR
    model = build_model()
    model.summary()
    train_model(model, train_data)

    # 3. Clear training data/model from memory
    del model
    del train_data
    tf.keras.backend.clear_session()

    # 4. Load 2,000 independent test samples
    test_data = load_dataset(
        filepath=TEST_DATASET_PATH,
        data_num=TEST_SAMPLES,
        nx=NX,
        ny=NY,
        snr_db=TEST_SNR_DB,
    )

    # 5. Evaluate the best saved model
    test_model(
        model_path=MODEL_SAVE_PATH,
        test_data=test_data,
    )


if __name__ == "__main__":
    main()
