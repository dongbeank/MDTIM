"""
Dataset generation utilities for MDTM.
"""

import os
import numpy as np
from tqdm.auto import tqdm


def generate_sine_data(num_samples, seq_len, dim, seed=123, save_dir='./data'):
    """Generate synthetic sine wave data with save/load capability.

    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length
        dim: Number of channels/features
        seed: Random seed
        save_dir: Directory to save/load data

    Returns:
        data: (num_samples, seq_len, dim) in [0, 1] range
    """
    os.makedirs(save_dir, exist_ok=True)
    npy_path = os.path.join(save_dir, f"sine_full_{seq_len}_{dim}_{num_samples}_{seed}.npy")

    if os.path.exists(npy_path):
        print(f"Loading pre-generated sine data from {npy_path}")
        return np.load(npy_path).astype(np.float32)

    print(f"Generating sine data (num={num_samples}, seq_len={seq_len}, dim={dim})...")
    st0 = np.random.get_state()
    np.random.seed(seed)

    data = []
    for i in tqdm(range(num_samples), desc="Generating sine data"):
        temp = []
        for k in range(dim):
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)
        temp = np.transpose(np.asarray(temp))
        temp = (temp + 1) * 0.5  # Normalize to [0, 1]
        data.append(temp)

    np.random.set_state(st0)
    data = np.array(data, dtype=np.float32)

    np.save(npy_path, data)
    print(f"Saved sine data to {npy_path}")

    return data
