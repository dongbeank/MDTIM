import math
import typing
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy import io

import utils
from Utils.generate_dataset import generate_sine_data

LOGGER = utils.get_logger(__name__)


class MDCustomDataset(Dataset):
    """
    Time Series Dataset for MDTM.

    Returns raw values only. Normalization is deferred to diffusion.py
    to avoid cheating (using masked values in min/max calculation).
    """
    def __init__(
        self,
        name,
        data_root,
        window=64,
        seed=123,
        period='train',
        n_bins=40,
        min_range=1.0,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    ):
        """
        Args:
            name: Dataset name
            data_root: Path to data file/directory
            window: Sequence length
            seed: Random seed
            period: 'train', 'val', or 'test'
            n_bins: Number of bins for tokenization
            min_range: Minimum range for normalization (handles constant values)
            train_ratio, val_ratio, test_ratio: Split ratios (default 8:1:1)
        """
        super().__init__()
        self.name = name
        self.window = window
        self.n_bins = n_bins
        self.min_range = min_range

        # Datasets that come pre-formatted as (N, T, C)
        self.is_presplit = name == 'sine'

        # 1. Load data
        self.rawdata, self.scaler = self._read_data(data_root, name, window=window, seed=seed)

        # 2. Get raw windows based on dataset type and period
        raw_windows, self.missing_mask = self._prepare_windows(period, train_ratio, val_ratio, test_ratio)

        self.sample_num = raw_windows.shape[0]
        self.n_channels = raw_windows.shape[2]

        # 3. Store raw windows (normalization deferred to diffusion.py)
        self.raw_windows = raw_windows.astype(np.float32)

        # Bin info
        bins = np.linspace(-1, 1, n_bins)
        self.bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0

    def _prepare_windows(self, period, train_ratio, val_ratio, test_ratio):
        """Prepare raw windows based on dataset type and split.

        Returns:
            (raw_windows, missing_mask): missing_mask is boolean array where True = missing (NaN)
                                         None for complete datasets
        """
        if self.is_presplit:
            # Synthetic data (sine, mujoco): already (N, T, C)
            raw_windows = self.rawdata
            total = raw_windows.shape[0]
        else:
            # CSV/mat data: create sliding windows
            sample_num_total = max(self.rawdata.shape[0] - self.window + 1, 0)
            raw_windows = np.zeros(
                (sample_num_total, self.window, self.rawdata.shape[-1]),
                dtype=np.float32
            )
            for i in range(sample_num_total):
                raw_windows[i] = self.rawdata[i:i + self.window]
            total = raw_windows.shape[0]

        # 8:1:1 split
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        if period == 'train':
            return raw_windows[:n_train], None
        elif period == 'val':
            return raw_windows[n_train:n_train + n_val], None
        else:
            return raw_windows[n_train + n_val:], None

    @staticmethod
    def _read_data(filepath, name='', window=24, seed=123):
        """Read data from file or generate synthetic data."""
        if name == 'fmri':
            data = io.loadmat(os.path.join(filepath, 'sim4.mat'))['ts']
            scaler = MinMaxScaler().fit(data)
            return data.astype(np.float32), scaler

        elif name == 'sine':
            samples = generate_sine_data(10000, window, dim=5, seed=seed)
            scaler = MinMaxScaler().fit(samples.reshape(-1, 5))
            return samples, scaler

        else:
            # CSV files (etth, energy, weather, etc.)
            df = pd.read_csv(filepath, header=0)
            if name == 'etth':
                df.drop(df.columns[0], axis=1, inplace=True)
            elif name == 'weather':
                df.drop(df.columns[0], axis=1, inplace=True)
                df = df.replace(-9999, np.nan)
                df = df.ffill().bfill()
            data = df.values
            scaler = MinMaxScaler().fit(data)
            return data.astype(np.float32), scaler

    def __getitem__(self, ind):
        result = {
            'raw_values': torch.from_numpy(self.raw_windows[ind]),
            'attention_mask': torch.ones(self.window, dtype=torch.float),
            'n_channels': self.n_channels,
            'bin_width': self.bin_width
        }

        # Add missing mask if available (True = missing/NaN position)
        if self.missing_mask is not None:
            result['missing_mask'] = torch.from_numpy(self.missing_mask[ind])

        return result

    def __len__(self):
        return self.sample_num

    def denormalize_window(self, normalized_data, window_min, window_max):
        """Per-window denormalization: [-1, 1] -> original scale"""
        data_01 = (normalized_data + 1) / 2
        return data_01 * (window_max - window_min) + window_min


class DummyTokenizer:
    """Dummy tokenizer for time series"""
    def __init__(self, vocab_size=41, n_bins=40):
        self.vocab_size = vocab_size
        self.mask_token_id = 0
        self.mask_token = '[MASK]'
        self.pad_token_id = 0
        self.pad_token = '[MASK]'
        self.bos_token_id = None
        self.eos_token_id = None
        self.n_bins = n_bins
        self.offset = 1

    def decode(self, tokens):
        return str(tokens.tolist() if hasattr(tokens, 'tolist') else tokens)

    def batch_decode(self, batch):
        return [self.decode(t) for t in batch]


def get_tokenizer(config):
    """Return dummy tokenizer for time series"""
    n_bins = getattr(config.model, 'n_bins', 40)
    vocab_size = n_bins + 1
    return DummyTokenizer(vocab_size=vocab_size, n_bins=n_bins)


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
    n_bins = getattr(config.model, 'n_bins', 40)
    min_range = getattr(config.data, 'min_range', 1.0)

    common_kwargs = {
        'name': config.data.dataset_name,
        'data_root': config.data.train_path,
        'window': config.model.length,
        'n_bins': n_bins,
        'min_range': min_range,
    }

    if skip_train:
        train_set = None
    else:
        train_set = MDCustomDataset(**common_kwargs, period='train')

    if skip_valid:
        valid_set = None
    else:
        valid_set = MDCustomDataset(**common_kwargs, period='val')

    if skip_train:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=True,
            persistent_workers=True)
        train_loader.tokenizer = tokenizer

    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator)
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop('shuffle', None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'random_state': self.generator.get_state(),
                'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get('random_state'))
        self.counter = state_dict['counter']
        self.restarting = True

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)
        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(
                    padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
