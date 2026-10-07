"""
MDTM Imputation - Low Variance Scoring.

1-step imputation with ensemble variance-based confidence scoring.

Usage:
    python imputation_all.py
    python imputation_all.py --data etth energy
    python imputation_all.py --data etth --missing_ratio 0.5
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from itertools import product
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diffusion
import dataloader as dl

MDTM_PATH = os.path.dirname(os.path.abspath(__file__))
WINDOW = 48

# Default configurations
DEFAULT_SEEDS = [2024, 2025, 2026]
DEFAULT_MISSING_RATIOS = [0.3, 0.5, 0.7]
DEFAULT_MASK_TYPES = ['uniform', 'geometric']

DATASET_CONFIG = {
    "etth": {
        "data_path": f"{MDTM_PATH}/data/ETTh.csv",
        "n_channels": 7,
        "version": 4,
        "standardize": True,
    },
    "energy": {
        "data_path": f"{MDTM_PATH}/data/energy_data.csv",
        "n_channels": 28,
        "version": 2,
        "standardize": True,
    },
    "weather": {
        "data_path": f"{MDTM_PATH}/data/weather.csv",
        "n_channels": 21,
        "version": 2,
        "standardize": True,
    },
    "sine": {
        "data_path": f"{MDTM_PATH}/data/sine_full_48_5_10000_2024.npy",
        "n_channels": 5,
        "version": 2,
        "standardize": False,
    },
}


def get_checkpoint(base_dir, version=None):
    if not os.path.exists(base_dir):
        return None
    if version is not None:
        ckpt = os.path.join(base_dir, f'v{version}', 'best.ckpt')
        return ckpt if os.path.exists(ckpt) else None
    versions = sorted([d for d in os.listdir(base_dir) if d.startswith('v') and d[1:].isdigit()],
                     key=lambda x: int(x[1:]), reverse=True)
    for v in versions:
        ckpt = os.path.join(base_dir, v, 'best.ckpt')
        if os.path.exists(ckpt):
            return ckpt
    return None


def uniform_mask(data, missing_ratio, seed):
    """Generate uniform random mask. True = observed, False = missing."""
    np.random.seed(seed)
    mask = np.ones(data.shape, dtype=bool)
    total_elements = data.size
    n_missing = int(total_elements * missing_ratio)
    flat_indices = np.random.choice(total_elements, n_missing, replace=False)
    flat_mask = mask.flatten()
    flat_mask[flat_indices] = False
    return flat_mask.reshape(data.shape)


def geom_noise_mask_single(L, lm, masking_ratio):
    """Geometric distribution mask for single sequence."""
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]
    state = int(np.random.rand() > masking_ratio)
    for i in range(L):
        keep_mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask


def geometric_mask(data, missing_ratio, seed, lm=3):
    """Geometric distribution mask. True = observed, False = missing."""
    np.random.seed(seed)
    N, T, C = data.shape
    mask = np.ones(data.shape, dtype=bool)
    for n in range(N):
        for c in range(C):
            mask[n, :, c] = geom_noise_mask_single(T, lm, missing_ratio)
    return mask


def load_model(checkpoint_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['hyper_parameters']['config']
    tokenizer = dl.get_tokenizer(config)
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path, tokenizer=tokenizer, config=config, map_location=device
    )
    model = model.to(device)
    model.eval()
    if model.ema:
        import itertools
        model.ema.copy_to(itertools.chain(
            model.backbone.parameters(),
            model.noise.parameters()))
    return model, device


def compute_crps(probs, gt_normalized, bin_centers, missing_mask):
    """
    Compute CRPS (Continuous Ranked Probability Score) for missing positions.

    Args:
        probs: (B, T, C, n_bins) - probability distribution over bins
        gt_normalized: (B, T, C) - ground truth in normalized space [-output_range, output_range]
        bin_centers: (n_bins,) - center value of each bin
        missing_mask: (B, T, C) - True = missing position

    Returns:
        crps: scalar - mean CRPS over missing positions
    """
    # CDF of the predicted distribution vs. Heaviside step at the ground truth
    cdf = probs.cumsum(dim=-1)  # (B, T, C, n_bins)
    gt_expanded = gt_normalized.unsqueeze(-1)  # (B, T, C, 1)
    bins_expanded = bin_centers.view(1, 1, 1, -1)  # (1, 1, 1, n_bins)
    heaviside = (bins_expanded >= gt_expanded).float()  # (B, T, C, n_bins)

    bin_width = bin_centers[1] - bin_centers[0]
    crps_per_pos = ((cdf - heaviside) ** 2).sum(dim=-1) * bin_width  # (B, T, C)

    crps_missing = crps_per_pos[missing_mask]
    if crps_missing.numel() == 0:
        return 0.0
    return crps_missing.mean().item()


def run_imputation(test_raw, mask, model, device, n_ensemble=10,
                   noise_scale=0.5, decode='weighted', return_crps=False):
    """
    Run MDTM imputation with low variance scoring (1-step).

    Uses ensemble predictions and fills all missing positions at once,
    prioritizing positions with lower prediction variance.

    Args:
        noise_scale: Scale of noise added to input tokens (default: 0.5 -> [-0.5, 0.5])
        decode: 'weighted' (expectation) or 'topprob' (argmax bin center)
        return_crps: also compute CRPS from the ensemble-averaged bin distribution

    Returns:
        pred: (N, T, C) imputed values in raw space
        crps: mean CRPS over missing positions (None if return_crps=False)
    """
    n_bins = model.n_bins
    output_range = model.output_range
    n_bins_output = model.n_bins_output
    mask_index = model.mask_index

    bin_edges = torch.linspace(-output_range, output_range, n_bins_output + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    n_samples = test_raw.shape[0]
    batch_size = 64
    all_results = []
    all_crps = []
    total_missing = 0

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_raw = test_raw[start_idx:end_idx]
        batch_mask = mask[start_idx:end_idx]
        B, T, C = batch_raw.shape

        result_raw = torch.from_numpy(batch_raw.copy()).float().to(device)
        original_mask = torch.from_numpy(batch_mask).bool().to(device)

        n_missing_per_sample = (~original_mask).sum(dim=(1, 2))
        max_missing = n_missing_per_sample.max().item()

        if max_missing == 0:
            all_results.append(result_raw.cpu().numpy())
            continue

        # === Normalize === (handle channels with no observed values)
        filled_for_min = torch.where(original_mask, result_raw, torch.tensor(float('inf'), device=device))
        filled_for_max = torch.where(original_mask, result_raw, torch.tensor(float('-inf'), device=device))
        current_mins = filled_for_min.min(dim=1, keepdim=True).values  # (B, 1, C)
        current_maxs = filled_for_max.max(dim=1, keepdim=True).values  # (B, 1, C)

        # Handle channels with no observed values (min=inf, max=-inf)
        no_obs_channel = torch.isinf(current_mins) & torch.isinf(current_maxs)
        sample_min = filled_for_min.reshape(B, -1).min(dim=1, keepdim=True).values.unsqueeze(-1)
        sample_max = filled_for_max.reshape(B, -1).max(dim=1, keepdim=True).values.unsqueeze(-1)
        sample_min = torch.where(torch.isinf(sample_min), torch.zeros_like(sample_min), sample_min)
        sample_max = torch.where(torch.isinf(sample_max), torch.ones_like(sample_max), sample_max)
        current_mins = torch.where(no_obs_channel, sample_min.expand_as(current_mins), current_mins)
        current_maxs = torch.where(no_obs_channel, sample_max.expand_as(current_maxs), current_maxs)

        data_range = current_maxs - current_mins
        needs_expand = data_range < 1.0
        c_mean = (current_mins + current_maxs) / 2
        current_mins = torch.where(needs_expand, c_mean - 0.5, current_mins)
        current_maxs = torch.where(needs_expand, c_mean + 0.5, current_maxs)

        normalized = (result_raw - current_mins) / (current_maxs - current_mins) * 2 - 1
        normalized = normalized.clamp(-1, 1)

        x_input = ((normalized + 1) / 2 * (n_bins - 1) + 1).round().long()
        x_input = torch.where(original_mask, x_input, torch.zeros_like(x_input))

        # === Ensemble inference for low variance scoring ===
        all_pred_normalized = []
        probs_sum = None

        for e in range(n_ensemble):
            noise = (torch.rand_like(x_input.float()) - 0.5) * 2 * noise_scale
            noised_input = (x_input.float() + noise * original_mask.float()).clamp(1, n_bins).round().long()
            noised_input = torch.where(original_mask, noised_input, torch.zeros_like(noised_input))

            sigma = torch.zeros(B, device=device)
            with torch.no_grad():
                logits = model.forward(noised_input, sigma)
            logits[..., mask_index] = float('-inf')
            probs = torch.softmax(logits, dim=-1)

            valid_probs = probs[..., 1:n_bins_output+1].float()
            if decode == 'topprob':
                pred_norm = bin_centers[valid_probs.argmax(dim=-1)]
            else:  # weighted
                pred_norm = (valid_probs * bin_centers.view(1, 1, 1, -1)).sum(dim=-1)
            all_pred_normalized.append(pred_norm)
            if return_crps:
                probs_sum = valid_probs if probs_sum is None else probs_sum + valid_probs

        # Stack: (n_ensemble, B, T, C)
        all_pred_normalized = torch.stack(all_pred_normalized, dim=0)

        # Mean prediction
        mean_pred_normalized = all_pred_normalized.mean(dim=0)
        pred_raw = (mean_pred_normalized + 1) / 2 * (current_maxs - current_mins) + current_mins

        # Keep original observations, fill missing with predictions
        original_raw = torch.from_numpy(batch_raw).float().to(device)
        final_raw = torch.where(original_mask, original_raw, pred_raw)
        all_results.append(final_raw.cpu().numpy())

        if return_crps:
            missing_mask = ~original_mask
            n_missing = missing_mask.sum().item()
            total_missing += n_missing
            avg_probs = probs_sum / n_ensemble
            gt_normalized = (original_raw - current_mins) / (current_maxs - current_mins) * 2 - 1
            gt_normalized = gt_normalized.clamp(-output_range, output_range)
            batch_crps = compute_crps(avg_probs, gt_normalized, bin_centers, missing_mask)
            all_crps.append(batch_crps * n_missing)  # weighted by count

    pred = np.concatenate(all_results, axis=0)
    if return_crps:
        crps = sum(all_crps) / total_missing if total_missing > 0 else 0.0
        return pred, crps
    return pred, None


def compute_metrics(gt, pred, mask, train_mean=None, train_std=None):
    """Compute MAE and MSE on missing positions."""
    missing_mask = ~mask

    if train_mean is not None and train_std is not None:
        gt_standardized = (gt - train_mean) / (train_std + 1e-8)
        pred_standardized = (pred - train_mean) / (train_std + 1e-8)
        gt_missing = gt_standardized[missing_mask]
        pred_missing = pred_standardized[missing_mask]
    else:
        gt_missing = gt[missing_mask]
        pred_missing = pred[missing_mask]

    mae = np.mean(np.abs(gt_missing - pred_missing))
    mse = np.mean((gt_missing - pred_missing) ** 2)
    return mae, mse


def load_data_with_stats(dataset):
    """Load data and compute train statistics for standardization."""
    config = DATASET_CONFIG[dataset]

    if dataset == 'sine':
        data_path = config['data_path']
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        all_data = np.load(data_path)
        n = len(all_data)
        val_end = int(n * 0.9)
        test_raw = all_data[val_end:]
        return test_raw, None, None

    else:
        data_path = config['data_path']
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)

        if dataset in ['etth', 'weather']:
            df.drop(df.columns[0], axis=1, inplace=True)
        if dataset == 'weather':
            df = df.replace(-9999, np.nan)
            df = df.ffill().bfill()

        data = df.values.astype(np.float32)

        n = len(data)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_data = data[:train_end]
        test_data = data[val_end:]

        train_mean = train_data.mean(axis=0)
        train_std = train_data.std(axis=0)

        n_windows = len(test_data) - WINDOW + 1
        test_raw = np.array([test_data[i:i+WINDOW] for i in range(n_windows)])

        return test_raw, train_mean, train_std


def main():
    parser = argparse.ArgumentParser(description='MDTM Imputation')
    parser.add_argument('--data', type=str, nargs='+',
                        default=['etth', 'energy', 'sine', 'weather'],
                        choices=['etth', 'energy', 'weather', 'sine'],
                        help='Datasets to evaluate')
    parser.add_argument('--missing_ratio', type=float, nargs='+',
                        default=DEFAULT_MISSING_RATIOS,
                        help='Missing ratios (default: 0.3 0.5 0.7)')
    parser.add_argument('--mask_type', type=str, nargs='+',
                        default=DEFAULT_MASK_TYPES,
                        choices=['uniform', 'geometric'],
                        help='Mask types (default: uniform geometric)')
    parser.add_argument('--seed', type=int, nargs='+',
                        default=DEFAULT_SEEDS,
                        help='Random seeds (default: 2024 2025 2026)')
    parser.add_argument('--n_ensemble', type=int, default=10,
                        help='Number of ensemble runs (default: 10)')
    parser.add_argument('--decode', type=str, default='weighted',
                        choices=['weighted', 'topprob'],
                        help='Decoding method: weighted (expectation) or topprob (argmax)')
    parser.add_argument('--noise_scale', type=float, nargs='+', default=[0.5],
                        help='Noise scale for input tokens (default: 0.5)')
    parser.add_argument('--output_dir', type=str,
                        default=f'{MDTM_PATH}/imputation_results',
                        help='Output directory for results')
    parser.add_argument('--version', type=int, default=None,
                        help='Model version to use (overrides dataset default)')
    parser.add_argument('--crps', action='store_true',
                        help='Also compute CRPS from the ensemble-averaged bin distribution')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_configs = (len(args.data) * len(args.mask_type) *
                     len(args.missing_ratio) * len(args.noise_scale) * len(args.seed))
    print(f"Total configurations: {total_configs}")
    print(f"  Datasets: {args.data}")
    print(f"  Decode method: {args.decode}")
    print(f"  Mask types: {args.mask_type}")
    print(f"  Missing ratios: {args.missing_ratio}")
    print(f"  Noise scales: {args.noise_scale}")
    print(f"  Seeds: {args.seed}")
    print(f"  Output: {args.output_dir}")
    print()

    all_results = []

    for dataset in args.data:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*70}")

        config = DATASET_CONFIG[dataset]
        standardize = config['standardize']

        base_dir = f"{MDTM_PATH}/checkpoints/{dataset}_{WINDOW}"
        if args.version is not None:
            ckpt_path = get_checkpoint(base_dir, args.version)
            version = args.version
        else:
            ckpt_path = get_checkpoint(base_dir, None)
            if ckpt_path:
                version = int(ckpt_path.split('/v')[1].split('/')[0])
            else:
                version = config['version']

        if ckpt_path is None:
            print(f"WARNING: No checkpoint found for {dataset} (v{version})")
            continue

        print(f"Checkpoint: {ckpt_path}")
        print("Loading model...")
        model, device = load_model(ckpt_path)
        print("Model loaded!")

        print("Loading test data...")
        try:
            test_raw, train_mean, train_std = load_data_with_stats(dataset)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue

        print(f"Test windows: {test_raw.shape}")

        if standardize and train_std is not None:
            print("Metrics will be computed on standardized space.")
        else:
            print("Metrics will be computed on original space.")

        configs = list(product(args.mask_type, args.missing_ratio,
                               args.noise_scale, args.seed))

        pbar = tqdm(configs, desc=f"{dataset}")
        for mask_type, missing_ratio, noise_scale, seed in pbar:
            pbar.set_postfix({
                'mask': mask_type[:4],
                'mr': f'{missing_ratio:.0%}',
                'noise': noise_scale,
            })

            if mask_type == 'uniform':
                mask = uniform_mask(test_raw, missing_ratio, seed)
            else:
                mask = geometric_mask(test_raw, missing_ratio, seed)

            actual_ratio = (~mask).sum() / mask.size

            pred, crps = run_imputation(
                test_raw.copy(), mask, model, device,
                n_ensemble=args.n_ensemble,
                noise_scale=noise_scale, decode=args.decode,
                return_crps=args.crps
            )

            if standardize:
                mae, mse = compute_metrics(test_raw, pred, mask, train_mean, train_std)
            else:
                mae, mse = compute_metrics(test_raw, pred, mask)

            rmse = np.sqrt(mse)

            result = {
                'dataset': dataset,
                'version': version,
                'standardized': standardize,
                'mask_type': mask_type,
                'missing_ratio': missing_ratio,
                'actual_ratio': actual_ratio,
                'noise_scale': noise_scale,
                'seed': seed,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
            }
            if args.crps:
                result['crps'] = crps
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    individual_path = os.path.join(args.output_dir, 'all_results.csv')
    results_df.to_csv(individual_path, index=False)
    print(f"\nIndividual results saved: {individual_path}")

    if len(args.seed) > 1:
        group_cols = ['dataset', 'mask_type', 'missing_ratio']
        stats_rows = []

        for name, group in results_df.groupby(group_cols):
            row = dict(zip(group_cols, name))
            row['n_seeds'] = len(group)
            row['standardized'] = group['standardized'].iloc[0]

            metrics = ['mae', 'mse', 'rmse'] + (['crps'] if args.crps else [])
            for metric in metrics:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                row[f'{metric}_str'] = f'{mean_val:.4f}+-{std_val:.4f}'

            stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(args.output_dir, 'stats_all.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics saved: {stats_path}")

    print(f"\n{'='*70}")
    print("SUMMARY (MAE)")
    print(f"{'='*70}")

    for dataset in args.data:
        dataset_df = results_df[results_df['dataset'] == dataset]
        if len(dataset_df) == 0:
            continue

        is_std = dataset_df['standardized'].iloc[0]
        std_label = "(standardized)" if is_std else "(original scale)"

        print(f"\n[{dataset.upper()}] {std_label}")
        print(f"{'Mask':<10} {'MR':<6} {'MAE':>20}")
        print("-" * 40)

        for mask_type in args.mask_type:
            for mr in args.missing_ratio:
                subset = dataset_df[
                    (dataset_df['mask_type'] == mask_type) &
                    (dataset_df['missing_ratio'] == mr)
                ]
                if len(subset) > 0:
                    mae_mean = subset['mae'].mean()
                    mae_std = subset['mae'].std()
                    mae_str = f'{mae_mean:.4f}+-{mae_std:.4f}'
                    print(f"{mask_type:<10} {mr:<6.0%} {mae_str:>20}")

    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
