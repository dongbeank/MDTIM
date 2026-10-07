# Discretizing Continuous Time Series for Imputation with Masked Diffusion Training

Official PyTorch implementation of **MDTIM** (Masked Diffusion Time-series Imputation Model), accepted to **NeurIPS 2026**.

[**Project Page**](https://dongbeank.github.io/MDTIM) | [**Paper (OpenReview)**](https://openreview.net/forum?id=pKTszDf7fn) <!-- TODO: add arXiv link -->

MDTIM adapts the masked (absorbing-state) discrete diffusion framework to continuous time series imputation. Missing entries are represented by a `[MASK]` token that is structurally orthogonal to all observed values, and the model directly predicts the original signal. **Stochastic Discretization** maps continuous values to ordinal-aware tokens while preserving continuous dynamics.

## Installation

```bash
conda create -n mdtim python=3.11 -y
conda activate mdtim
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Data

CSV benchmarks (ETTh, Energy, Weather) ship in `data/`. The synthetic Sine dataset is generated automatically on first use (see `Utils/generate_dataset.py`).

## Training

```bash
# One dataset
python main.py --data etth --mode train

# Or use the provided scripts
bash scripts/train_etth.sh
bash scripts/train_all.sh
```

Checkpoints are versioned under `checkpoints/<dataset>_<window>/v<N>/best.ckpt`.

Useful flags (see `config.py` for the full list):

| Flag | Description |
|---|---|
| `--data {etth,energy,weather,sine,...}` | Dataset preset |
| `--seed <int>` | Training seed |
| `--fft_weight <float>` | Spectral (FFT) loss weight (0 disables) |
| `--soft_label_window <int>` | Ordinal soft-label window (0 = one-hot) |
| `--shared_channel_embedding` | Share one token embedding across all channels (scalability study) |

## Evaluation (imputation)

```bash
# All datasets, 3 mask seeds, uniform+geometric masks at 30/50/70% missing
python imputation_all.py

# Specific settings
python imputation_all.py --data etth --missing_ratio 0.3 0.5 0.7 --mask_type uniform

# Also report CRPS (probabilistic calibration)
python imputation_all.py --data etth energy sine --crps
```

Results are written to `imputation_results/all_results.csv` (per-seed) and `stats_all.csv` (mean ± std over mask seeds). MAE/MSE/RMSE are computed on missing positions only; CSV benchmarks are standardized with train-split statistics.

## Citation

```bibtex
@inproceedings{kim2026discretizing,
  title={Discretizing Continuous Time Series for Imputation with Masked Diffusion Training},
  author={Kim, Dongbin and Lee, Seungyun and Shin, Geonwoo and Lee, Jaewook},
  booktitle={Advances in Neural Information Processing Systems},
  volume={39},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
