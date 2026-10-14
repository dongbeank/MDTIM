"""
MDTM Training and Evaluation
Simple entry point without Hydra dependency.

Usage:
    # Train with ETTh dataset
    python main.py --data etth --mode train

    # Train with custom settings
    python main.py --data etth --lr 1e-4 --max_steps 20000

    # Evaluate (uses latest version by default)
    python main.py --data etth --mode eval

    # Evaluate specific version
    python main.py --data etth --mode eval --version 1

    # Imputation
    python main.py --data etth --mode impute --missing_ratio 0.3
"""
import os
import glob

import lightning as L
import numpy as np
import torch

import dataloader
import diffusion
import utils
from config import Config, ConfigCompat


def get_version_folder(base_dir: str, version: int = None) -> str:
    """Get the version folder (v1, v2, ...) or base folder if no versions exist.

    Args:
        base_dir: Base checkpoint folder path
        version: Specific version number to load. None = latest version.
    """
    existing_versions = glob.glob(os.path.join(base_dir, 'v*'))
    existing_versions = [v for v in existing_versions if os.path.isdir(v)]

    if existing_versions:
        version_nums = []
        for v in existing_versions:
            try:
                version_nums.append((int(os.path.basename(v)[1:]), v))
            except ValueError:
                continue

        if version_nums:
            if version is not None:
                # Find specific version
                for vnum, vpath in version_nums:
                    if vnum == version:
                        return vpath
                raise ValueError(f"Version {version} not found. Available: {[v[0] for v in version_nums]}")
            else:
                # Return latest version
                version_nums.sort(key=lambda x: x[0], reverse=True)
                return version_nums[0][1]

    return base_dir


def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    """Print sample batch for debugging."""
    for dl_type, dl in [('train', train_ds), ('valid', valid_ds)]:
        if dl is None:
            continue
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch raw_values.shape', batch['raw_values'].shape)

        if batch['raw_values'].ndim == 3:
            print(f'Multivariate data with {batch["raw_values"].shape[2]} channels')
            first = batch['raw_values'][0, :min(k, 10), :min(3, batch["raw_values"].shape[2])]
            print(f'First values (sample 0, first 10 steps, first 3 channels):', first.tolist())
        else:
            first = batch['raw_values'][0, :k]
            print(f'First {k} values:', first.tolist())


def generate_samples(config, tokenizer):
    """Generate time series samples from the model."""
    print('Generating samples.')

    # Find checkpoint path
    base_dir = f"./checkpoints/{config.data.dataset_name}_{config.model.length}"
    version_dir = get_version_folder(base_dir, config._config.version)
    print(f'Using version folder: {version_dir}')

    # Use provided checkpoint_path or find best.ckpt in version folder
    if config.eval.checkpoint_path:
        ckpt_path = config.eval.checkpoint_path
    else:
        ckpt_path = os.path.join(version_dir, 'best.ckpt')

    print(f'Loading checkpoint from: {ckpt_path}')

    model = diffusion.Diffusion.load_from_checkpoint(
        ckpt_path,
        tokenizer=tokenizer,
        config=config
    )

    if config.eval.disable_ema:
        print('Disabling EMA.')
        model.ema = None

    all_samples = []
    for _ in range(config.sampling.num_sample_batches):
        samples = model.restore_model_and_sample(
            num_steps=config.sampling.steps)
        all_samples.append(samples.cpu())

    all_samples = torch.cat(all_samples, dim=0)
    print(f'Generated {all_samples.shape[0]} samples with shape {all_samples.shape}')

    samples_np = all_samples.numpy()

    # Save samples to version folder
    save_path = os.path.join(version_dir, 'generated_samples.npy')
    np.save(save_path, samples_np)
    print(f'Saved samples to {save_path}')

    return all_samples, samples_np


def train(config, tokenizer):
    """Main training loop."""
    print('Starting Training.')
    config._config.print()

    # Setup checkpoint path for resuming
    ckpt_path = None
    if config.checkpointing.resume_from_ckpt and config.checkpointing.resume_ckpt_path:
        if os.path.exists(config.checkpointing.resume_ckpt_path):
            ckpt_path = config.checkpointing.resume_ckpt_path

    # Create dataloaders
    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    # Create model
    model = diffusion.Diffusion(config, tokenizer=valid_ds.tokenizer)

    # Setup callbacks - save to checkpoints/{dataset}_{length}/v{N}/
    base_dir = f"./checkpoints/{config.data.dataset_name}_{config.model.length}"
    os.makedirs(base_dir, exist_ok=True)

    # Determine next version number (v1, v2, ...)
    existing_versions = glob.glob(os.path.join(base_dir, 'v*'))
    existing_versions = [v for v in existing_versions if os.path.isdir(v)]

    if existing_versions:
        version_nums = []
        for v in existing_versions:
            try:
                version_nums.append(int(os.path.basename(v)[1:]))
            except ValueError:
                continue
        next_version = max(version_nums) + 1 if version_nums else 1
    else:
        next_version = 1

    ckpt_dir = os.path.join(base_dir, f'v{next_version}')
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'Saving checkpoints to: {ckpt_dir}')

    # Save config
    config._config.save(os.path.join(ckpt_dir, 'config.json'))

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='last',
            save_top_k=-1,
            save_last=False,
            every_n_train_steps=500,
            auto_insert_metric_name=False,
            verbose=True,
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='best',
            save_top_k=1,
            monitor='val/nll',
            mode='min',
            auto_insert_metric_name=False,
            verbose=True,
        ),
    ]

    # Create trainer
    trainer = L.Trainer(
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes,
        precision=config.trainer.precision,
        max_steps=config.trainer.max_steps,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
        default_root_dir=config.checkpointing.save_dir,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
    print(f'Training complete. Checkpoints saved to {ckpt_dir}')


def main():
    """Main entry point."""
    # Parse args and create config
    parser = Config.get_parser()
    args = parser.parse_args()
    cfg = Config.from_args(args)

    # Wrap with compatibility layer
    config = ConfigCompat(cfg)

    # Set seed
    L.seed_everything(config._config.seed)

    # Get tokenizer
    tokenizer = dataloader.get_tokenizer(config)

    # Run based on mode
    if config._config.mode == 'eval':
        generate_samples(config, tokenizer)
    elif config._config.mode == 'impute':
        # TODO: Add imputation mode
        print('Imputation mode not yet implemented')
    else:
        train(config, tokenizer)


if __name__ == '__main__':
    main()
