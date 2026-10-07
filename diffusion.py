import itertools
import math
from dataclasses import dataclass

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils

LOG2 = math.log(2)

# Soft labeling configuration for time series tokens
# Note: ts_token_end is set dynamically based on config.model.n_bins
SOFT_LABEL_CONFIG = {
    'ts_token_start': 1,    # Time series tokens start at index 1 (0 is mask)
    'ts_token_end': 40,     # Default, will be overridden by n_bins from config
    'sigma': 1.0,           # Gaussian std for soft labels (controls spread)
    'window': 2,            # 0 = hard labeling, >0 = soft labeling with ±window neighbors
}


def get_soft_label_config(n_bins, window=None):
    """Get soft label config with correct ts_token_end based on n_bins."""
    return {
        'ts_token_start': 1,
        'ts_token_end': n_bins,
        'sigma': SOFT_LABEL_CONFIG['sigma'],
        'window': window if window is not None else SOFT_LABEL_CONFIG['window'],
    }


def _create_soft_labels(x0, vocab_size, config=SOFT_LABEL_CONFIG):
    """Create soft labels for time series tokens (1-n_bins), hard labels for mask(0).
    Vectorized implementation for speed.

    Handles both 2D (B, T) and 3D (B, T, C) inputs.
    Output: (B, T, vocab_size) or (B, T, C, vocab_size)

    NOTE: This uses token ID distance. For boundary-aware soft labels,
    use _create_soft_labels_continuous() instead.
    """
    device = x0.device
    original_shape = x0.shape
    ts_start = config['ts_token_start']
    ts_end = config['ts_token_end']
    sigma = config['sigma']
    window = config['window']
    n_ts_tokens = ts_end - ts_start + 1

    # Flatten to 2D for processing if needed
    if x0.ndim == 3:
        B, T, C = x0.shape
        x0_flat = x0.reshape(-1, T * C)  # Not needed, process as-is
        # Actually, reshape to (B*C, T) or keep as (B, T, C) and add extra dim
        # Easier: reshape (B, T, C) to (B*T*C,) then back
        x0_flat = x0.reshape(-1)  # (B*T*C,)
    else:
        x0_flat = x0.reshape(-1)  # (B*T,)

    # centers: (N, 1) where N = total elements
    centers = x0_flat.float().unsqueeze(-1)

    # ts_range: (n_ts_tokens,)
    ts_range = torch.arange(ts_start, ts_end + 1, device=device).float()

    # distances: (N, n_ts_tokens)
    distances = ts_range - centers

    # Gaussian weights with window
    within_window = (distances.abs() <= window).float()
    gaussian_weights = torch.exp(-0.5 * (distances / sigma) ** 2) * within_window

    # Normalize
    weight_sum = gaussian_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    gaussian_weights = gaussian_weights / weight_sum

    # Build full soft_labels: (N, vocab_size)
    soft_labels = torch.zeros(x0_flat.shape[0], vocab_size, device=device)
    soft_labels[:, ts_start:ts_end+1] = gaussian_weights

    # For mask tokens (0), use one-hot at index 0
    # Handle float labels: mask is 0, valid tokens >= 1
    if x0_flat.dtype.is_floating_point:
        mask_positions = (x0_flat < 0.5)
    else:
        mask_positions = (x0_flat == 0)
    if mask_positions.any():
        mask_one_hot = torch.zeros(vocab_size, device=device)
        mask_one_hot[0] = 1.0
        soft_labels[mask_positions] = mask_one_hot

    # Reshape back to original shape + vocab_size
    soft_labels = soft_labels.reshape(*original_shape, vocab_size)

    return soft_labels



def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = tokenizer
        # n_bins from config, default 40
        self.n_bins = getattr(self.config.model, 'n_bins', 40)
        # output_range: [-r, r] for output, input is always [-1, 1]
        self.output_range = getattr(self.config.model, 'output_range', 1.5)
        # Input: n_bins+1 vocab (0=mask, 1~n_bins for [-1,1])
        # Output: n_bins_output+1 vocab (0=mask, 1~n_bins_output for [-output_range, output_range])
        self.input_vocab_size = self.n_bins + 1
        self.n_bins_output = int(self.n_bins * self.output_range)
        self.output_vocab_size = self.n_bins_output + 1
        self.sampler = self.config.sampling.predictor
        self.n_channels = getattr(self.config.model, 'n_channels', 1)  # multivariate support
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables

        # FFT loss weight (0 = disabled)
        self.fft_weight = getattr(self.config.training, 'fft_weight', 0.0)

        # Soft label sigma ratio: sigma = sigma_ratio * bin_width
        # 1.0 = probability spreads across ~4 neighboring bins
        self.soft_label_sigma_ratio = getattr(self.config.training, 'soft_label_sigma_ratio', 1.0)
        # Soft label window: 0 = one-hot (hard labeling), >0 = soft labeling
        self.soft_label_window = getattr(self.config.training, 'soft_label_window', 2)

        # mask token setup
        self.mask_index = 0  # mask is always 0 for time series

        self.backbone = models.dit.DiTDual(
            self.config,
            input_vocab_size=self.input_vocab_size,
            output_vocab_size=self.output_vocab_size)

        self.softplus = torch.nn.Softplus()
        # metrics (only nll)
        metrics = torchmetrics.MetricCollection({
            'nll': NLL(),
        })
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(),
                                self.noise.parameters()),
                decay=self.config.training.ema)
        else:
            self.ema = None

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._validate_configuration()

    def _validate_configuration(self):
        assert not (self.change_of_variables and self.importance_sampling)
        assert self.sampler in {'ddpm', 'ddpm_cache'}

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress']['current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
            'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
            'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed)
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (distributed
                    and self.fast_forward_epochs is not None
                    and self.fast_forward_batches is not None):
                dl_sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': (self.fast_forward_batches
                                * self.config.loader.batch_size)})
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))

    def _process_sigma(self, sigma):
        if sigma is None:
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma):
        """Forward pass. Returns logits with SUBS parameterization."""
        sigma = self._process_sigma(sigma)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            logits = self.backbone(x, sigma)

        # SUBS parameterization: mask token gets -inf
        logits[..., self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits

    def _compute_loss(self, batch, prefix):
        raw_values = batch['raw_values']  # (B, T, C) raw continuous values
        bin_width = batch.get('bin_width', 0.05)
        if isinstance(bin_width, torch.Tensor):
            bin_width = bin_width[0].item()

        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask']
        else:
            attention_mask = None

        # For multivariate (B, T, C), expand attention_mask to (B, T, C)
        if raw_values.ndim == 3 and attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(-1).expand_as(raw_values)

        # Handle missing_mask: exclude missing positions from loss
        # missing_mask: True where originally NaN (should be excluded from loss)
        missing_mask = batch.get('missing_mask', None)
        if missing_mask is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(raw_values, dtype=torch.float)
            # Set attention_mask to 0 for missing positions
            attention_mask = attention_mask * (~missing_mask).float()

        losses = self._loss(raw_values, attention_mask, bin_width, missing_mask=missing_mask)
        loss = losses.loss

        if prefix == 'train':
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        elif prefix == 'test':
            self.test_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, prefix='train')
        self.log(name='trainer/loss',
                 value=loss.item(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))
            self.ema.copy_to(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))
        self.backbone.eval()
        self.noise.eval()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, prefix='val')

    def on_validation_epoch_end(self):
        if self.ema:
            self.ema.restore(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(),
                            self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        # Create scheduler based on config
        scheduler_type = getattr(self.config.training, 'lr_scheduler', 'constant_warmup')
        warmup_steps = getattr(self.config.training, 'warmup_steps', 2500)

        if scheduler_type == 'constant_warmup':
            from transformers import get_constant_schedule_with_warmup
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler_type == 'cosine_warmup':
            from transformers import get_cosine_schedule_with_warmup
            max_steps = self.config.trainer.max_steps
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
        else:  # 'none' or unknown
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]

    def q_xt(self, x, move_chance):
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def q_xt_time(self, raw_values, move_chance, min_range=1.0, label_noise=True):
        """Diffusion forward process for raw continuous time series.

        1. Determine move_indices (which positions to mask)
        2. Compute min/max from UNMASKED values only (per window, per channel)
        3. Normalize → clip to [-output_range, output_range] → tokenize to 1~n_bins_output
        4. Create x_input for model (input_vocab_size) from xt

        Model input: n_bins+1 tokens (0=mask, 1~n_bins for [-1,1])
        Model output: n_bins_output+1 tokens (0=mask, 1~n_bins_output for [-output_range, output_range])

        Args:
            raw_values: (B, T, C) raw continuous values (not normalized)
            move_chance: (B, 1, 1) probability of masking each position
            min_range: minimum range for normalization (handles constant values)
            label_noise: whether to add ±0.5 uniform noise before rounding

        Returns:
            x_input: (B, T, C) model input tokens (0=mask, 1~n_bins)
            x0: (B, T, C) ground truth tokens in output space (1~n_bins_output)
            window_mins: (B, 1, C) per-window, per-channel min values
            window_maxs: (B, 1, C) per-window, per-channel max values
        """
        B, T, C = raw_values.shape
        device = raw_values.device
        n_bins_output = self.n_bins_output  # e.g., 60 for n_bins=40, output_range=1.5
        output_range = self.output_range    # e.g., 1.5 for [-1.5, 1.5]

        # 1. Determine which positions to mask
        move_indices = torch.rand(B, T, C, device=device) < move_chance  # True = masked

        # 2. Compute min/max from UNMASKED values only (per window, per channel) - vectorized
        unmasked = ~move_indices  # (B, T, C), True = observed

        # Mask out the masked positions with inf/-inf for min/max computation
        masked_for_min = torch.where(unmasked, raw_values, torch.tensor(float('inf'), device=device))
        masked_for_max = torch.where(unmasked, raw_values, torch.tensor(float('-inf'), device=device))

        # Compute min/max along time dimension
        window_mins = masked_for_min.min(dim=1, keepdim=True).values  # (B, 1, C)
        window_maxs = masked_for_max.max(dim=1, keepdim=True).values  # (B, 1, C)

        # Handle all-masked case: fallback to full data min/max
        all_masked = ~unmasked.any(dim=1, keepdim=True)  # (B, 1, C)
        full_mins = raw_values.min(dim=1, keepdim=True).values
        full_maxs = raw_values.max(dim=1, keepdim=True).values
        window_mins = torch.where(all_masked, full_mins, window_mins)
        window_maxs = torch.where(all_masked, full_maxs, window_maxs)

        # Handle constant values (range < min_range)
        data_range = window_maxs - window_mins
        needs_expand = data_range < min_range
        c_mean = (window_mins + window_maxs) / 2
        window_mins = torch.where(needs_expand, c_mean - min_range / 2, window_mins)
        window_maxs = torch.where(needs_expand, c_mean + min_range / 2, window_maxs)

        # 3. Normalize to [-1, 1] based on unmasked min/max
        # Masked values may fall outside [-1, 1]
        normalized = (raw_values - window_mins) / (window_maxs - window_mins) * 2 - 1

        # 4. Clip to [-output_range, output_range] and tokenize to [1, n_bins_output]
        normalized = normalized.clamp(-output_range, output_range)
        # [-output_range, output_range] → [1, n_bins_output]
        float_labels = (normalized + output_range) / (2 * output_range) * (n_bins_output - 1) + 1

        # 5. Add label noise and round to tokens
        # token_offset = n_bins_output * (output_range - 1) / (2 * output_range)
        # e.g., output_range=1.5, n_bins_output=60: offset=10, unmasked=[11,50]
        # e.g., output_range=2.0, n_bins_output=80: offset=20, unmasked=[21,60]
        # Use round() instead of int() to avoid floating point precision issues
        token_offset = round(n_bins_output * (output_range - 1) / (2 * output_range))
        unmasked_min = token_offset + 1
        unmasked_max = n_bins_output - token_offset

        if label_noise:
            noise = torch.rand_like(float_labels) - 0.5  # uniform [-0.5, 0.5]
            noised_labels = float_labels + noise
            # Unmasked positions: clamp to valid input range
            # Masked positions: can be [1, n_bins_output]
            noised_labels = torch.where(
                move_indices,
                noised_labels.clamp(1, n_bins_output),
                noised_labels.clamp(unmasked_min, unmasked_max)
            )
            tokens = torch.round(noised_labels).long()
        else:
            tokens = torch.round(float_labels).clamp(1, n_bins_output).long()

        # x0: ground truth tokens in output space (1~n_bins_output)
        x0 = tokens

        # xt: masked version (masked positions = 0)
        xt = torch.where(move_indices, self.mask_index, x0)

        # x_input: model input (n_bins+1 vocab)
        # xt is 0 (mask) or unmasked_min~unmasked_max (unmasked, since unmasked values are in [-1,1])
        # Convert to 1~n_bins by subtracting offset
        x_input = torch.where(xt > 0, xt - token_offset, xt)

        return x_input, x0, window_mins, window_maxs

    def _sample_prior(self, *batch_dims):
        """Sample from prior (all mask tokens).

        For multivariate: batch_dims = (B, T, C)
        For univariate: batch_dims = (B, T)
        """
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        """DDPM update with caching.

        x: (B, T) or (B, T, C)
        """
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1

        # Expand move_chance based on input dimensionality
        if x.ndim == 3:  # (B, T, C)
            move_chance_t = t[:, None, None, None]  # (B, 1, 1, 1)
            move_chance_s = (t - dt)[:, None, None, None]
        else:  # (B, T)
            move_chance_t = t[:, None, None]  # (B, 1, 1)
            move_chance_s = (t - dt)[:, None, None]

        if p_x0 is None:
            logits = self.forward(x, sigma_t)
            p_x0 = logits.exp()

        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[..., self.mask_index] = move_chance_s.squeeze(-1)
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt):
        """DDPM update step.

        x: (B, T) or (B, T, C)
        """
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)

        # Expand based on input dimensionality
        if x.ndim == 3:  # (B, T, C)
            move_chance_t = move_chance_t[:, None, None, None]
            move_chance_s = move_chance_s[:, None, None, None]
        else:
            move_chance_t = move_chance_t[:, None, None]
            move_chance_s = move_chance_s[:, None, None]

        unet_conditioning = sigma_t
        logits = self.forward(x, unet_conditioning)
        log_p_x0 = logits
        assert move_chance_t.ndim == log_p_x0.ndim
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[..., self.mask_index] = move_chance_s.squeeze(-1)
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5):
        """Generate samples from the model. Returns (B, T, C) for multivariate."""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        if num_steps is None:
            num_steps = self.config.sampling.steps

        # For multivariate: x is (B, T, C)
        if self.n_channels > 1:
            x = self._sample_prior(
                batch_size_per_gpu,
                self.config.model.length,
                self.n_channels).to(self.device)
        else:
            x = self._sample_prior(
                batch_size_per_gpu,
                self.config.model.length).to(self.device)

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                x = self._ddpm_update(x, t, dt)
            else:  # ddpm_cache
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache)
                if not torch.allclose(x_next, x):
                    p_x0_cache = None
                x = x_next

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            unet_conditioning = self.noise(t)[0]
            logits = self.forward(x, unet_conditioning)
            x = logits.argmax(dim=-1)  # (B, T) or (B, T, C) token ids

        # Convert token ids to continuous values (bin centers)
        n_bins = self.n_bins
        bin_edges = torch.linspace(-1, 1, n_bins + 1, device=x.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (n_bins,)

        # Get bin index from token id (token_id - 1 since 0 is mask)
        bin_idx = (x - 1).clamp(0, n_bins - 1)

        return bin_centers[bin_idx]

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        if self.ema:
            self.ema.store(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))
            self.ema.copy_to(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(num_steps=num_steps, eps=eps)
        if self.ema:
            self.ema.restore(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))
        self.backbone.train()
        self.noise.train()
        return samples

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _forward_pass_diffusion(self, raw_values, bin_width=0.05, missing_mask=None):
        """Forward pass for diffusion training (SUBS, continuous time).

        raw_values: (B, T, C) raw continuous values (not normalized)
        missing_mask: (B, T, C) boolean, True = originally missing (NaN) position
        """
        t = self._sample_t(raw_values.shape[0], raw_values.device)

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
            f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None, None]  # (B, 1, 1)
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None, None])  # (B, 1, 1)

        # Use q_xt_time: mask first, then compute min/max from unmasked, then normalize
        label_noise = getattr(self.config.training, 'label_noise', True)
        min_range = getattr(self.config.data, 'min_range', 1.0)
        x_input, x0, window_mins, window_maxs = self.q_xt_time(
            raw_values, move_chance, min_range=min_range, label_noise=label_noise
        )

        # For missing positions, always keep mask token
        if missing_mask is not None:
            x_input = torch.where(missing_mask, self.mask_index, x_input)
            x0 = torch.where(missing_mask, self.mask_index, x0)

        model_output = self.forward(x_input, unet_conditioning)
        utils.print_nans(model_output, 'model_output')

        # Only compute loss on MASKED positions (x_input == 0)
        masked_positions = (x_input == self.mask_index)

        # Soft labels for output space
        n_bins_output = self.n_bins_output
        soft_label_config = get_soft_label_config(n_bins_output, window=self.soft_label_window)
        soft_labels = _create_soft_labels(x0, model_output.shape[-1], soft_label_config)
        log_p_theta = (soft_labels * model_output).sum(dim=-1)

        # Zero out unmasked positions
        log_p_theta = log_p_theta * masked_positions.float()

        # FFT loss (only if weight > 0)
        if self.fft_weight > 0:
            gt_normalized = (raw_values - window_mins) / (window_maxs - window_mins) * 2 - 1
            gt_normalized = gt_normalized.clamp(-self.output_range, self.output_range)
            fft_loss = self._compute_fft_loss(model_output, x_input, gt_normalized, masked_positions)
        else:
            fft_loss = torch.zeros(x_input.shape[0], device=x_input.device)

        if self.change_of_variables or self.importance_sampling:
            diffusion_loss = log_p_theta * torch.log1p(- torch.exp(- self.noise.sigma_min))
            return diffusion_loss, fft_loss

        # Time-dependent weight
        weight_1d = dsigma / torch.expm1(sigma)  # (B,)
        weight = weight_1d[:, None, None]  # (B, 1, 1)

        # Apply weight to diffusion loss
        diffusion_loss = - log_p_theta * weight  # (B, T, C)

        # Apply same weight to FFT loss
        fft_loss = fft_loss * weight_1d  # (B,)

        return diffusion_loss, fft_loss

    def _compute_fft_loss(self, model_output, x_input, gt_normalized, masked_positions):
        """Compute FFT loss between predicted and ground truth continuous values.

        Uses softmax + weighted mean (expectation) for differentiable prediction.
        Gradient flows through model_output → softmax → weighted sum.

        Args:
            model_output: log probabilities (B, T, C, output_vocab_size) - output space
            x_input: model input tokens (B, T, C), 0=mask, 1~n_bins for input space
            gt_normalized: ground truth normalized values (B, T, C) in [-output_range, output_range]
            masked_positions: (B, T, C) boolean, True = masked position

        Returns:
            fft_loss: per-sample loss (B,)
        """
        n_bins_output = self.n_bins_output
        output_range = self.output_range
        device = model_output.device

        # Bin centers for weighted mean: n_bins_output values in [-output_range, output_range]
        bin_edges = torch.linspace(-output_range, output_range, n_bins_output + 1, device=device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # (n_bins_output,)

        # Pred: softmax → weighted mean (differentiable)
        logits_bins = model_output[..., 1:]  # exclude mask token, shape (B, T, C, n_bins_output)
        probs = F.softmax(logits_bins, dim=-1)  # (B, T, C, n_bins_output)
        pred_continuous = (probs * bin_centers).sum(dim=-1)  # (B, T, C) - gradient flows!

        # For unmasked positions use GT, for masked use model prediction
        pred_continuous = torch.where(masked_positions, pred_continuous, gt_normalized)

        # Compute FFT along time dimension
        # (B, T, C) -> transpose to (B, C, T) for FFT along time
        pred_fft = torch.fft.fft(pred_continuous.transpose(1, 2), dim=-1)
        gt_fft = torch.fft.fft(gt_normalized.transpose(1, 2), dim=-1)

        # L1 loss per sample: mean over (C, T) -> (B,)
        fft_loss = torch.abs(pred_fft.real - gt_fft.real).mean(dim=[1, 2]) + \
                   torch.abs(pred_fft.imag - gt_fft.imag).mean(dim=[1, 2])

        return fft_loss  # (B,)

    def _loss(self, raw_values, attention_mask, bin_width=0.05, missing_mask=None):
        if missing_mask is not None:
            missing_mask = missing_mask.to(raw_values.device)

        loss, fft_loss = self._forward_pass_diffusion(
            raw_values, bin_width, missing_mask=missing_mask)

        nlls = loss * attention_mask
        count = attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count
        total_loss = token_nll

        # Add FFT loss if weight > 0
        if fft_loss is not None and self.fft_weight > 0:
            fft_loss_mean = fft_loss.mean()
            total_loss = total_loss + self.fft_weight * fft_loss_mean
            if self.training:
                self.log('train/fft_loss', fft_loss_mean, prog_bar=True)
                self.log('train/fft_loss_weighted', self.fft_weight * fft_loss_mean, prog_bar=False)

        return Loss(loss=total_loss, nlls=nlls, token_mask=attention_mask)
