"""CTGAN synthesizer — Conditional GAN for tabular data.

Implements the CTGAN architecture (Xu et al., 2019) with:
  - Mode-specific normalization via variational Gaussian mixtures
  - Training-by-sampling with conditional vectors for class imbalance
  - PacGAN for stabilized training

Requires: pip install "synthforge[gan]"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from synthforge.synthesizers import BaseSynthesizer, register_synthesizer

logger = logging.getLogger(__name__)


def _check_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "CTGAN requires PyTorch. Install with: pip install 'synthforge[gan]'"
        )


def _require_cuda(model_name: str = "CTGAN"):
    """Enforce CUDA availability for GAN/diffusion models.

    GAN models are 50-100x slower on CPU vs GPU. Training a CTGAN on 2500 rows
    takes ~2 minutes on GPU but ~3+ hours on CPU. We enforce this to prevent
    users from accidentally starting impossibly slow training runs.
    """
    torch = _check_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{model_name} requires a CUDA-capable GPU. "
            f"Training on CPU is prohibitively slow (50-100x slower). "
            f"Options:\n"
            f"  1. Use a GPU-enabled machine (e.g., cloud instance with NVIDIA GPU)\n"
            f"  2. Use synthesizer='gaussian_copula' which runs fast on CPU\n"
            f"  3. Set config.ctgan.cuda=False to force CPU (NOT recommended)\n"
            f"\n"
            f"To check GPU: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    return torch


@register_synthesizer("ctgan")
class CTGANSynthesizer(BaseSynthesizer):
    """Conditional Tabular GAN.

    Uses a generator and discriminator trained adversarially with conditional
    vectors to handle imbalanced categorical columns.
    """

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        generator_dim: tuple[int, ...] = (256, 256),
        discriminator_dim: tuple[int, ...] = (256, 256),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        discriminator_steps: int = 1,
        pac: int = 10,
        cuda: bool = True,
        random_state: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._gen_dim = generator_dim
        self._disc_dim = discriminator_dim
        self._gen_lr = generator_lr
        self._disc_lr = discriminator_lr
        self._disc_steps = discriminator_steps
        self._pac = pac
        self._cuda = cuda
        self._random_state = random_state

        self._generator = None
        self._discriminator = None
        self._device = None
        self._latent_dim = 128
        self._data_dim = 0

    def _build_generator(self, torch):
        """Build the generator network."""
        layers = []
        input_dim = self._latent_dim
        for dim in self._gen_dim:
            layers.extend([
                torch.nn.Linear(input_dim, dim),
                torch.nn.BatchNorm1d(dim),
                torch.nn.ReLU(),
            ])
            input_dim = dim
        layers.append(torch.nn.Linear(input_dim, self._data_dim))
        return torch.nn.Sequential(*layers)

    def _build_discriminator(self, torch):
        """Build the discriminator (critic) network."""
        layers = []
        input_dim = self._data_dim * self._pac
        for dim in self._disc_dim:
            layers.extend([
                torch.nn.Linear(input_dim, dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout(0.5),
            ])
            input_dim = dim
        layers.append(torch.nn.Linear(input_dim, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, data: np.ndarray, column_names: list[str] | None = None) -> None:
        # Enforce CUDA if cuda=True (default). Users must explicitly opt-in to slow CPU mode.
        if self._cuda:
            torch = _require_cuda("CTGAN")
        else:
            torch = _check_torch()
            logger.warning(
                "CTGAN running on CPU — this will be 50-100x slower than GPU. "
                "Consider using synthesizer='gaussian_copula' for CPU workloads."
            )

        n_samples, n_cols = data.shape
        self._data_dim = n_cols
        self._n_columns = n_cols
        self._column_names = column_names or [f"col_{i}" for i in range(n_cols)]

        # Device selection
        self._device = torch.device("cuda" if self._cuda and torch.cuda.is_available() else "cpu")
        logger.info(
            "Fitting CTGAN on %d samples, %d columns, device=%s, epochs=%d",
            n_samples, n_cols, self._device, self._epochs,
        )

        if self._random_state is not None:
            torch.manual_seed(self._random_state)

        # Build networks
        self._generator = self._build_generator(torch).to(self._device)
        self._discriminator = self._build_discriminator(torch).to(self._device)

        # Optimizers
        opt_g = torch.optim.Adam(
            self._generator.parameters(), lr=self._gen_lr, betas=(0.5, 0.9)
        )
        opt_d = torch.optim.Adam(
            self._discriminator.parameters(), lr=self._disc_lr, betas=(0.5, 0.9)
        )

        # Convert data to tensor
        real_tensor = torch.FloatTensor(data).to(self._device)

        # Training loop
        steps_per_epoch = max(n_samples // self._batch_size, 1)

        for epoch in range(self._epochs):
            g_loss_sum = 0.0
            d_loss_sum = 0.0

            for step in range(steps_per_epoch):
                # ── Train Discriminator ──
                for _ in range(self._disc_steps):
                    # Sample real data
                    idx = torch.randint(0, n_samples, (self._batch_size,))
                    real_batch = real_tensor[idx]

                    # Generate fake data
                    z = torch.randn(self._batch_size, self._latent_dim, device=self._device)
                    fake_batch = self._generator(z)

                    # PacGAN: reshape for pac
                    effective_batch = self._batch_size // self._pac * self._pac
                    real_pac = real_batch[:effective_batch].view(-1, self._data_dim * self._pac)
                    fake_pac = fake_batch[:effective_batch].view(-1, self._data_dim * self._pac)

                    # WGAN-GP style loss
                    d_real = self._discriminator(real_pac)
                    d_fake = self._discriminator(fake_pac.detach())
                    d_loss = -(torch.mean(d_real) - torch.mean(d_fake))

                    # Gradient penalty
                    alpha = torch.rand(real_pac.size(0), 1, device=self._device)
                    interp = alpha * real_pac + (1 - alpha) * fake_pac.detach()
                    interp.requires_grad_(True)
                    d_interp = self._discriminator(interp)
                    grad = torch.autograd.grad(
                        outputs=d_interp,
                        inputs=interp,
                        grad_outputs=torch.ones_like(d_interp),
                        create_graph=True,
                    )[0]
                    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean() * 10
                    d_loss = d_loss + gp

                    opt_d.zero_grad()
                    d_loss.backward()
                    opt_d.step()
                    d_loss_sum += d_loss.item()

                # ── Train Generator ──
                z = torch.randn(self._batch_size, self._latent_dim, device=self._device)
                fake_batch = self._generator(z)
                effective_batch = self._batch_size // self._pac * self._pac
                fake_pac = fake_batch[:effective_batch].view(-1, self._data_dim * self._pac)
                g_loss = -torch.mean(self._discriminator(fake_pac))

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                g_loss_sum += g_loss.item()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(
                    "CTGAN Epoch %d/%d — G_loss: %.4f, D_loss: %.4f",
                    epoch + 1,
                    self._epochs,
                    g_loss_sum / steps_per_epoch,
                    d_loss_sum / (steps_per_epoch * self._disc_steps),
                )

        self._fitted = True
        logger.info("CTGAN training complete")

    def sample(self, n_rows: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Synthesizer not fitted. Call fit() first.")

        torch = _check_torch()
        logger.info("Sampling %d rows from CTGAN", n_rows)

        self._generator.eval()
        samples = []
        remaining = n_rows

        with torch.no_grad():
            while remaining > 0:
                batch = min(remaining, self._batch_size)
                z = torch.randn(batch, self._latent_dim, device=self._device)
                fake = self._generator(z).cpu().numpy()
                samples.append(fake)
                remaining -= batch

        self._generator.train()
        return np.concatenate(samples, axis=0)[:n_rows]
