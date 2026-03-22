"""TVAE synthesizer — Variational Autoencoder for tabular data.

More stable than CTGAN, often produces similar or better quality with
fewer hyperparameter adjustments.

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
            "TVAE requires PyTorch. Install with: pip install 'synthforge[gan]'"
        )


def _require_cuda(model_name: str = "TVAE"):
    """Enforce CUDA for GAN/VAE models — CPU is prohibitively slow."""
    torch = _check_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{model_name} requires a CUDA-capable GPU. "
            f"Training on CPU is 50-100x slower. "
            f"Use synthesizer='gaussian_copula' for CPU workloads, "
            f"or set config.tvae.cuda=False to force CPU (NOT recommended)."
        )
    return torch


@register_synthesizer("tvae")
class TVAESynthesizer(BaseSynthesizer):
    """Tabular Variational Autoencoder.

    Encoder maps data to a latent Gaussian distribution, decoder reconstructs.
    Loss = reconstruction + KL divergence.
    """

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        encoder_dim: tuple[int, ...] = (128, 128),
        decoder_dim: tuple[int, ...] = (128, 128),
        latent_dim: int = 128,
        l2scale: float = 1e-5,
        loss_factor: int = 2,
        cuda: bool = True,
        random_state: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._epochs = epochs
        self._batch_size = batch_size
        self._enc_dim = encoder_dim
        self._dec_dim = decoder_dim
        self._latent_dim = latent_dim
        self._l2scale = l2scale
        self._loss_factor = loss_factor
        self._cuda = cuda
        self._random_state = random_state

        self._encoder = None
        self._decoder = None
        self._device = None
        self._data_dim = 0

    def _build_encoder(self, torch):
        layers = []
        input_dim = self._data_dim
        for dim in self._enc_dim:
            layers.extend([
                torch.nn.Linear(input_dim, dim),
                torch.nn.ReLU(),
            ])
            input_dim = dim
        return torch.nn.Sequential(*layers), input_dim

    def _build_decoder(self, torch):
        layers = []
        input_dim = self._latent_dim
        for dim in self._dec_dim:
            layers.extend([
                torch.nn.Linear(input_dim, dim),
                torch.nn.ReLU(),
            ])
            input_dim = dim
        layers.append(torch.nn.Linear(input_dim, self._data_dim))
        return torch.nn.Sequential(*layers)

    def fit(self, data: np.ndarray, column_names: list[str] | None = None) -> None:
        if self._cuda:
            torch = _require_cuda("TVAE")
        else:
            torch = _check_torch()
            logger.warning("TVAE on CPU — 50-100x slower. Use gaussian_copula for CPU.")

        n_samples, n_cols = data.shape
        self._data_dim = n_cols
        self._n_columns = n_cols
        self._column_names = column_names or [f"col_{i}" for i in range(n_cols)]

        self._device = torch.device("cuda" if self._cuda and torch.cuda.is_available() else "cpu")
        logger.info(
            "Fitting TVAE on %d samples, %d columns, device=%s", n_samples, n_cols, self._device
        )

        if self._random_state is not None:
            torch.manual_seed(self._random_state)

        # Build encoder (shared layers + mu/logvar heads)
        enc_body, enc_out_dim = self._build_encoder(torch)
        enc_body = enc_body.to(self._device)
        fc_mu = torch.nn.Linear(enc_out_dim, self._latent_dim).to(self._device)
        fc_logvar = torch.nn.Linear(enc_out_dim, self._latent_dim).to(self._device)

        # Build decoder
        self._decoder = self._build_decoder(torch).to(self._device)

        # Optimizer
        all_params = (
            list(enc_body.parameters())
            + list(fc_mu.parameters())
            + list(fc_logvar.parameters())
            + list(self._decoder.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=1e-3, weight_decay=self._l2scale)

        # Data tensor
        real_tensor = torch.FloatTensor(data).to(self._device)

        # Training loop
        steps_per_epoch = max(n_samples // self._batch_size, 1)

        for epoch in range(self._epochs):
            epoch_loss = 0.0

            for step in range(steps_per_epoch):
                idx = torch.randint(0, n_samples, (self._batch_size,))
                batch = real_tensor[idx]

                # Encode
                h = enc_body(batch)
                mu = fc_mu(h)
                logvar = fc_logvar(h)

                # Reparameterize
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                # Decode
                recon = self._decoder(z)

                # Loss = MSE reconstruction + KL divergence
                recon_loss = torch.nn.functional.mse_loss(recon, batch, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss / self._loss_factor + kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(
                    "TVAE Epoch %d/%d — Loss: %.4f",
                    epoch + 1, self._epochs, epoch_loss / steps_per_epoch,
                )

        # Store encoder parts for potential later use
        self._encoder = (enc_body, fc_mu, fc_logvar)
        self._fitted = True
        logger.info("TVAE training complete")

    def sample(self, n_rows: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Synthesizer not fitted. Call fit() first.")

        torch = _check_torch()
        logger.info("Sampling %d rows from TVAE", n_rows)

        self._decoder.eval()
        samples = []
        remaining = n_rows

        with torch.no_grad():
            while remaining > 0:
                batch = min(remaining, self._batch_size)
                z = torch.randn(batch, self._latent_dim, device=self._device)
                fake = self._decoder(z).cpu().numpy()
                samples.append(fake)
                remaining -= batch

        self._decoder.train()
        return np.concatenate(samples, axis=0)[:n_rows]
