"""Strategy engine — selects and configures the optimal synthesizer pipeline
based on data characteristics, hardware availability, and user preferences.

Selection hierarchy (when auto):
  1. If CUDA available → prefer SOTA diffusion models (TabSyn > TabDDPM > CTGAN)
  2. If only CPU + torch → warn and fallback to Gaussian Copula
  3. If no torch → Gaussian Copula (always works, no deps)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from synthforge.config import SynthForgeConfig, SynthesizerType, DataStrategy
from synthforge.metadata import Metadata, SDType
from synthforge.synthesizers import BaseSynthesizer, get_synthesizer, list_synthesizers

logger = logging.getLogger(__name__)


# ── Hardware detection ─────────────────────────────────────────────

def _torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _transformers_available() -> bool:
    try:
        import transformers
        return True
    except ImportError:
        return False


# ── Strategy recommendation ────────────────────────────────────────

def recommend_strategy(metadata: Metadata) -> DataStrategy:
    """Recommend a data strategy based on metadata analysis."""
    try:
        return DataStrategy(metadata.data_strategy)
    except ValueError:
        return DataStrategy.MIXED


def recommend_synthesizer(
    metadata: Metadata,
    strategy: DataStrategy,
    config: SynthForgeConfig,
) -> SynthesizerType:
    """Recommend the best synthesizer for data + hardware.

    Priority:
      CUDA available → TabSyn (best overall), TabDDPM (best for time-series),
                        CTGAN (best for high-cardinality categorical)
      CPU only       → Gaussian Copula (fast, no GPU needed)
    """
    n_cols = len(metadata.columns)
    n_cat = len(metadata.categorical_columns)
    n_num = len(metadata.numerical_columns)

    high_card = any(
        m.cardinality > 50
        for m in metadata.columns.values()
        if m.sdtype == SDType.CATEGORICAL
    )

    has_cuda = _cuda_available()
    has_torch = _torch_available()

    # ── CUDA path: use SOTA models ─────────────────────────────
    if has_cuda:
        registered = set(list_synthesizers())

        if strategy == DataStrategy.NUMERICAL:
            # TabSyn excels at numerical: 86% better marginal fidelity than CTGAN
            if "tabsyn" in registered:
                return SynthesizerType.TABSYN
            return SynthesizerType.TVAE

        if strategy == DataStrategy.CATEGORICAL:
            # CTGAN's conditional vectors handle class imbalance best
            if high_card:
                return SynthesizerType.CTGAN
            # TabSyn also handles categorical well via latent space
            if "tabsyn" in registered:
                return SynthesizerType.TABSYN
            return SynthesizerType.CTGAN

        if strategy == DataStrategy.TIMESERIES:
            # TabDDPM's dual diffusion captures temporal patterns well
            if "tabddpm" in registered:
                return SynthesizerType.TABDDPM
            return SynthesizerType.TVAE

        # Mixed: TabSyn's latent space unifies mixed types naturally
        if "tabsyn" in registered:
            return SynthesizerType.TABSYN
        if high_card or n_cat > n_num:
            return SynthesizerType.CTGAN
        return SynthesizerType.TVAE

    # ── CPU-only path: Gaussian Copula only ────────────────────
    if has_torch:
        logger.warning(
            "PyTorch found but NO CUDA GPU detected. Neural synthesizers (CTGAN, TVAE, "
            "TabDDPM, TabSyn) require GPU and are 50-100x slower on CPU. "
            "Using Gaussian Copula (fast CPU synthesizer)."
        )

    return SynthesizerType.GAUSSIAN_COPULA


# ── Synthesizer instantiation ──────────────────────────────────────

def create_synthesizer(
    synth_type: SynthesizerType,
    config: SynthForgeConfig,
) -> BaseSynthesizer:
    """Instantiate a synthesizer from type and config."""

    if synth_type == SynthesizerType.GAUSSIAN_COPULA:
        cls = get_synthesizer("gaussian_copula")
        return cls(
            default_distribution=config.gaussian_copula.default_distribution,
            random_state=config.generation.seed,
        )

    if synth_type == SynthesizerType.CTGAN:
        cls = get_synthesizer("ctgan")
        return cls(
            epochs=config.ctgan.epochs,
            batch_size=config.ctgan.batch_size,
            generator_dim=config.ctgan.generator_dim,
            discriminator_dim=config.ctgan.discriminator_dim,
            generator_lr=config.ctgan.generator_lr,
            discriminator_lr=config.ctgan.discriminator_lr,
            discriminator_steps=config.ctgan.discriminator_steps,
            pac=config.ctgan.pac,
            cuda=config.ctgan.cuda,
            random_state=config.generation.seed,
        )

    if synth_type == SynthesizerType.TVAE:
        cls = get_synthesizer("tvae")
        return cls(
            epochs=config.tvae.epochs,
            batch_size=config.tvae.batch_size,
            encoder_dim=config.tvae.encoder_dim,
            decoder_dim=config.tvae.decoder_dim,
            l2scale=config.tvae.l2scale,
            loss_factor=config.tvae.loss_factor,
            cuda=config.tvae.cuda,
            random_state=config.generation.seed,
        )

    if synth_type == SynthesizerType.TABDDPM:
        cls = get_synthesizer("tabddpm")
        return cls(
            n_timesteps=config.tabddpm.n_timesteps,
            epochs=config.tabddpm.epochs,
            batch_size=config.tabddpm.batch_size,
            lr=config.tabddpm.lr,
            hidden_dims=config.tabddpm.hidden_dims,
            scheduler=config.tabddpm.scheduler,
            cuda=config.tabddpm.cuda,
            random_state=config.generation.seed,
        )

    if synth_type == SynthesizerType.TABSYN:
        cls = get_synthesizer("tabsyn")
        return cls(
            vae_epochs=config.tabsyn.vae_epochs,
            vae_lr=config.tabsyn.vae_lr,
            vae_latent_dim=config.tabsyn.vae_latent_dim,
            vae_encoder_dim=config.tabsyn.vae_encoder_dim,
            vae_decoder_dim=config.tabsyn.vae_decoder_dim,
            diff_epochs=config.tabsyn.diff_epochs,
            diff_lr=config.tabsyn.diff_lr,
            n_timesteps=config.tabsyn.n_timesteps,
            diff_hidden_dims=config.tabsyn.diff_hidden_dims,
            batch_size=config.tabsyn.batch_size,
            cuda=config.tabsyn.cuda,
            random_state=config.generation.seed,
        )

    if synth_type == SynthesizerType.GREAT:
        cls = get_synthesizer("great")
        return cls(
            model_name=config.great.model_name,
            epochs=config.great.epochs,
            batch_size=config.great.batch_size,
            lr=config.great.lr,
            max_length=config.great.max_length,
            temperature=config.great.temperature,
            top_k=config.great.top_k,
            cuda=config.great.cuda,
            random_state=config.generation.seed,
        )

    raise ValueError(
        f"Unknown synthesizer type: '{synth_type}'. "
        f"Available: {list_synthesizers()}"
    )


# ── Orchestrator ───────────────────────────────────────────────────

class StrategyEngine:
    """Orchestrates strategy selection and synthesizer creation."""

    def __init__(self, config: SynthForgeConfig):
        self._config = config

    def resolve(self, metadata: Metadata) -> tuple[DataStrategy, SynthesizerType, BaseSynthesizer]:
        """Resolve strategy, synthesizer type, and instantiate."""
        # Resolve strategy
        if self._config.strategy == DataStrategy.AUTO:
            strategy = recommend_strategy(metadata)
        else:
            strategy = self._config.strategy

        # Resolve synthesizer
        if self._config.synthesizer == SynthesizerType.AUTO:
            synth_type = recommend_synthesizer(metadata, strategy, self._config)
        else:
            synth_type = self._config.synthesizer

        # Instantiate
        synthesizer = create_synthesizer(synth_type, self._config)

        logger.info(
            "Strategy resolved: strategy=%s, synthesizer=%s, cuda=%s",
            strategy.value,
            synth_type.value,
            _cuda_available(),
        )
        return strategy, synth_type, synthesizer
