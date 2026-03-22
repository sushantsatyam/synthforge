"""Synthesizer base class and registry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Global registry for synthesizer plugins
_REGISTRY: dict[str, type[BaseSynthesizer]] = {}


def register_synthesizer(name: str):
    """Decorator to register a synthesizer class."""
    def decorator(cls: type[BaseSynthesizer]) -> type[BaseSynthesizer]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_synthesizer(name: str) -> type[BaseSynthesizer]:
    """Look up a synthesizer by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown synthesizer '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_synthesizers() -> list[str]:
    """List all registered synthesizer names."""
    return sorted(_REGISTRY.keys())


class BaseSynthesizer(ABC):
    """Abstract base class following sklearn-style fit/sample API."""

    def __init__(self, **kwargs: Any):
        self._fitted = False
        self._n_columns = 0
        self._column_names: list[str] = []

    @abstractmethod
    def fit(self, data: np.ndarray, column_names: list[str] | None = None) -> None:
        """Train the generative model on transformed numerical data.

        Args:
            data: 2D numpy array of shape (n_samples, n_features).
            column_names: Optional names for each column.
        """

    @abstractmethod
    def sample(self, n_rows: int) -> np.ndarray:
        """Generate n_rows of synthetic data.

        Returns:
            2D numpy array of shape (n_rows, n_features).
        """

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def sample_dataframe(self, n_rows: int) -> pd.DataFrame:
        """Sample and return as DataFrame with column names."""
        data = self.sample(n_rows)
        return pd.DataFrame(data, columns=self._column_names)


# ── Auto-register built-in synthesizers ────────────────────────────
# Importing each module triggers its @register_synthesizer decorator.

from synthforge.synthesizers.gaussian_copula import GaussianCopulaSynthesizer  # noqa: E402, F401

# Conditionally import torch-dependent synthesizers
try:
    from synthforge.synthesizers.ctgan import CTGANSynthesizer  # noqa: E402, F401
    from synthforge.synthesizers.tvae import TVAESynthesizer  # noqa: E402, F401
except ImportError:
    pass  # torch not installed — GAN/VAE synthesizers unavailable

# SOTA diffusion models (require torch + CUDA)
try:
    from synthforge.synthesizers.tabddpm import TabDDPMSynthesizer  # noqa: E402, F401
    from synthforge.synthesizers.tabsyn import TabSynSynthesizer  # noqa: E402, F401
except ImportError:
    pass  # torch not installed — diffusion synthesizers unavailable

# LLM-based generation (requires torch + transformers)
try:
    from synthforge.synthesizers.great import GReaTSynthesizer  # noqa: E402, F401
except ImportError:
    pass  # transformers not installed
