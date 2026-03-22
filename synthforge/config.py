"""Configuration models for SynthForge using Pydantic v2."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SynthesizerType(str, Enum):
    """Available synthesizer backends."""

    AUTO = "auto"
    GAUSSIAN_COPULA = "gaussian_copula"
    CTGAN = "ctgan"
    TVAE = "tvae"
    TABDDPM = "tabddpm"          # Diffusion model (ICML 2023) — best quality
    TABSYN = "tabsyn"            # Latent diffusion (ICLR 2024) — best quality/speed
    GREAT = "great"              # LLM-based (ICLR 2023) — semantic understanding


class DataStrategy(str, Enum):
    """Data-type-specific generation strategies."""

    AUTO = "auto"
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    MIXED = "mixed"
    TIMESERIES = "timeseries"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    enabled: bool = False
    provider: str | None = Field(
        default=None,
        description="LiteLLM provider string, e.g. 'anthropic', 'openai', 'ollama'",
    )
    model: str | None = Field(
        default=None,
        description="Model identifier, e.g. 'claude-sonnet-4-20250514', 'gpt-4o-mini'",
    )
    api_key: str | None = Field(default=None, description="API key (if not in env)")
    api_base: str | None = Field(default=None, description="Custom API base URL")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=16384)
    timeout: int = Field(default=60, ge=5)


class PrivacyConfig(BaseModel):
    """Privacy protection settings."""

    detect_pii: bool = True
    detect_mnpi: bool = False
    replace_pii_with_faker: bool = True
    pii_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    """Evaluation pipeline settings."""

    run_diagnostics: bool = True
    run_statistical_fidelity: bool = True
    run_ml_utility: bool = False
    run_privacy_check: bool = False
    run_llm_validation: bool = False
    tstr_test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    quality_threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class GenerationConfig(BaseModel):
    """Generation behavior settings."""

    batch_size: int = Field(default=50_000, ge=100)
    enforce_min_max: bool = True
    enforce_constraints: bool = True
    max_retries_per_batch: int = Field(default=3, ge=1, le=10)
    seed: int | None = None


class CTGANConfig(BaseModel):
    """CTGAN hyperparameters."""

    epochs: int = Field(default=300, ge=1)
    batch_size: int = Field(default=500, ge=10)
    generator_dim: tuple[int, ...] = (256, 256)
    discriminator_dim: tuple[int, ...] = (256, 256)
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    discriminator_steps: int = 1
    pac: int = 10
    cuda: bool = True


class TVAEConfig(BaseModel):
    """TVAE hyperparameters."""

    epochs: int = Field(default=300, ge=1)
    batch_size: int = Field(default=500, ge=10)
    encoder_dim: tuple[int, ...] = (128, 128)
    decoder_dim: tuple[int, ...] = (128, 128)
    l2scale: float = 1e-5
    loss_factor: int = 2
    cuda: bool = True


class GaussianCopulaConfig(BaseModel):
    """Gaussian Copula hyperparameters."""

    default_distribution: str = "beta"
    numerical_distributions: dict[str, str] | None = None


class TabDDPMConfig(BaseModel):
    """TabDDPM (Diffusion) hyperparameters."""

    n_timesteps: int = Field(default=1000, ge=100, le=5000)
    epochs: int = Field(default=300, ge=1)
    batch_size: int = Field(default=256, ge=16)
    lr: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 512, 256)
    num_numerical_gaussians: int = 25  # GMM components for mode-specific normalization
    scheduler: str = "cosine"  # 'linear' or 'cosine'
    cuda: bool = True


class TabSynConfig(BaseModel):
    """TabSyn (Latent Diffusion) hyperparameters."""

    # VAE stage
    vae_epochs: int = Field(default=200, ge=1)
    vae_lr: float = 1e-3
    vae_latent_dim: int = 64
    vae_encoder_dim: tuple[int, ...] = (128, 128)
    vae_decoder_dim: tuple[int, ...] = (128, 128)
    # Diffusion stage
    diff_epochs: int = Field(default=200, ge=1)
    diff_lr: float = 1e-3
    n_timesteps: int = Field(default=1000, ge=100)
    diff_hidden_dims: tuple[int, ...] = (256, 256)
    batch_size: int = Field(default=256, ge=16)
    cuda: bool = True


class GReaTConfig(BaseModel):
    """GReaT (LLM-based generation) hyperparameters."""

    model_name: str = "distilgpt2"  # HuggingFace model ID
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=1)
    lr: float = 5e-5
    max_length: int = 512
    temperature: float = 0.7
    top_k: int = 50
    cuda: bool = True


class SynthForgeConfig(BaseModel):
    """Master configuration for the SynthForge pipeline."""

    synthesizer: SynthesizerType = SynthesizerType.AUTO
    strategy: DataStrategy = DataStrategy.AUTO
    llm: LLMConfig = Field(default_factory=LLMConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ctgan: CTGANConfig = Field(default_factory=CTGANConfig)
    tvae: TVAEConfig = Field(default_factory=TVAEConfig)
    gaussian_copula: GaussianCopulaConfig = Field(default_factory=GaussianCopulaConfig)
    tabddpm: TabDDPMConfig = Field(default_factory=TabDDPMConfig)
    tabsyn: TabSynConfig = Field(default_factory=TabSynConfig)
    great: GReaTConfig = Field(default_factory=GReaTConfig)
    verbose: bool = True

    def with_llm(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> SynthForgeConfig:
        """Return a copy with LLM enabled."""
        return self.model_copy(
            update={
                "llm": LLMConfig(
                    enabled=True,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    **kwargs,
                )
            }
        )
