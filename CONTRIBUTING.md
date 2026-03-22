# Contributing to SynthForge

## Development Setup

```bash
git clone https://github.com/yourname/synthforge.git
cd synthforge
pip install -e ".[dev]"
```

## Architecture Overview

```
SynthForge Pipeline:  Profile → Fit → Generate → Evaluate → (Validate)

synthforge/
├── forge.py              ← Orchestrator (public API: SynthForge class)
├── config.py             ← Pydantic v2 configuration models
├── metadata.py           ← Schema detection + semantic types
├── transforms/           ← Reversible data transforms (RDT equivalent)
├── synthesizers/         ← Generation backends (plugin registry pattern)
│   ├── gaussian_copula   ← Default: fast, CPU-only, no deps
│   ├── ctgan             ← GAN-based (requires torch)
│   └── tvae              ← VAE-based (requires torch)
├── constraints/          ← CAG: constraint-augmented generation
├── strategies/           ← Auto-selects synthesizer by data characteristics
├── llm/                  ← LLM-augmented pipeline (via LiteLLM)
│   ├── schema_enricher   ← Infer column semantics + relationships
│   ├── pii_detector      ← 3-layer PII: heuristic + Presidio + LLM
│   ├── mnpi_detector     ← Financial MNPI detection
│   └── validator         ← LLM-as-judge semantic validation
└── evaluation/           ← 5-layer quality assessment
```

## Adding a New Synthesizer

1. Create `synthforge/synthesizers/your_synth.py`
2. Inherit from `BaseSynthesizer`
3. Implement `fit(data, column_names)` and `sample(n_rows)`
4. Decorate with `@register_synthesizer("your_synth")`
5. Import in `synthesizers/__init__.py`
6. Add config model in `config.py`
7. Add creation logic in `strategies/__init__.py`

```python
from synthforge.synthesizers import BaseSynthesizer, register_synthesizer

@register_synthesizer("my_model")
class MySynthesizer(BaseSynthesizer):
    def fit(self, data, column_names=None):
        # Train on numpy array (n_samples, n_features)
        self._fitted = True

    def sample(self, n_rows):
        # Return numpy array (n_rows, n_features)
        return generated_data
```

## Adding a New Constraint

1. Inherit from `BaseConstraint` in `constraints/__init__.py`
2. Implement: `is_valid(df)`, `transform(df)`, `reverse_transform(df)`, `columns`
3. Always add type guards for column existence and numeric dtype checks

## Running Tests

```bash
make test          # Run all tests
make test-cov      # With coverage report
make lint          # Ruff + mypy
make format        # Auto-format
```

## Code Style

- Python 3.10+ with `from __future__ import annotations`
- Pydantic v2 for configuration
- sklearn-style API: `fit()` / `sample()` / `fit_generate()`
- Type hints everywhere
- Logging via `logging.getLogger(__name__)`
- Tests via pytest with fixtures
