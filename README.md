# SynthForge

> Next-generation synthetic data generation with LLM-augmented pipelines.

SynthForge combines **statistical generative models** (Gaussian Copula, CTGAN, TVAE, Diffusion) with **LLM-powered intelligence** (schema enrichment, PII/MNPI detection, semantic validation) to produce high-fidelity synthetic tabular data from small production samples.

## Quick Start

```python
import pandas as pd
from synthforge import SynthForge

# Load a sample from production (e.g., 2500 rows from Redshift)
df = pd.read_csv("production_sample.csv")

# One-line generation
forge = SynthForge()
synthetic_df = forge.fit_generate(df, num_rows=100_000)

# With LLM enrichment (auto-detects PII, infers semantics)
forge = SynthForge(llm_provider="anthropic", llm_model="claude-sonnet-4-20250514")
forge.profile(df)                    # Schema enrichment + PII detection
forge.fit(df)                        # Train synthesizer
synthetic_df = forge.generate(100_000)  # Bulk generate
report = forge.evaluate(df, synthetic_df)  # Quality report
```

## Key Features

- **Intelligent Schema Detection**: LLM-powered column semantic inference beyond statistical type detection
- **PII/MNPI Detection**: Presidio + LLM augmentation for catching non-obvious sensitive data
- **Multiple Synthesizers**: Gaussian Copula (fast), CTGAN/TVAE (balanced), TabSyn (highest quality)
- **Data-Type Strategies**: Specialized pipelines for categorical, numerical, time-series, and mixed-type tables
- **Evaluation-First**: Built-in quality reports with statistical fidelity, ML utility, and privacy metrics
- **Configurable Scale**: From 1K to 10M+ rows with batch generation and optional GPU acceleration
- **LLM-Agnostic**: Works with Claude, OpenAI, Ollama, vLLM, or any LiteLLM-supported provider

## Installation

```bash
pip install synthforge                  # Core (Gaussian Copula only)
pip install "synthforge[gan]"           # + CTGAN/TVAE
pip install "synthforge[llm]"           # + LLM enrichment
pip install "synthforge[evaluation]"    # + quality reports
pip install "synthforge[all]"           # Everything
```

## Architecture

```
Production Sample (DataFrame/CSV)
        │
        ▼
┌─────────────────────────────────┐
│  1. PROFILE (LLM-augmented)     │
│  • Auto-detect metadata         │
│  • Semantic column inference     │
│  • PII / MNPI detection         │
│  • Business rule extraction     │
│  • Synthesizer recommendation   │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│  2. FIT (Statistical/Neural)    │
│  • Reversible data transforms   │
│  • Constraint-aware training    │
│  • Auto-select or user-pick:    │
│    GaussianCopula / CTGAN /     │
│    TVAE / TabSyn / Diffusion    │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│  3. GENERATE (Batch)            │
│  • Configurable row count       │
│  • Batch chunking for scale     │
│  • Constraint enforcement       │
│  • PII replacement (Faker)      │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│  4. EVALUATE (5-layer pipeline) │
│  • Diagnostic checks            │
│  • Statistical fidelity         │
│  • ML utility (TSTR)            │
│  • Privacy (MIA, Anonymeter)    │
│  • LLM semantic validation      │
└─────────────────────────────────┘
```

## License

Proprietary. All rights reserved.
