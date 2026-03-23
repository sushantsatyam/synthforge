# Changelog

All notable changes to SynthForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-03-23

### Added
- **6 Synthesizer backends**: Gaussian Copula (CPU), CTGAN, TVAE, TabDDPM (ICML 2023), TabSyn (ICLR 2024), GReaT (ICLR 2023)
- **CUDA enforcement**: All neural models require GPU by default with clear error messages and auto-fallback to Gaussian Copula
- **LLM-augmented pipeline**: Schema enrichment, 3-layer PII detection (heuristic + Presidio + LLM), MNPI detection, LLM-as-judge semantic validation
- **LLM-agnostic**: Works with Claude, OpenAI, Ollama, vLLM via LiteLLM
- **5-layer evaluation**: Diagnostics, KS/TV/Correlation/C2ST fidelity, TSTR ML utility, MIA privacy, LLM semantic validation
- **Reversible data transforms**: Numerical (quantile/standard), Categorical (label encoding), Datetime (timestamp), Boolean, all with null handling
- **Constraint-Augmented Generation**: Inequality, PositiveValue, ValueRange, FixedCombinations with type-safe guards
- **Auto-strategy engine**: Selects optimal synthesizer based on data type (numerical/categorical/mixed/timeseries) and hardware (CUDA/CPU)
- **Batch generation**: Configurable batch sizes for generating millions of rows
- **PII auto-replacement**: Detected PII columns replaced with Faker-generated values
- **CLI**: `synthforge input.csv -o output.csv -n 100000 --evaluate`
- **51 tests**, 4 examples (HR, Financial, Sensor, E-commerce)
