"""SynthForge: Main orchestrator for the synthetic data pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from synthforge.config import DataStrategy, LLMConfig, SynthesizerType, SynthForgeConfig
from synthforge.constraints import ConstraintPipeline
from synthforge.evaluation import EvaluationReport, Evaluator
from synthforge.metadata import Metadata, SDType, detect_metadata
from synthforge.strategies import StrategyEngine
from synthforge.synthesizers import BaseSynthesizer
from synthforge.transforms import TransformPipeline

logger = logging.getLogger(__name__)


class SynthForge:
    """Main entry point for synthetic data generation."""

    def __init__(
        self,
        config: SynthForgeConfig | None = None,
        *,
        synthesizer: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        verbose: bool = True,
    ):
        self._config = config or SynthForgeConfig()
        if synthesizer:
            self._config.synthesizer = SynthesizerType(synthesizer)
        if llm_provider and llm_model:
            self._config = self._config.with_llm(llm_provider, llm_model, api_key=llm_api_key)
        self._config.verbose = verbose

        level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(level=level, format="%(name)s | %(levelname)s | %(message)s")

        self._metadata: Metadata | None = None
        self._transform_pipeline: TransformPipeline | None = None
        self._constraint_pipeline = ConstraintPipeline()
        self._synthesizer: BaseSynthesizer | None = None
        self._strategy: DataStrategy | None = None
        self._synth_type: SynthesizerType | None = None
        self._faker = Faker()
        self._fitted = False
        self._original_df: pd.DataFrame | None = None
        self._llm_client = None

    @property
    def metadata(self) -> Metadata | None:
        return self._metadata

    @property
    def config(self) -> SynthForgeConfig:
        return self._config

    def _get_llm_client(self):
        if self._llm_client is None and self._config.llm.enabled:
            from synthforge.llm.client import LLMClient
            self._llm_client = LLMClient(self._config.llm)
        return self._llm_client

    # ── Step 1: PROFILE ────────────────────────────────────────────

    def profile(self, df: pd.DataFrame, primary_key: str | None = None) -> Metadata:
        """Detect metadata, enrich with LLM, detect PII/MNPI."""
        logger.info("PROFILE: %d rows x %d cols", len(df), len(df.columns))
        t0 = time.time()

        self._metadata = detect_metadata(df, primary_key=primary_key)

        if self._config.llm.enabled:
            from synthforge.llm.schema_enricher import SchemaEnricher
            self._metadata = SchemaEnricher(self._get_llm_client()).enrich(df, self._metadata)

        if self._config.privacy.detect_pii:
            from synthforge.llm.pii_detector import PIIDetector
            client = self._get_llm_client() if self._config.llm.enabled else None
            self._metadata = PIIDetector(
                client=client, confidence_threshold=self._config.privacy.pii_confidence_threshold
            ).detect(df, self._metadata)

        if self._config.privacy.detect_mnpi and self._config.llm.enabled:
            from synthforge.llm.mnpi_detector import MNPIDetector
            self._metadata = MNPIDetector(self._get_llm_client()).detect(df, self._metadata)

        logger.info("Profile done (%.1fs): strategy=%s, PII=%d, MNPI=%d",
                     time.time() - t0, self._metadata.data_strategy,
                     len(self._metadata.pii_columns), len(self._metadata.mnpi_columns))
        return self._metadata

    # ── Step 2: FIT ────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, primary_key: str | None = None) -> SynthForge:
        """Fit the synthesizer on input data."""
        logger.info("FIT: Training synthesizer")
        t0 = time.time()
        self._original_df = df.copy()

        if self._metadata is None:
            self.profile(df, primary_key=primary_key)

        engine = StrategyEngine(self._config)
        self._strategy, self._synth_type, self._synthesizer = engine.resolve(self._metadata)

        self._transform_pipeline = TransformPipeline()
        self._transform_pipeline.fit(df, self._metadata)
        transformed = self._transform_pipeline.transform(df)

        if transformed.empty or transformed.shape[1] == 0:
            raise ValueError("No transformable columns found.")

        data = np.nan_to_num(transformed.values.astype(np.float64), nan=0.0)
        self._synthesizer.fit(data, column_names=transformed.columns.tolist())
        self._fitted = True

        logger.info("Fit done (%.1fs): %s/%s, %d->%d features",
                     time.time() - t0, self._synth_type.value, self._strategy.value,
                     len(df.columns), transformed.shape[1])
        return self

    # ── Step 3: GENERATE ───────────────────────────────────────────

    def generate(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data in batches."""
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        logger.info("GENERATE: %d rows", num_rows)
        t0 = time.time()
        batch_size = self._config.generation.batch_size
        batches, remaining = [], num_rows

        with tqdm(total=num_rows, desc="Generating", disable=not self._config.verbose) as pbar:
            while remaining > 0:
                chunk_size = min(remaining, batch_size)
                raw = self._synthesizer.sample(chunk_size)
                raw_df = pd.DataFrame(raw, columns=self._transform_pipeline.transformed_columns)
                chunk = self._transform_pipeline.inverse_transform(raw_df)
                if self._constraint_pipeline.constraint_count > 0:
                    chunk = self._constraint_pipeline.reverse_transform(chunk)
                batches.append(chunk)
                remaining -= chunk_size
                pbar.update(chunk_size)

        result = pd.concat(batches, ignore_index=True).head(num_rows)
        result = self._post_process(result)
        logger.info("Generated %d rows (%.1fs)", len(result), time.time() - t0)
        return result

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._metadata is None:
            return df

        # Add ID columns
        for col in self._metadata.id_columns:
            if col not in df.columns:
                df[col] = range(1, len(df) + 1)

        # Replace PII with Faker
        if self._config.privacy.replace_pii_with_faker:
            for col in self._metadata.pii_columns:
                cm = self._metadata.columns[col]
                if cm.faker_provider:
                    fn = getattr(self._faker, cm.faker_provider, None)
                    if fn:
                        df[col] = [fn() for _ in range(len(df))]

        # Enforce min/max
        if self._config.generation.enforce_min_max and self._original_df is not None:
            for col in self._metadata.numerical_columns:
                if col in df.columns and col in self._original_df.columns:
                    orig = self._original_df[col].dropna()
                    if len(orig) > 0:
                        df[col] = df[col].clip(lower=float(orig.min()), upper=float(orig.max()))

        # Reorder columns
        if self._original_df is not None:
            oc = [c for c in self._original_df.columns if c in df.columns]
            ec = [c for c in df.columns if c not in self._original_df.columns]
            df = df[oc + ec]

        return df

    # ── Step 4: EVALUATE ───────────────────────────────────────────

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> EvaluationReport:
        """Evaluate synthetic data quality."""
        if self._metadata is None:
            self._metadata = detect_metadata(real_df)
        evaluator = Evaluator(self._config.evaluation.quality_threshold)
        report = evaluator.evaluate(
            real_df, synthetic_df, self._metadata,
            run_diagnostics=self._config.evaluation.run_diagnostics,
            run_fidelity=self._config.evaluation.run_statistical_fidelity,
            run_ml_utility=self._config.evaluation.run_ml_utility,
            run_privacy=self._config.evaluation.run_privacy_check,
        )
        logger.info("Evaluation:\n%s", report.summary())
        return report

    # ── Step 5: VALIDATE (LLM semantic) ────────────────────────────

    def validate_semantics(self, synthetic_df: pd.DataFrame):
        if not self._config.llm.enabled:
            logger.warning("LLM not enabled; skipping semantic validation")
            return None
        from synthforge.llm.validator import SemanticValidator
        return SemanticValidator(self._get_llm_client()).validate(synthetic_df, self._metadata)

    # ── Convenience ────────────────────────────────────────────────

    def fit_generate(self, df: pd.DataFrame, num_rows: int | None = None,
                     primary_key: str | None = None) -> pd.DataFrame:
        """Profile + fit + generate in one call."""
        self.fit(df, primary_key=primary_key)
        return self.generate(num_rows or len(df))

    def add_constraint(self, constraint) -> SynthForge:
        self._constraint_pipeline.add(constraint)
        return self
