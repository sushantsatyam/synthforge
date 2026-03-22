"""PII detection combining rule-based (Presidio) + LLM augmentation.

Layer 1: Column name heuristics (fast, always available)
Layer 2: Presidio NER + regex on sample values (if presidio installed)
Layer 3: LLM analysis for non-obvious PII (if LLM enabled)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from synthforge.llm.client import LLMClient
from synthforge.metadata import Metadata, SemanticType, _PII_SEMANTIC_TYPES, _FAKER_MAP

logger = logging.getLogger(__name__)

_PII_DETECTION_PROMPT = """\
You are a data privacy expert. Analyze these columns for PII (Personally Identifiable Information).

Consider both OBVIOUS PII (names, emails, SSNs) and NON-OBVIOUS PII:
- Customer reference numbers that could be traced back to individuals
- Free-text fields containing embedded PII
- Combinations of columns that together identify individuals (quasi-identifiers)
- IP addresses, device IDs, geolocation coordinates
- Biometric data, health information, financial account numbers

COLUMNS AND SAMPLE VALUES:
{column_samples}

ALREADY DETECTED PII (by rules):
{known_pii}

Return JSON:
{{
  "additional_pii": {{
    "<column_name>": {{
      "pii_type": "<description of PII type>",
      "confidence": <0.0 to 1.0>,
      "reasoning": "<why this is PII>",
      "faker_provider": "<recommended faker method or null>"
    }}
  }},
  "quasi_identifiers": [
    {{
      "columns": ["col_a", "col_b", "col_c"],
      "risk": "<low/medium/high>",
      "reasoning": "<why these together could identify someone>"
    }}
  ],
  "risk_summary": "<overall PII risk assessment>"
}}"""


class PIIDetector:
    """Multi-layer PII detection engine."""

    def __init__(self, client: LLMClient | None = None, confidence_threshold: float = 0.7):
        self._client = client
        self._threshold = confidence_threshold
        self._presidio_available = False
        self._analyzer = None

        try:
            from presidio_analyzer import AnalyzerEngine
            self._analyzer = AnalyzerEngine()
            self._presidio_available = True
            logger.info("Presidio analyzer available for PII detection")
        except ImportError:
            logger.info("Presidio not installed; using heuristic + LLM detection only")

    def detect(self, df: pd.DataFrame, metadata: Metadata) -> Metadata:
        """Run multi-layer PII detection and update metadata.

        Args:
            df: Input DataFrame.
            metadata: Existing metadata (already has heuristic PII flags).

        Returns:
            Updated metadata with enhanced PII detection.
        """
        # Layer 1: Already done in metadata.detect_metadata() via column name patterns

        # Layer 2: Presidio value-based scanning
        if self._presidio_available:
            self._run_presidio(df, metadata)

        # Layer 3: LLM-based detection for non-obvious PII
        if self._client and self._client.enabled:
            self._run_llm_detection(df, metadata)

        pii_count = len(metadata.pii_columns)
        logger.info("PII detection complete: %d PII columns identified", pii_count)
        return metadata

    def _run_presidio(self, df: pd.DataFrame, metadata: Metadata) -> None:
        """Scan sample values with Presidio NER."""
        from presidio_analyzer import AnalyzerEngine

        for col_name, col_meta in metadata.columns.items():
            if col_meta.is_pii:
                continue  # Already flagged

            # Only scan string/object columns
            if not pd.api.types.is_object_dtype(df[col_name].dtype):
                continue

            # Sample up to 50 values
            sample_values = df[col_name].dropna().head(50).astype(str).tolist()
            if not sample_values:
                continue

            # Analyze each value
            pii_hits: dict[str, int] = {}
            for val in sample_values:
                if len(val) < 3:
                    continue
                results = self._analyzer.analyze(text=val, language="en")
                for r in results:
                    if r.score >= self._threshold:
                        pii_hits[r.entity_type] = pii_hits.get(r.entity_type, 0) + 1

            # If >20% of samples have PII, flag the column
            if pii_hits:
                most_common = max(pii_hits, key=pii_hits.get)
                hit_rate = max(pii_hits.values()) / len(sample_values)

                if hit_rate > 0.2:
                    col_meta.is_pii = True
                    col_meta.pii_confidence = min(hit_rate, 0.95)
                    # Map Presidio entity types to our semantic types
                    entity_map = {
                        "PERSON": SemanticType.PERSON_NAME,
                        "EMAIL_ADDRESS": SemanticType.EMAIL,
                        "PHONE_NUMBER": SemanticType.PHONE,
                        "US_SSN": SemanticType.SSN,
                        "CREDIT_CARD": SemanticType.CREDIT_CARD,
                        "IP_ADDRESS": SemanticType.IP_ADDRESS,
                        "LOCATION": SemanticType.ADDRESS,
                    }
                    if most_common in entity_map:
                        col_meta.semantic_type = entity_map[most_common]
                        if col_meta.semantic_type in _FAKER_MAP:
                            col_meta.faker_provider = _FAKER_MAP[col_meta.semantic_type]

                    logger.info(
                        "Presidio detected PII in '%s': %s (%.0f%% hit rate)",
                        col_name, most_common, hit_rate * 100,
                    )

    def _run_llm_detection(self, df: pd.DataFrame, metadata: Metadata) -> None:
        """Use LLM to catch non-obvious PII patterns."""
        # Build column samples
        column_samples = []
        for col_name in df.columns:
            sample_vals = df[col_name].dropna().head(5).astype(str).tolist()
            column_samples.append(
                f"  {col_name} (dtype={df[col_name].dtype}): {sample_vals}"
            )

        known_pii = [
            f"  {col_name}: {metadata.columns[col_name].semantic_type.value}"
            for col_name in metadata.pii_columns
        ] or ["  None detected yet"]

        prompt = _PII_DETECTION_PROMPT.format(
            column_samples="\n".join(column_samples),
            known_pii="\n".join(known_pii),
        )

        try:
            result = self._client.complete_json(prompt=prompt, max_tokens=2048)
            additional = result.get("additional_pii", {})

            for col_name, info in additional.items():
                if col_name in metadata.columns:
                    confidence = info.get("confidence", 0.5)
                    if confidence >= self._threshold:
                        col_meta = metadata.columns[col_name]
                        col_meta.is_pii = True
                        col_meta.pii_confidence = confidence
                        faker_prov = info.get("faker_provider")
                        if faker_prov and faker_prov != "null":
                            col_meta.faker_provider = faker_prov

                        logger.info(
                            "LLM detected PII in '%s': %s (confidence=%.2f)",
                            col_name, info.get("pii_type", "unknown"), confidence,
                        )

            # Store quasi-identifier info
            quasi = result.get("quasi_identifiers", [])
            for qi in quasi:
                metadata.business_rules.append(
                    f"QUASI-IDENTIFIER: {qi.get('columns', [])} — {qi.get('reasoning', '')}"
                )

        except Exception as e:
            logger.warning("LLM PII detection failed: %s", e)
