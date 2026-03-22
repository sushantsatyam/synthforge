"""LLM-powered schema enrichment.

Infers semantic column types, relationships, business rules, and Faker providers
from column names + sample data — dramatically improving on statistical-only detection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from synthforge.llm.client import LLMClient
from synthforge.metadata import ColumnMeta, Metadata, SDType, SemanticType, _FAKER_MAP

logger = logging.getLogger(__name__)

_ENRICHMENT_SYSTEM_PROMPT = """\
You are a data engineering expert analyzing tabular database schemas.
Given column names, data types, and sample values, you infer:
1. Semantic types (what real-world concept each column represents)
2. PII flags (personally identifiable information)
3. MNPI flags (material non-public information for financial data)
4. Cross-column relationships (foreign keys, hierarchical groups, functional dependencies)
5. Business rules (constraints that should hold between columns)
6. Recommended Faker providers for generating realistic replacement values
7. Best synthesizer strategy for this table (categorical / numerical / timeseries / mixed)

Respond ONLY with valid JSON, no other text."""

_ENRICHMENT_PROMPT_TEMPLATE = """\
Analyze this table schema and sample data:

TABLE INFO:
- Row count: {row_count}
- Column count: {col_count}

COLUMNS:
{column_info}

SAMPLE DATA (first 5 rows):
{sample_data}

COLUMN STATISTICS:
{column_stats}

Return JSON with this exact structure:
{{
  "columns": {{
    "<column_name>": {{
      "semantic_type": "<one of: person_name, first_name, last_name, email, phone, ssn, \
address, city, state, zip_code, country, ip_address, credit_card, date_of_birth, url, \
latitude, longitude, currency, percentage, count, identifier, foreign_key, primary_key, \
timestamp, duration, freetext, generic>",
      "is_pii": <true/false>,
      "is_mnpi": <true/false>,
      "pii_confidence": <0.0 to 1.0>,
      "faker_provider": "<faker method name or null>",
      "description": "<brief description of what this column represents>"
    }}
  }},
  "relationships": [
    {{"type": "hierarchy", "columns": ["col_a", "col_b"], "description": "..."}}
  ],
  "business_rules": [
    "rule description in natural language"
  ],
  "recommended_strategy": "categorical|numerical|timeseries|mixed",
  "synthesizer_recommendation": "gaussian_copula|ctgan|tvae",
  "notes": "any additional observations about data quality or patterns"
}}"""


class SchemaEnricher:
    """Uses LLM to enrich metadata beyond statistical detection."""

    def __init__(self, client: LLMClient):
        self._client = client

    def enrich(self, df: pd.DataFrame, metadata: Metadata) -> Metadata:
        """Enrich existing metadata with LLM-inferred semantics.

        Args:
            df: The input DataFrame.
            metadata: Pre-detected metadata (from heuristic detection).

        Returns:
            Updated Metadata with enriched column info.
        """
        if not self._client.enabled:
            logger.warning("LLM not enabled; returning metadata unchanged")
            return metadata

        # Build the prompt
        column_info = self._format_column_info(df, metadata)
        sample_data = self._format_sample_data(df)
        column_stats = self._format_column_stats(df, metadata)

        prompt = _ENRICHMENT_PROMPT_TEMPLATE.format(
            row_count=len(df),
            col_count=len(df.columns),
            column_info=column_info,
            sample_data=sample_data,
            column_stats=column_stats,
        )

        try:
            result = self._client.complete_json(
                prompt=prompt,
                system=_ENRICHMENT_SYSTEM_PROMPT,
                max_tokens=4096,
            )
            return self._apply_enrichment(metadata, result)

        except Exception as e:
            logger.error("Schema enrichment failed: %s", e)
            return metadata

    def _format_column_info(self, df: pd.DataFrame, meta: Metadata) -> str:
        lines = []
        for col_name, col_meta in meta.columns.items():
            lines.append(
                f"  - {col_name}: dtype={df[col_name].dtype}, "
                f"sdtype={col_meta.sdtype.value}, "
                f"cardinality={col_meta.cardinality}, "
                f"null_frac={col_meta.null_fraction:.2%}"
            )
        return "\n".join(lines)

    def _format_sample_data(self, df: pd.DataFrame) -> str:
        sample = df.head(5)
        return sample.to_string(index=False, max_cols=20)

    def _format_column_stats(self, df: pd.DataFrame, meta: Metadata) -> str:
        lines = []
        for col_name in df.columns:
            col_meta = meta.columns.get(col_name)
            if not col_meta:
                continue

            if col_meta.sdtype == SDType.NUMERICAL:
                desc = df[col_name].describe()
                lines.append(
                    f"  - {col_name}: min={desc.get('min', 'N/A')}, "
                    f"max={desc.get('max', 'N/A')}, "
                    f"mean={desc.get('mean', 'N/A'):.2f}, "
                    f"std={desc.get('std', 'N/A'):.2f}"
                )
            elif col_meta.sdtype == SDType.CATEGORICAL:
                top_vals = df[col_name].value_counts().head(5).to_dict()
                lines.append(f"  - {col_name}: top_values={top_vals}")
        return "\n".join(lines)

    def _apply_enrichment(self, metadata: Metadata, result: dict) -> Metadata:
        """Apply LLM enrichment results to metadata."""
        columns_info = result.get("columns", {})

        for col_name, enrichment in columns_info.items():
            if col_name not in metadata.columns:
                continue

            col_meta = metadata.columns[col_name]

            # Update semantic type
            sem_str = enrichment.get("semantic_type", "generic")
            try:
                col_meta.semantic_type = SemanticType(sem_str)
            except ValueError:
                col_meta.semantic_type = SemanticType.GENERIC

            # Update PII/MNPI flags
            col_meta.is_pii = enrichment.get("is_pii", col_meta.is_pii)
            col_meta.is_mnpi = enrichment.get("is_mnpi", col_meta.is_mnpi)
            col_meta.pii_confidence = enrichment.get("pii_confidence", col_meta.pii_confidence)

            # Update Faker provider
            faker_prov = enrichment.get("faker_provider")
            if faker_prov and faker_prov != "null":
                col_meta.faker_provider = faker_prov
            elif col_meta.semantic_type in _FAKER_MAP:
                col_meta.faker_provider = _FAKER_MAP[col_meta.semantic_type]

            # Store description
            desc = enrichment.get("description", "")
            if desc:
                col_meta.extra["description"] = desc

        # Update relationships and business rules
        metadata.relationships = result.get("relationships", metadata.relationships)
        metadata.business_rules = result.get("business_rules", metadata.business_rules)

        # Update strategy recommendation
        rec_strategy = result.get("recommended_strategy", metadata.data_strategy)
        if rec_strategy in ("categorical", "numerical", "timeseries", "mixed"):
            metadata.data_strategy = rec_strategy

        logger.info(
            "Schema enrichment applied: %d columns enriched, %d relationships, %d rules",
            len(columns_info),
            len(metadata.relationships),
            len(metadata.business_rules),
        )
        return metadata
