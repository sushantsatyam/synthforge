"""LLM-as-judge semantic validation for synthetic data.

Catches logical impossibilities that statistical tests miss:
- A 5-year-old with a PhD
- Japanese names with Mexican zip codes
- Shipping dates before order dates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from synthforge.llm.client import LLMClient
from synthforge.metadata import Metadata

logger = logging.getLogger(__name__)

_VALIDATION_SYSTEM = """\
You are a data quality expert reviewing synthetic tabular data rows for logical
impossibilities, semantic inconsistencies, and suspicious patterns.
Focus on impossible combinations, temporal violations, business rule violations,
statistical anomalies, and cross-column incoherence."""

_VALIDATION_PROMPT = """\
Review these {n_rows} rows of synthetic data for semantic validity.

COLUMN DESCRIPTIONS:
{column_descriptions}

KNOWN BUSINESS RULES:
{business_rules}

DATA ROWS (JSON):
{data_json}

Return JSON:
{{
  "issues": [
    {{
      "row_indices": [<0-based row indices>],
      "columns": ["<affected columns>"],
      "severity": "low|medium|high|critical",
      "issue_type": "<impossible_combination|temporal_violation|business_rule|statistical_anomaly|cross_column_incoherence>",
      "description": "<what's wrong>"
    }}
  ],
  "overall_quality": "excellent|good|acceptable|poor",
  "valid_row_fraction": <0.0 to 1.0>,
  "summary": "<brief assessment>"
}}"""


@dataclass
class ValidationIssue:
    row_indices: list[int]
    columns: list[str]
    severity: str
    issue_type: str
    description: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    overall_quality: str = "unknown"
    valid_row_fraction: float = 1.0
    summary: str = ""
    batches_checked: int = 0
    total_rows_checked: int = 0

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_quality": self.overall_quality,
            "valid_row_fraction": self.valid_row_fraction,
            "summary": self.summary,
            "total_issues": self.issue_count,
            "batches_checked": self.batches_checked,
            "total_rows_checked": self.total_rows_checked,
            "issues": [
                {
                    "row_indices": i.row_indices,
                    "columns": i.columns,
                    "severity": i.severity,
                    "issue_type": i.issue_type,
                    "description": i.description,
                }
                for i in self.issues
            ],
        }


class SemanticValidator:
    """Validate synthetic data using LLM-as-judge pattern."""

    def __init__(self, client: LLMClient, batch_size: int = 30, max_batches: int = 5):
        self._client = client
        self._batch_size = batch_size
        self._max_batches = max_batches

    def validate(self, synthetic_df: pd.DataFrame, metadata: Metadata) -> ValidationReport:
        if not self._client.enabled:
            return ValidationReport(summary="LLM not enabled")

        report = ValidationReport()
        n_rows = len(synthetic_df)
        n_batches = min(self._max_batches, max(1, n_rows // self._batch_size))

        for batch_idx in range(n_batches):
            sample = synthetic_df.sample(n=min(self._batch_size, n_rows), random_state=batch_idx)
            batch_result = self._validate_batch(sample.reset_index(drop=True), metadata)
            report.issues.extend(batch_result.get("issues_parsed", []))
            report.batches_checked += 1
            report.total_rows_checked += len(sample)
            report.overall_quality = batch_result.get("overall_quality", report.overall_quality)
            report.valid_row_fraction = batch_result.get("valid_row_fraction", report.valid_row_fraction)

        sev = [i.severity for i in report.issues]
        report.summary = (
            f"{len(report.issues)} issues in {report.total_rows_checked} rows. "
            f"Critical: {sev.count('critical')}, High: {sev.count('high')}"
        ) if report.issues else f"No issues in {report.total_rows_checked} rows."

        logger.info("Semantic validation: %s", report.summary)
        return report

    def _validate_batch(self, batch_df: pd.DataFrame, metadata: Metadata) -> dict[str, Any]:
        col_descs = [
            f"  {n}: {m.extra.get('description', m.semantic_type.value)} ({m.sdtype.value})"
            for n, m in metadata.columns.items()
        ]
        rules = metadata.business_rules or ["None"]
        data_json = batch_df.to_json(orient="records", indent=1)

        prompt = _VALIDATION_PROMPT.format(
            n_rows=len(batch_df),
            column_descriptions="\n".join(col_descs),
            business_rules="\n".join(f"  - {r}" for r in rules),
            data_json=data_json,
        )

        try:
            result = self._client.complete_json(prompt=prompt, system=_VALIDATION_SYSTEM, max_tokens=3000)
            issues = [
                ValidationIssue(
                    row_indices=i.get("row_indices", []),
                    columns=i.get("columns", []),
                    severity=i.get("severity", "low"),
                    issue_type=i.get("issue_type", "unknown"),
                    description=i.get("description", ""),
                )
                for i in result.get("issues", [])
            ]
            return {
                "issues_parsed": issues,
                "overall_quality": result.get("overall_quality", "unknown"),
                "valid_row_fraction": result.get("valid_row_fraction", 1.0),
            }
        except Exception as e:
            logger.warning("Batch validation failed: %s", e)
            return {"issues_parsed": []}
