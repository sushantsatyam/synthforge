"""MNPI (Material Non-Public Information) detector for financial/corporate data.

Detects columns containing information that could constitute insider trading
risk or regulatory violations if exposed.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from synthforge.llm.client import LLMClient
from synthforge.metadata import Metadata

logger = logging.getLogger(__name__)

_MNPI_PROMPT = """\
You are a compliance expert specializing in SEC regulations and insider trading.
Analyze these columns for MNPI (Material Non-Public Information) risk.

MNPI categories to check:
1. PRE-RELEASE FINANCIALS: Revenue, earnings, margins, forecasts before public filing
2. M&A INFORMATION: Deal values, target companies, acquisition plans
3. STRATEGIC PLANS: Product launches, market entries, restructuring before announcement
4. PERSONNEL CHANGES: C-suite changes, board appointments before disclosure
5. LEGAL/REGULATORY: Pending lawsuits, regulatory actions, investigations
6. CLIENT/DEAL PIPELINE: Active deals, customer data tied to non-public transactions

COLUMNS AND SAMPLE VALUES:
{column_samples}

TABLE CONTEXT:
- Row count: {row_count}
- Appears to be: {data_context}

Return JSON:
{{
  "mnpi_columns": {{
    "<column_name>": {{
      "mnpi_category": "<category from above>",
      "risk_level": "low|medium|high|critical",
      "confidence": <0.0 to 1.0>,
      "reasoning": "<why this could be MNPI>",
      "recommendation": "<how to handle: mask/redact/aggregate/exclude>"
    }}
  }},
  "cross_column_risks": [
    {{
      "columns": ["col_a", "col_b"],
      "risk": "<description of combined MNPI risk>"
    }}
  ],
  "overall_mnpi_risk": "none|low|medium|high|critical",
  "regulatory_notes": "<relevant SEC/regulatory considerations>"
}}"""


class MNPIDetector:
    """Detect material non-public information in financial/corporate data."""

    def __init__(self, client: LLMClient):
        self._client = client

    def detect(self, df: pd.DataFrame, metadata: Metadata) -> Metadata:
        """Scan for MNPI and update metadata.

        Args:
            df: Input DataFrame.
            metadata: Existing metadata.

        Returns:
            Updated metadata with MNPI flags.
        """
        if not self._client.enabled:
            logger.warning("LLM not enabled; skipping MNPI detection")
            return metadata

        column_samples = []
        for col_name in df.columns:
            sample_vals = df[col_name].dropna().head(5).astype(str).tolist()
            column_samples.append(f"  {col_name}: {sample_vals}")

        # Determine data context from column names
        all_cols = " ".join(df.columns).lower()
        if any(w in all_cols for w in ("revenue", "earnings", "profit", "ebitda", "margin")):
            context = "financial/earnings data"
        elif any(w in all_cols for w in ("deal", "acquisition", "merger", "target")):
            context = "M&A/deal pipeline data"
        elif any(w in all_cols for w in ("trade", "order", "position", "portfolio")):
            context = "trading/investment data"
        else:
            context = "general business data"

        prompt = _MNPI_PROMPT.format(
            column_samples="\n".join(column_samples),
            row_count=len(df),
            data_context=context,
        )

        try:
            result = self._client.complete_json(prompt=prompt, max_tokens=2048)
            mnpi_cols = result.get("mnpi_columns", {})

            for col_name, info in mnpi_cols.items():
                if col_name in metadata.columns:
                    risk = info.get("risk_level", "low")
                    confidence = info.get("confidence", 0.5)

                    if risk in ("medium", "high", "critical") and confidence >= 0.6:
                        metadata.columns[col_name].is_mnpi = True
                        metadata.columns[col_name].extra["mnpi_category"] = info.get(
                            "mnpi_category", ""
                        )
                        metadata.columns[col_name].extra["mnpi_risk"] = risk
                        metadata.columns[col_name].extra["mnpi_recommendation"] = info.get(
                            "recommendation", ""
                        )
                        logger.info(
                            "MNPI detected in '%s': %s (risk=%s, confidence=%.2f)",
                            col_name,
                            info.get("mnpi_category", "unknown"),
                            risk,
                            confidence,
                        )

            # Store cross-column risks
            cross_risks = result.get("cross_column_risks", [])
            for cr in cross_risks:
                metadata.business_rules.append(
                    f"MNPI-RISK: {cr.get('columns', [])} — {cr.get('risk', '')}"
                )

            overall = result.get("overall_mnpi_risk", "none")
            logger.info(
                "MNPI detection complete: %d columns flagged, overall risk=%s",
                len(metadata.mnpi_columns),
                overall,
            )

        except Exception as e:
            logger.warning("MNPI detection failed: %s", e)

        return metadata
