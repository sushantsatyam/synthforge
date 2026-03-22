"""Enhanced metadata detection and representation.

Goes beyond SDV's statistical detection by classifying columns into semantic types
and detecting PII/MNPI flags, relationships, and business rules.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SDType(str, Enum):
    """Synthetic data types — the core column classification."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    ID = "id"
    TEXT = "text"
    UNKNOWN = "unknown"


class SemanticType(str, Enum):
    """Semantic column types inferred by LLM or heuristics."""

    # PII types
    PERSON_NAME = "person_name"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    ADDRESS = "address"
    CITY = "city"
    STATE = "state"
    ZIP_CODE = "zip_code"
    COUNTRY = "country"
    IP_ADDRESS = "ip_address"
    CREDIT_CARD = "credit_card"
    DATE_OF_BIRTH = "date_of_birth"

    # Business types
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    COUNT = "count"
    IDENTIFIER = "identifier"
    FOREIGN_KEY = "foreign_key"
    PRIMARY_KEY = "primary_key"
    TIMESTAMP = "timestamp"
    DURATION = "duration"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    URL = "url"

    # Generic
    FREETEXT = "freetext"
    GENERIC = "generic"


@dataclass
class ColumnMeta:
    """Metadata for a single column."""

    name: str
    sdtype: SDType
    semantic_type: SemanticType = SemanticType.GENERIC
    is_pii: bool = False
    is_mnpi: bool = False
    pii_confidence: float = 0.0
    nullable: bool = False
    null_fraction: float = 0.0
    cardinality: int = 0
    cardinality_ratio: float = 0.0  # cardinality / nrows
    faker_provider: str | None = None  # e.g. "faker.name", "faker.email"
    distribution: str | None = None  # detected best-fit distribution
    constraints: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Metadata:
    """Full table metadata including column info, relationships, and rules."""

    columns: dict[str, ColumnMeta] = field(default_factory=dict)
    primary_key: str | None = None
    row_count: int = 0
    relationships: list[dict[str, str]] = field(default_factory=list)
    business_rules: list[str] = field(default_factory=list)
    data_strategy: str = "auto"  # categorical / numerical / mixed / timeseries

    @property
    def column_names(self) -> list[str]:
        return list(self.columns.keys())

    @property
    def numerical_columns(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.sdtype == SDType.NUMERICAL]

    @property
    def categorical_columns(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.sdtype == SDType.CATEGORICAL]

    @property
    def datetime_columns(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.sdtype == SDType.DATETIME]

    @property
    def boolean_columns(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.sdtype == SDType.BOOLEAN]

    @property
    def id_columns(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.sdtype == SDType.ID]

    @property
    def pii_columns(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.is_pii]

    @property
    def mnpi_columns(self) -> list[str]:
        return [c for c, m in self.columns.items() if m.is_mnpi]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "primary_key": self.primary_key,
            "row_count": self.row_count,
            "data_strategy": self.data_strategy,
            "columns": {
                name: {
                    "sdtype": col.sdtype.value,
                    "semantic_type": col.semantic_type.value,
                    "is_pii": col.is_pii,
                    "is_mnpi": col.is_mnpi,
                    "nullable": col.nullable,
                    "null_fraction": round(col.null_fraction, 4),
                    "cardinality": col.cardinality,
                    "faker_provider": col.faker_provider,
                }
                for name, col in self.columns.items()
            },
            "relationships": self.relationships,
            "business_rules": self.business_rules,
        }


# ──────────────────────────────────────────────────────────────────
# Heuristic-based detection (fast, no LLM needed)
# ──────────────────────────────────────────────────────────────────

# Regex patterns for PII column name detection
_PII_NAME_PATTERNS: dict[SemanticType, list[re.Pattern]] = {
    SemanticType.EMAIL: [re.compile(r"e[-_]?mail", re.I)],
    SemanticType.PHONE: [re.compile(r"phone|mobile|cell|tel(ephone)?", re.I)],
    SemanticType.SSN: [re.compile(r"ssn|social.?sec", re.I)],
    SemanticType.FIRST_NAME: [re.compile(r"(first|f)[-_]?name|fname|given.?name", re.I)],
    SemanticType.LAST_NAME: [re.compile(r"(last|l)[-_]?name|lname|sur.?name|family.?name", re.I)],
    SemanticType.PERSON_NAME: [re.compile(r"^(full)?[-_]?name$", re.I)],
    SemanticType.ADDRESS: [re.compile(r"address|street|addr", re.I)],
    SemanticType.CITY: [re.compile(r"^city$", re.I)],
    SemanticType.STATE: [re.compile(r"^state$|^province$", re.I)],
    SemanticType.ZIP_CODE: [re.compile(r"zip|postal", re.I)],
    SemanticType.COUNTRY: [re.compile(r"^country$|^nation$", re.I)],
    SemanticType.IP_ADDRESS: [re.compile(r"ip[-_]?(addr|address)?$", re.I)],
    SemanticType.CREDIT_CARD: [re.compile(r"credit.?card|card.?num|cc.?num", re.I)],
    SemanticType.DATE_OF_BIRTH: [re.compile(r"(date.?of.?)?birth|dob|birthday", re.I)],
    SemanticType.URL: [re.compile(r"url|website|link|href", re.I)],
    SemanticType.LATITUDE: [re.compile(r"lat(itude)?$", re.I)],
    SemanticType.LONGITUDE: [re.compile(r"lon(g|gitude)?$", re.I)],
}

_PII_SEMANTIC_TYPES = {
    SemanticType.EMAIL,
    SemanticType.PHONE,
    SemanticType.SSN,
    SemanticType.FIRST_NAME,
    SemanticType.LAST_NAME,
    SemanticType.PERSON_NAME,
    SemanticType.ADDRESS,
    SemanticType.CITY,
    SemanticType.STATE,
    SemanticType.ZIP_CODE,
    SemanticType.IP_ADDRESS,
    SemanticType.CREDIT_CARD,
    SemanticType.DATE_OF_BIRTH,
}

_FAKER_MAP: dict[SemanticType, str] = {
    SemanticType.EMAIL: "email",
    SemanticType.PHONE: "phone_number",
    SemanticType.SSN: "ssn",
    SemanticType.FIRST_NAME: "first_name",
    SemanticType.LAST_NAME: "last_name",
    SemanticType.PERSON_NAME: "name",
    SemanticType.ADDRESS: "street_address",
    SemanticType.CITY: "city",
    SemanticType.STATE: "state",
    SemanticType.ZIP_CODE: "zipcode",
    SemanticType.COUNTRY: "country",
    SemanticType.IP_ADDRESS: "ipv4",
    SemanticType.CREDIT_CARD: "credit_card_number",
    SemanticType.DATE_OF_BIRTH: "date_of_birth",
    SemanticType.URL: "url",
    SemanticType.LATITUDE: "latitude",
    SemanticType.LONGITUDE: "longitude",
}


def _infer_semantic_type_from_name(col_name: str) -> SemanticType | None:
    """Match column name against known PII/semantic patterns."""
    for sem_type, patterns in _PII_NAME_PATTERNS.items():
        for pat in patterns:
            if pat.search(col_name):
                return sem_type
    return None


def _detect_sdtype(series: pd.Series) -> SDType:
    """Detect the base synthetic data type from a pandas Series."""
    if series.empty:
        return SDType.UNKNOWN

    dtype = series.dtype

    # Boolean
    if pd.api.types.is_bool_dtype(dtype):
        return SDType.BOOLEAN

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return SDType.DATETIME

    # Numeric
    if pd.api.types.is_numeric_dtype(dtype):
        nunique = series.nunique()
        n = len(series.dropna())
        # If integer with all-unique values, likely an ID
        if pd.api.types.is_integer_dtype(dtype) and n > 0 and nunique == n and n > 5:
            return SDType.ID
        # If integer with very low cardinality ratio, treat as categorical
        if pd.api.types.is_integer_dtype(dtype) and n > 0 and nunique / n < 0.02 and nunique <= 20:
            return SDType.CATEGORICAL
        return SDType.NUMERICAL

    # Object / string — further analysis needed
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        sample = series.dropna()
        if sample.empty:
            return SDType.UNKNOWN

        # Try datetime parse
        try:
            pd.to_datetime(sample.head(50), format="mixed")
            return SDType.DATETIME
        except (ValueError, TypeError):
            pass

        # Check if boolean-like
        unique_lower = set(sample.astype(str).str.lower().unique())
        if unique_lower <= {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}:
            return SDType.BOOLEAN

        # Check cardinality for ID detection
        nunique = sample.nunique()
        if nunique == len(sample) and len(sample) > 10:
            return SDType.ID

        # High cardinality text
        if nunique / len(sample) > 0.8 and sample.astype(str).str.len().mean() > 30:
            return SDType.TEXT

        return SDType.CATEGORICAL

    return SDType.UNKNOWN


def _detect_data_strategy(meta: Metadata) -> str:
    """Determine the dominant data strategy."""
    n_num = len(meta.numerical_columns)
    n_cat = len(meta.categorical_columns)
    n_dt = len(meta.datetime_columns)
    total = n_num + n_cat + n_dt + len(meta.boolean_columns)

    if total == 0:
        return "mixed"

    # Time-series heuristic: has datetime + sequential numeric
    if n_dt >= 1 and n_num >= 1:
        return "timeseries"

    num_ratio = n_num / total
    cat_ratio = n_cat / total

    if num_ratio > 0.7:
        return "numerical"
    if cat_ratio > 0.7:
        return "categorical"
    return "mixed"


def detect_metadata(
    df: pd.DataFrame,
    primary_key: str | None = None,
) -> Metadata:
    """Auto-detect metadata from a DataFrame using heuristics.

    This is the fast, no-LLM path. For richer detection, use the LLM enricher.

    Args:
        df: Input DataFrame.
        primary_key: Explicit primary key column name.

    Returns:
        Metadata object with detected types, PII flags, and strategy.
    """
    meta = Metadata(row_count=len(df))
    nrows = len(df)

    for col_name in df.columns:
        series = df[col_name]
        sdtype = _detect_sdtype(series)
        sem_type = _infer_semantic_type_from_name(col_name)

        # Override sdtype for known PII types
        if sem_type == SemanticType.DATE_OF_BIRTH and sdtype != SDType.DATETIME:
            sdtype = SDType.DATETIME
        if sem_type in (SemanticType.LATITUDE, SemanticType.LONGITUDE):
            sdtype = SDType.NUMERICAL
        # PII string columns should not be classified as ID even if all unique
        if sem_type in _PII_SEMANTIC_TYPES and sdtype == SDType.ID:
            sdtype = SDType.CATEGORICAL

        is_pii = sem_type in _PII_SEMANTIC_TYPES if sem_type else False
        faker_provider = _FAKER_MAP.get(sem_type) if sem_type else None

        null_count = series.isna().sum()
        nunique = series.nunique()

        col_meta = ColumnMeta(
            name=col_name,
            sdtype=sdtype,
            semantic_type=sem_type or SemanticType.GENERIC,
            is_pii=is_pii,
            pii_confidence=0.8 if is_pii else 0.0,
            nullable=null_count > 0,
            null_fraction=null_count / nrows if nrows > 0 else 0.0,
            cardinality=nunique,
            cardinality_ratio=nunique / nrows if nrows > 0 else 0.0,
            faker_provider=faker_provider,
        )

        # ID detection for primary key candidate — skip PII columns
        if sdtype == SDType.ID and primary_key is None and nunique == nrows and not is_pii:
            if meta.primary_key is None:
                meta.primary_key = col_name

        meta.columns[col_name] = col_meta

    if primary_key:
        meta.primary_key = primary_key
        if primary_key in meta.columns:
            meta.columns[primary_key].sdtype = SDType.ID

    meta.data_strategy = _detect_data_strategy(meta)
    logger.info(
        "Detected metadata: %d columns (%d num, %d cat, %d dt, %d bool), "
        "strategy=%s, PII=%d columns",
        len(meta.columns),
        len(meta.numerical_columns),
        len(meta.categorical_columns),
        len(meta.datetime_columns),
        len(meta.boolean_columns),
        meta.data_strategy,
        len(meta.pii_columns),
    )
    return meta
