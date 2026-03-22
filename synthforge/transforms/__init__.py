"""Reversible data transforms for preprocessing tabular data before synthesis.

Each transformer converts a column to a numerical representation suitable for
generative models, and provides an inverse transform to convert back.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler

from synthforge.metadata import ColumnMeta, Metadata, SDType

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    """Abstract base class for reversible column transforms."""

    @abstractmethod
    def fit(self, series: pd.Series, col_meta: ColumnMeta) -> None:
        """Learn transform parameters from real data."""

    @abstractmethod
    def transform(self, series: pd.Series) -> pd.DataFrame:
        """Transform a column into one or more numerical columns."""

    @abstractmethod
    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        """Reverse the transform back to original representation."""

    @property
    @abstractmethod
    def output_columns(self) -> list[str]:
        """Names of the output columns produced by transform."""


class NumericalTransform(BaseTransform):
    """Transform numerical columns with null handling and optional normalization."""

    def __init__(self, normalize: str = "quantile"):
        """
        Args:
            normalize: 'standard', 'quantile', or 'none'.
        """
        self._normalize = normalize
        self._col_name: str = ""
        self._has_nulls: bool = False
        self._scaler: Any = None
        self._min: float = 0.0
        self._max: float = 0.0
        self._fill_value: float = 0.0

    def fit(self, series: pd.Series, col_meta: ColumnMeta) -> None:
        self._col_name = col_meta.name
        self._has_nulls = col_meta.nullable
        clean = series.dropna().astype(float)

        if len(clean) == 0:
            self._fill_value = 0.0
            return

        self._min = float(clean.min())
        self._max = float(clean.max())
        self._fill_value = float(clean.median())

        if self._normalize == "quantile" and len(clean) > 10:
            n_quantiles = min(len(clean), 1000)
            self._scaler = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=42,
            )
            self._scaler.fit(clean.values.reshape(-1, 1))
        elif self._normalize == "standard":
            self._scaler = StandardScaler()
            self._scaler.fit(clean.values.reshape(-1, 1))

    def transform(self, series: pd.Series) -> pd.DataFrame:
        result = {}
        filled = series.fillna(self._fill_value).astype(float)

        if self._scaler is not None:
            values = self._scaler.transform(filled.values.reshape(-1, 1)).ravel()
        else:
            values = filled.values.astype(float)

        result[f"{self._col_name}.value"] = values

        if self._has_nulls:
            result[f"{self._col_name}.is_null"] = series.isna().astype(float).values

        return pd.DataFrame(result, index=series.index)

    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        values = data[f"{self._col_name}.value"].values.astype(float)

        if self._scaler is not None:
            values = self._scaler.inverse_transform(values.reshape(-1, 1)).ravel()

        # Clip to observed range
        values = np.clip(values, self._min, self._max)

        result = pd.Series(values, index=data.index, name=self._col_name)

        if self._has_nulls and f"{self._col_name}.is_null" in data.columns:
            null_mask = data[f"{self._col_name}.is_null"] > 0.5
            result[null_mask] = np.nan

        return result

    @property
    def output_columns(self) -> list[str]:
        cols = [f"{self._col_name}.value"]
        if self._has_nulls:
            cols.append(f"{self._col_name}.is_null")
        return cols


class CategoricalTransform(BaseTransform):
    """Transform categorical columns using label encoding + null handling."""

    def __init__(self) -> None:
        self._col_name: str = ""
        self._has_nulls: bool = False
        self._encoder: LabelEncoder | None = None
        self._categories: list[str] = []

    def fit(self, series: pd.Series, col_meta: ColumnMeta) -> None:
        self._col_name = col_meta.name
        self._has_nulls = col_meta.nullable
        clean = series.dropna().astype(str)
        self._categories = sorted(clean.unique().tolist())
        self._encoder = LabelEncoder()
        self._encoder.fit(self._categories)

    def transform(self, series: pd.Series) -> pd.DataFrame:
        result = {}
        filled = series.fillna("__NULL__").astype(str)

        # Map known categories; unknowns get -1
        encoded = []
        cat_set = set(self._categories)
        for val in filled:
            if val in cat_set:
                encoded.append(int(self._encoder.transform([val])[0]))
            else:
                encoded.append(-1)

        result[f"{self._col_name}.value"] = encoded

        if self._has_nulls:
            result[f"{self._col_name}.is_null"] = series.isna().astype(float).values

        return pd.DataFrame(result, index=series.index)

    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        encoded = data[f"{self._col_name}.value"].values
        n_classes = len(self._categories)

        # Round and clip to valid range
        indices = np.clip(np.round(encoded).astype(int), 0, max(n_classes - 1, 0))
        decoded = self._encoder.inverse_transform(indices)
        result = pd.Series(decoded, index=data.index, name=self._col_name)

        if self._has_nulls and f"{self._col_name}.is_null" in data.columns:
            null_mask = data[f"{self._col_name}.is_null"] > 0.5
            result[null_mask] = np.nan

        return result

    @property
    def output_columns(self) -> list[str]:
        cols = [f"{self._col_name}.value"]
        if self._has_nulls:
            cols.append(f"{self._col_name}.is_null")
        return cols


class DatetimeTransform(BaseTransform):
    """Convert datetime to numerical (unix timestamp in seconds)."""

    def __init__(self) -> None:
        self._col_name: str = ""
        self._has_nulls: bool = False
        self._min_ts: float = 0.0
        self._max_ts: float = 0.0
        self._fill_ts: float = 0.0

    def fit(self, series: pd.Series, col_meta: ColumnMeta) -> None:
        self._col_name = col_meta.name
        self._has_nulls = col_meta.nullable

        dt_series = pd.to_datetime(series, errors="coerce")
        clean = dt_series.dropna()

        if len(clean) == 0:
            return

        ts = clean.astype(np.int64) / 1e9
        self._min_ts = float(ts.min())
        self._max_ts = float(ts.max())
        self._fill_ts = float(ts.median())

    def transform(self, series: pd.Series) -> pd.DataFrame:
        dt_series = pd.to_datetime(series, errors="coerce")
        ts = dt_series.astype(np.int64) / 1e9
        ts = ts.fillna(self._fill_ts)

        result = {f"{self._col_name}.value": ts.values}
        if self._has_nulls:
            result[f"{self._col_name}.is_null"] = series.isna().astype(float).values

        return pd.DataFrame(result, index=series.index)

    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        ts = data[f"{self._col_name}.value"].values.astype(float)
        ts = np.clip(ts, self._min_ts, self._max_ts)
        dt = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None)
        result = pd.Series(dt, index=data.index, name=self._col_name)

        if self._has_nulls and f"{self._col_name}.is_null" in data.columns:
            null_mask = data[f"{self._col_name}.is_null"] > 0.5
            result[null_mask] = pd.NaT

        return result

    @property
    def output_columns(self) -> list[str]:
        cols = [f"{self._col_name}.value"]
        if self._has_nulls:
            cols.append(f"{self._col_name}.is_null")
        return cols


class BooleanTransform(BaseTransform):
    """Convert booleans to 0/1 float."""

    def __init__(self) -> None:
        self._col_name: str = ""
        self._has_nulls: bool = False

    def fit(self, series: pd.Series, col_meta: ColumnMeta) -> None:
        self._col_name = col_meta.name
        self._has_nulls = col_meta.nullable

    def transform(self, series: pd.Series) -> pd.DataFrame:
        bool_map = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0,
                     "t": 1.0, "f": 0.0, "y": 1.0, "n": 0.0}
        if pd.api.types.is_bool_dtype(series.dtype):
            values = series.astype(float).fillna(0.5).values
        else:
            values = series.astype(str).str.lower().map(bool_map).fillna(0.5).values

        result = {f"{self._col_name}.value": values}
        if self._has_nulls:
            result[f"{self._col_name}.is_null"] = series.isna().astype(float).values
        return pd.DataFrame(result, index=series.index)

    def inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        values = data[f"{self._col_name}.value"].values
        bools = values > 0.5
        result = pd.Series(bools, index=data.index, name=self._col_name)

        if self._has_nulls and f"{self._col_name}.is_null" in data.columns:
            null_mask = data[f"{self._col_name}.is_null"] > 0.5
            result = result.astype(object)
            result[null_mask] = np.nan

        return result

    @property
    def output_columns(self) -> list[str]:
        cols = [f"{self._col_name}.value"]
        if self._has_nulls:
            cols.append(f"{self._col_name}.is_null")
        return cols


# ──────────────────────────────────────────────────────────────────
# Transform pipeline
# ──────────────────────────────────────────────────────────────────

_TRANSFORM_MAP: dict[SDType, type[BaseTransform]] = {
    SDType.NUMERICAL: NumericalTransform,
    SDType.CATEGORICAL: CategoricalTransform,
    SDType.DATETIME: DatetimeTransform,
    SDType.BOOLEAN: BooleanTransform,
}


class TransformPipeline:
    """Manages a set of column transforms for an entire table."""

    def __init__(self) -> None:
        self._transforms: dict[str, BaseTransform] = {}
        self._skip_columns: set[str] = set()  # ID columns, text columns

    def fit(self, df: pd.DataFrame, metadata: Metadata) -> None:
        """Fit transforms for all columns based on metadata."""
        self._transforms.clear()
        self._skip_columns.clear()

        for col_name, col_meta in metadata.columns.items():
            if col_name not in df.columns:
                continue

            if col_meta.sdtype in (SDType.ID, SDType.TEXT, SDType.UNKNOWN):
                self._skip_columns.add(col_name)
                continue

            # PII columns with Faker providers are replaced post-generation,
            # so exclude them from statistical training to reduce noise.
            if col_meta.is_pii and col_meta.faker_provider:
                self._skip_columns.add(col_name)
                continue

            transform_cls = _TRANSFORM_MAP.get(col_meta.sdtype)
            if transform_cls is None:
                self._skip_columns.add(col_name)
                continue

            transform = transform_cls()
            transform.fit(df[col_name], col_meta)
            self._transforms[col_name] = transform

        logger.info(
            "TransformPipeline: %d transforms fitted, %d columns skipped",
            len(self._transforms),
            len(self._skip_columns),
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform all columns into a numerical DataFrame."""
        parts = []
        for col_name, transform in self._transforms.items():
            if col_name in df.columns:
                parts.append(transform.transform(df[col_name]))
        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverse transform back to original column types."""
        result = {}
        for col_name, transform in self._transforms.items():
            # Gather the output columns for this transform
            out_cols = [c for c in transform.output_columns if c in data.columns]
            if out_cols:
                sub = data[out_cols]
                result[col_name] = transform.inverse_transform(sub)

        return pd.DataFrame(result, index=data.index)

    @property
    def transformed_columns(self) -> list[str]:
        """All output column names in the transformed space."""
        cols = []
        for transform in self._transforms.values():
            cols.extend(transform.output_columns)
        return cols

    @property
    def skip_columns(self) -> set[str]:
        return self._skip_columns.copy()
