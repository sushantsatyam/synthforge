"""Constraint-Augmented Generation (CAG) system.

Constraints define business rules that synthetic data must satisfy.
The CAG approach transforms data before training to embed constraints,
and inverse-transforms after generation. Falls back to reject sampling.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseConstraint(ABC):
    """Abstract base for all constraints."""

    @abstractmethod
    def is_valid(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series indicating which rows satisfy the constraint."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data to embed the constraint (pre-training)."""

    @abstractmethod
    def reverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse transform after generation."""

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        """Columns involved in this constraint."""


class Inequality(BaseConstraint):
    """Ensure column_a < column_b (or <=, >, >=)."""

    def __init__(
        self,
        low_column: str,
        high_column: str,
        strict: bool = False,
    ):
        self._low = low_column
        self._high = high_column
        self._strict = strict

    def is_valid(self, df: pd.DataFrame) -> pd.Series:
        if self._low not in df.columns or self._high not in df.columns:
            return pd.Series(True, index=df.index)
        if not pd.api.types.is_numeric_dtype(df[self._low]) or not pd.api.types.is_numeric_dtype(df[self._high]):
            return pd.Series(True, index=df.index)
        if self._strict:
            return df[self._low] < df[self._high]
        return df[self._low] <= df[self._high]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace high_column with the diff (high - low)."""
        df = df.copy()
        if self._low not in df.columns or self._high not in df.columns:
            return df
        if not pd.api.types.is_numeric_dtype(df[self._low]) or not pd.api.types.is_numeric_dtype(df[self._high]):
            return df
        diff_col = f"__{self._low}__{self._high}__diff"
        df[diff_col] = df[self._high] - df[self._low]
        df = df.drop(columns=[self._high])
        return df

    def reverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        diff_col = f"__{self._low}__{self._high}__diff"
        if diff_col in df.columns and self._low in df.columns:
            if pd.api.types.is_numeric_dtype(df[self._low]):
                diff = df[diff_col].clip(lower=0)
                if self._strict:
                    diff = diff.clip(lower=1e-6)
                df[self._high] = df[self._low] + diff
                df = df.drop(columns=[diff_col])
        return df

    @property
    def columns(self) -> list[str]:
        return [self._low, self._high]


class PositiveValue(BaseConstraint):
    """Ensure a column is always positive."""

    def __init__(self, column: str, strict: bool = True):
        self._column = column
        self._strict = strict

    def is_valid(self, df: pd.DataFrame) -> pd.Series:
        if self._column not in df.columns or not pd.api.types.is_numeric_dtype(df[self._column]):
            return pd.Series(True, index=df.index)
        if self._strict:
            return df[self._column] > 0
        return df[self._column] >= 0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._column in df.columns and pd.api.types.is_numeric_dtype(df[self._column]):
            df[self._column] = np.log1p(df[self._column].clip(lower=0))
        return df

    def reverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._column not in df.columns or not pd.api.types.is_numeric_dtype(df[self._column]):
            return df
        df[self._column] = np.expm1(df[self._column]).clip(lower=0)
        if self._strict:
            df[self._column] = df[self._column].clip(lower=1e-10)
        return df

    @property
    def columns(self) -> list[str]:
        return [self._column]


class FixedCombinations(BaseConstraint):
    """Ensure a set of columns only has combinations seen in training data."""

    def __init__(self, column_names: list[str]):
        self._columns = column_names
        self._valid_combos: set[tuple] = set()

    def fit(self, df: pd.DataFrame) -> None:
        """Learn valid combinations from training data."""
        self._valid_combos = set(
            df[self._columns].dropna().itertuples(index=False, name=None)
        )

    def is_valid(self, df: pd.DataFrame) -> pd.Series:
        combos = df[self._columns].apply(tuple, axis=1)
        return combos.isin(self._valid_combos)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine columns into a single hash column."""
        self.fit(df)
        df = df.copy()
        hash_col = "__" + "_".join(self._columns) + "__combo"
        df[hash_col] = df[self._columns].apply(lambda r: hash(tuple(r)), axis=1)
        df = df.drop(columns=self._columns[1:])  # Keep first, drop rest
        return df

    def reverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore original columns by mapping back from first column."""
        # For simplicity, use nearest-neighbor matching on the first column
        return df  # Placeholder: full implementation maps combo hashes

    @property
    def columns(self) -> list[str]:
        return self._columns


class ValueRange(BaseConstraint):
    """Ensure a column stays within [low, high]."""

    def __init__(self, column: str, low: float | None = None, high: float | None = None):
        self._column = column
        self._low = low
        self._high = high

    def is_valid(self, df: pd.DataFrame) -> pd.Series:
        if self._column not in df.columns or not pd.api.types.is_numeric_dtype(df[self._column]):
            return pd.Series(True, index=df.index)
        valid = pd.Series(True, index=df.index)
        if self._low is not None:
            valid &= df[self._column] >= self._low
        if self._high is not None:
            valid &= df[self._column] <= self._high
        return valid

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # Range is enforced post-generation

    def reverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._column not in df.columns:
            return df
        if not pd.api.types.is_numeric_dtype(df[self._column]):
            return df  # Skip non-numeric columns silently
        if self._low is not None:
            df[self._column] = df[self._column].clip(lower=self._low)
        if self._high is not None:
            df[self._column] = df[self._column].clip(upper=self._high)
        return df

    @property
    def columns(self) -> list[str]:
        return [self._column]


class ConstraintPipeline:
    """Manages multiple constraints for a table."""

    def __init__(self, constraints: list[BaseConstraint] | None = None):
        self._constraints = constraints or []

    def add(self, constraint: BaseConstraint) -> None:
        self._constraints.append(constraint)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self._constraints:
            df = c.transform(df)
        return df

    def reverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reverse in opposite order
        for c in reversed(self._constraints):
            df = c.reverse_transform(df)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with per-constraint validity columns."""
        results = {}
        for i, c in enumerate(self._constraints):
            results[f"constraint_{i}_{c.__class__.__name__}"] = c.is_valid(df)
        return pd.DataFrame(results, index=df.index)

    def validity_rate(self, df: pd.DataFrame) -> float:
        """Fraction of rows satisfying all constraints."""
        if not self._constraints:
            return 1.0
        all_valid = pd.Series(True, index=df.index)
        for c in self._constraints:
            all_valid &= c.is_valid(df)
        return float(all_valid.mean())

    def filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows satisfying all constraints (reject sampling)."""
        if not self._constraints:
            return df
        mask = pd.Series(True, index=df.index)
        for c in self._constraints:
            mask &= c.is_valid(df)
        return df[mask].reset_index(drop=True)

    @property
    def constraint_count(self) -> int:
        return len(self._constraints)
