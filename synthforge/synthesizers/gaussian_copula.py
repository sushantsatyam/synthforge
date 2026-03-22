"""Gaussian Copula synthesizer — fast, CPU-only, no deep learning deps.

Uses scipy/sklearn for marginal fitting + multivariate Gaussian for correlation capture.
This is the default synthesizer for speed and stability.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

from synthforge.synthesizers import BaseSynthesizer, register_synthesizer

logger = logging.getLogger(__name__)


@register_synthesizer("gaussian_copula")
class GaussianCopulaSynthesizer(BaseSynthesizer):
    """Gaussian Copula: fits marginals per column + Gaussian correlation structure.

    Algorithm:
      1. For each column, fit a marginal distribution (or use empirical CDF).
      2. Transform data to uniform margins via the fitted CDF (probability integral transform).
      3. Transform uniform margins to standard normal via inverse normal CDF.
      4. Fit a multivariate normal to the transformed data (learn correlation matrix).
      5. To sample: draw from multivariate normal → inverse normal CDF → inverse marginal CDF.
    """

    def __init__(
        self,
        default_distribution: str = "beta",
        random_state: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._default_dist = default_distribution
        self._rng = np.random.default_rng(random_state)
        self._marginals: list[dict[str, Any]] = []
        self._correlation: np.ndarray | None = None
        self._means: np.ndarray | None = None

    # ─── Supported marginal distributions ──────────────────────────

    _DISTRIBUTIONS = {
        "norm": stats.norm,
        "beta": stats.beta,
        "gamma": stats.gamma,
        "lognorm": stats.lognorm,
        "uniform": stats.uniform,
        "expon": stats.expon,
    }

    def _fit_marginal(self, col_data: np.ndarray, col_idx: int) -> dict[str, Any]:
        """Fit the best marginal distribution for a single column.

        Tries multiple distributions and picks the one with lowest KS statistic.
        Falls back to empirical CDF for degenerate or hard-to-fit columns.
        """
        clean = col_data[~np.isnan(col_data)]

        if len(clean) < 5 or np.std(clean) < 1e-10:
            return {"type": "empirical", "values": clean.copy()}

        # Try multiple distributions, pick best KS fit
        candidates = ["beta", "norm", "gamma", "lognorm", "uniform", "expon"]
        best: dict[str, Any] | None = None
        best_ks = 1.0

        for dist_name in candidates:
            dist_fn = self._DISTRIBUTIONS.get(dist_name)
            if dist_fn is None:
                continue
            try:
                params = dist_fn.fit(clean)
                # Quick KS test on a sample for speed
                sample = clean[:min(500, len(clean))]
                ks_stat, _ = stats.ks_2samp(sample, dist_fn.rvs(*params, size=len(sample)))
                if ks_stat < best_ks:
                    best_ks = ks_stat
                    best = {"type": dist_name, "params": params, "dist_fn": dist_fn}
            except Exception:
                continue

        if best is not None and best_ks < 0.3:
            return best

        # Fallback: empirical CDF (always works, captures arbitrary shapes)
        return {"type": "empirical", "values": clean.copy()}

    def _cdf(self, values: np.ndarray, marginal: dict) -> np.ndarray:
        """Apply CDF transform using fitted marginal."""
        if marginal["type"] == "empirical":
            # Use rank-based empirical CDF
            emp = marginal["values"]
            sorted_emp = np.sort(emp)
            n = len(sorted_emp)
            result = np.searchsorted(sorted_emp, values, side="right") / (n + 1)
        else:
            result = marginal["dist_fn"].cdf(values, *marginal["params"])

        # Clip to avoid infinities in ppf
        return np.clip(result, 1e-6, 1 - 1e-6)

    def _ppf(self, quantiles: np.ndarray, marginal: dict) -> np.ndarray:
        """Apply inverse CDF (quantile function)."""
        quantiles = np.clip(quantiles, 1e-6, 1 - 1e-6)

        if marginal["type"] == "empirical":
            emp = marginal["values"]
            sorted_emp = np.sort(emp)
            n = len(sorted_emp)
            indices = np.clip((quantiles * (n + 1)).astype(int) - 1, 0, n - 1)
            return sorted_emp[indices]
        else:
            return marginal["dist_fn"].ppf(quantiles, *marginal["params"])

    # ─── Fit / Sample ──────────────────────────────────────────────

    def fit(self, data: np.ndarray, column_names: list[str] | None = None) -> None:
        n_samples, n_cols = data.shape
        self._n_columns = n_cols
        self._column_names = column_names or [f"col_{i}" for i in range(n_cols)]

        logger.info("Fitting GaussianCopula on %d samples, %d columns", n_samples, n_cols)

        # Step 1: Fit marginals
        self._marginals = []
        for j in range(n_cols):
            marginal = self._fit_marginal(data[:, j], j)
            self._marginals.append(marginal)

        # Step 2–3: Transform to normal space
        normal_data = np.zeros_like(data)
        for j in range(n_cols):
            col_data = data[:, j]
            uniform = self._cdf(col_data, self._marginals[j])
            normal_data[:, j] = stats.norm.ppf(uniform)

        # Handle any remaining infinities
        normal_data = np.nan_to_num(normal_data, nan=0.0, posinf=3.0, neginf=-3.0)

        # Step 4: Fit multivariate Gaussian
        self._means = np.mean(normal_data, axis=0)
        self._correlation = np.cov(normal_data, rowvar=False)

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(self._correlation)
        if np.any(eigvals < -1e-8):
            # Regularize
            self._correlation += np.eye(n_cols) * (abs(eigvals.min()) + 1e-6)

        self._fitted = True
        logger.info("GaussianCopula fit complete")

    def sample(self, n_rows: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Synthesizer not fitted. Call fit() first.")

        logger.info("Sampling %d rows from GaussianCopula", n_rows)

        # Step 5a: Draw from multivariate normal
        normal_samples = self._rng.multivariate_normal(
            mean=self._means,
            cov=self._correlation,
            size=n_rows,
        )

        # Step 5b: Transform to uniform via normal CDF
        uniform_samples = stats.norm.cdf(normal_samples)

        # Step 5c: Transform to original space via inverse marginal CDF
        result = np.zeros((n_rows, self._n_columns))
        for j in range(self._n_columns):
            result[:, j] = self._ppf(uniform_samples[:, j], self._marginals[j])

        return result
