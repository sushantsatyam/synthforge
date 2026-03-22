"""Five-layer evaluation pipeline for synthetic data quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from synthforge.metadata import Metadata, SDType

logger = logging.getLogger(__name__)


@dataclass
class EvalMetric:
    name: str
    value: float
    threshold: float | None = None
    passed: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    diagnostics: list[EvalMetric] = field(default_factory=list)
    statistical_fidelity: list[EvalMetric] = field(default_factory=list)
    ml_utility: list[EvalMetric] = field(default_factory=list)
    privacy: list[EvalMetric] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        def _m(ms):
            return [{"name": m.name, "value": round(m.value, 4), "passed": m.passed} for m in ms]
        return {
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "diagnostics": _m(self.diagnostics),
            "statistical_fidelity": _m(self.statistical_fidelity),
            "ml_utility": _m(self.ml_utility),
            "privacy": _m(self.privacy),
        }

    def summary(self) -> str:
        lines = [f"Overall: {self.overall_score:.2%} {'PASS' if self.passed else 'FAIL'}"]
        for name, ms in [("Diagnostics", self.diagnostics), ("Fidelity", self.statistical_fidelity),
                         ("ML Utility", self.ml_utility), ("Privacy", self.privacy)]:
            if ms:
                lines.append(f"  {name}: {np.mean([m.value for m in ms]):.2%}")
                for m in ms:
                    s = "PASS" if m.passed else "FAIL" if m.passed is not None else "-"
                    lines.append(f"    {m.name}: {m.value:.4f} [{s}]")
        return "\n".join(lines)


class Evaluator:
    def __init__(self, quality_threshold: float = 0.70):
        self._threshold = quality_threshold

    def evaluate(self, real_df, synthetic_df, metadata, run_diagnostics=True,
                 run_fidelity=True, run_ml_utility=False, run_privacy=False):
        report = EvaluationReport()
        if run_diagnostics:
            report.diagnostics = self._diagnostics(real_df, synthetic_df, metadata)
        if run_fidelity:
            report.statistical_fidelity = self._fidelity(real_df, synthetic_df, metadata)
        if run_ml_utility:
            report.ml_utility = self._ml_utility(real_df, synthetic_df, metadata)
        if run_privacy:
            report.privacy = self._privacy(real_df, synthetic_df, metadata)
        all_m = report.diagnostics + report.statistical_fidelity + report.ml_utility + report.privacy
        report.overall_score = np.mean([m.value for m in all_m]) if all_m else 0.0
        report.passed = report.overall_score >= self._threshold
        return report

    def _non_pii_numerical(self, meta):
        """Numerical columns excluding PII (PII columns are Faker-replaced, not comparable)."""
        return [c for c in meta.numerical_columns if not meta.columns[c].is_pii]

    def _non_pii_categorical(self, meta):
        """Categorical columns excluding PII."""
        return [c for c in meta.categorical_columns if not meta.columns[c].is_pii]

    def _diagnostics(self, real_df, syn_df, meta):
        metrics = []
        shared = set(real_df.columns) & set(syn_df.columns)
        conf = len(shared) / len(real_df.columns) if len(real_df.columns) else 1.0
        metrics.append(EvalMetric("schema_conformance", conf, 0.95, conf >= 0.95))

        bounds = []
        for col in self._non_pii_numerical(meta):
            if col in syn_df.columns and col in real_df.columns:
                rmin, rmax = real_df[col].min(), real_df[col].max()
                sv = syn_df[col].dropna()
                if len(sv) > 0:
                    bounds.append(float(((sv >= rmin) & (sv <= rmax)).mean()))
        if bounds:
            ab = np.mean(bounds)
            metrics.append(EvalMetric("boundary_adherence", ab, 0.90, ab >= 0.90))

        cats = []
        for col in self._non_pii_categorical(meta):
            if col in syn_df.columns and col in real_df.columns:
                rc = set(real_df[col].dropna().unique())
                sc = set(syn_df[col].dropna().unique())
                if sc:
                    cats.append(len(sc & rc) / len(sc))
        if cats:
            ac = np.mean(cats)
            metrics.append(EvalMetric("category_adherence", ac, 0.85, ac >= 0.85))
        return metrics

    def _fidelity(self, real_df, syn_df, meta):
        metrics = []
        ks = []
        for col in self._non_pii_numerical(meta):
            if col in syn_df.columns and col in real_df.columns:
                rv, sv = real_df[col].dropna().values, syn_df[col].dropna().values
                if len(rv) >= 5 and len(sv) >= 5:
                    stat, _ = sp_stats.ks_2samp(rv, sv)
                    ks.append(1 - stat)
        if ks:
            ak = np.mean(ks)
            metrics.append(EvalMetric("ks_complement", ak, 0.70, ak >= 0.70))

        tv = []
        for col in self._non_pii_categorical(meta):
            if col in syn_df.columns and col in real_df.columns:
                rf = real_df[col].value_counts(normalize=True).to_dict()
                sf = syn_df[col].value_counts(normalize=True).to_dict()
                allc = set(rf.keys()) | set(sf.keys())
                tvd = sum(abs(rf.get(c, 0) - sf.get(c, 0)) for c in allc) / 2
                tv.append(1 - tvd)
        if tv:
            at = np.mean(tv)
            metrics.append(EvalMetric("tv_complement", at, 0.70, at >= 0.70))

        nc = [c for c in self._non_pii_numerical(meta) if c in real_df.columns and c in syn_df.columns]
        if len(nc) >= 2:
            real_clean = real_df[nc].replace([np.inf, -np.inf], np.nan).dropna()
            syn_clean = syn_df[nc].replace([np.inf, -np.inf], np.nan).dropna()
            if len(real_clean) >= 10 and len(syn_clean) >= 10:
                rc = real_clean.corr().values
                sc = syn_clean.corr().values
                mask = np.triu(np.ones_like(rc, dtype=bool), k=1)
                rflat = rc[mask]
                sflat = sc[mask]
                # Guard against NaN in correlation matrices
                valid = np.isfinite(rflat) & np.isfinite(sflat)
                if valid.sum() > 0:
                    diff = np.mean(np.abs(rflat[valid] - sflat[valid]))
                    cs = max(0.0, 1.0 - diff)
                else:
                    cs = 0.0
                metrics.append(EvalMetric("correlation_similarity", cs, 0.65, cs >= 0.65))

        c2 = self._c2st(real_df, syn_df, meta)
        if c2 is not None:
            # C2ST interpretation:
            #   AUC ≈ 0.5 → classifier can't distinguish → perfect (score=1.0)
            #   AUC → 1.0 → trivially distinguishable → bad (score=0.0)
            #   AUC < 0.5 → classifier worse than random → still indistinguishable → score=1.0
            clamped_auc = max(c2, 0.5)
            score = max(0.0, 1.0 - 2.0 * (clamped_auc - 0.5))
            metrics.append(EvalMetric("c2st_score", score, 0.70, score >= 0.70, {"raw_auc": c2}))
        return metrics

    def _c2st(self, real_df, syn_df, meta):
        nc = [c for c in self._non_pii_numerical(meta) if c in real_df.columns and c in syn_df.columns]
        if len(nc) < 2:
            return None
        rs = real_df[nc].replace([np.inf, -np.inf], np.nan).dropna().head(2000)
        ss = syn_df[nc].replace([np.inf, -np.inf], np.nan).dropna().head(2000)
        if len(rs) < 50 or len(ss) < 50:
            return None
        X = pd.concat([rs, ss], ignore_index=True).values
        y = np.array([0]*len(rs) + [1]*len(ss))
        # Check for remaining nan/inf in X
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            clf.fit(Xtr, ytr)
            return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        except Exception as e:
            logger.warning("C2ST failed: %s", e)
            return None

    def _ml_utility(self, real_df, syn_df, meta):
        metrics = []
        target = None
        for col in self._non_pii_categorical(meta):
            if col in real_df.columns and col in syn_df.columns and 2 <= real_df[col].nunique() <= 10:
                target = col
                break
        if not target:
            return metrics
        fcols = [c for c in self._non_pii_numerical(meta) if c in real_df.columns and c in syn_df.columns]
        if len(fcols) < 2:
            return metrics
        try:
            le = LabelEncoder()
            rc = real_df[fcols + [target]].dropna()
            sc = syn_df[fcols + [target]].dropna()
            if len(rc) < 50 or len(sc) < 50:
                return metrics
            le.fit(pd.concat([rc[target], sc[target]]).astype(str))
            Xr, yr = rc[fcols].values, le.transform(rc[target].astype(str))
            Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
            clf1 = RandomForestClassifier(100, random_state=42)
            clf1.fit(Xtr, ytr)
            trtr = accuracy_score(yte, clf1.predict(Xte))
            Xs, ys = sc[fcols].values, le.transform(sc[target].astype(str))
            clf2 = RandomForestClassifier(100, random_state=42)
            clf2.fit(Xs, ys)
            tstr = accuracy_score(yte, clf2.predict(Xte))
            ratio = tstr / trtr if trtr > 0 else 0
            metrics.append(EvalMetric("tstr_trtr_ratio", ratio, 0.85, ratio >= 0.85,
                                      {"trtr": trtr, "tstr": tstr, "target": target}))
        except Exception as e:
            logger.warning("TSTR failed: %s", e)
        return metrics

    def _privacy(self, real_df, syn_df, meta):
        metrics = []
        mia = self._c2st(real_df, syn_df, meta)
        if mia is not None:
            safe = mia < 0.60
            metrics.append(EvalMetric("mia_resistance", 1 - mia, 0.40, safe, {"mia_auc": mia}))
        return metrics
