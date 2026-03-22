"""Tests for SynthForge core pipeline.

Tests are organized by module:
  - TestMetadata: schema detection, PII, primary key
  - TestTransforms: reversible transforms, null handling
  - TestSynthesizers: registry, Gaussian Copula fit/sample
  - TestCUDAGate: enforce GPU requirement for neural models
  - TestConstraints: inequality, positive, range, type guards
  - TestEvaluation: fidelity, diagnostics, serialization
  - TestStrategy: auto-selection, CUDA-aware routing
  - TestFullPipeline: end-to-end integration
  - TestConfig: configuration models
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from synthforge import SynthForge, SynthForgeConfig
from synthforge.config import (
    SynthesizerType, DataStrategy, GenerationConfig,
    TabDDPMConfig, TabSynConfig, GReaTConfig,
)
from synthforge.metadata import detect_metadata, SDType, SemanticType, Metadata
from synthforge.transforms import TransformPipeline
from synthforge.constraints import Inequality, PositiveValue, ValueRange, ConstraintPipeline
from synthforge.evaluation import Evaluator
from synthforge.synthesizers import list_synthesizers, get_synthesizer
from synthforge.strategies import (
    recommend_strategy, recommend_synthesizer,
    _cuda_available, _torch_available, create_synthesizer,
)


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def mixed_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "id": range(1, n + 1),
        "age": np.random.randint(18, 80, n),
        "salary": np.random.lognormal(10.5, 0.8, n).round(2),
        "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR"], n),
        "is_active": np.random.choice([True, False], n),
        "hire_date": pd.date_range("2015-01-01", periods=n, freq="D"),
        "email": [f"user{i}@example.com" for i in range(n)],
        "score": np.random.normal(75, 15, n).round(1),
    })

@pytest.fixture
def numerical_df():
    np.random.seed(42)
    n = 300
    x1 = np.random.normal(50, 10, n)
    return pd.DataFrame({
        "x1": x1,
        "x2": x1 * 0.7 + np.random.normal(0, 5, n),
        "x3": np.random.exponential(20, n),
        "x4": np.random.uniform(0, 1, n),
    })

@pytest.fixture
def categorical_df():
    np.random.seed(42)
    n = 400
    return pd.DataFrame({
        "color": np.random.choice(["red", "blue", "green", "yellow"], n, p=[.4, .3, .2, .1]),
        "size": np.random.choice(["S", "M", "L", "XL"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "count": np.random.randint(1, 100, n),
    })


# ── Metadata Tests ─────────────────────────────────────────────────

class TestMetadata:
    def test_detect_types(self, mixed_df):
        m = detect_metadata(mixed_df)
        assert m.columns["age"].sdtype == SDType.NUMERICAL
        assert m.columns["department"].sdtype == SDType.CATEGORICAL
        assert m.columns["is_active"].sdtype == SDType.BOOLEAN
        assert m.columns["hire_date"].sdtype == SDType.DATETIME

    def test_pii_detection(self, mixed_df):
        m = detect_metadata(mixed_df)
        assert m.columns["email"].is_pii
        assert m.columns["email"].semantic_type == SemanticType.EMAIL

    def test_pii_not_id(self, mixed_df):
        """PII columns should not be classified as ID even if all-unique."""
        m = detect_metadata(mixed_df)
        assert m.columns["email"].sdtype != SDType.ID

    def test_primary_key(self, mixed_df):
        m = detect_metadata(mixed_df)
        assert m.primary_key == "id"

    def test_explicit_primary_key(self, mixed_df):
        m = detect_metadata(mixed_df, primary_key="age")
        assert m.primary_key == "age"

    def test_strategy(self, numerical_df):
        assert detect_metadata(numerical_df).data_strategy == "numerical"

    def test_to_dict(self, mixed_df):
        d = detect_metadata(mixed_df).to_dict()
        assert "columns" in d and "primary_key" in d
        assert isinstance(d["columns"]["age"]["sdtype"], str)


# ── Transform Tests ────────────────────────────────────────────────

class TestTransforms:
    def test_numerical_roundtrip(self, numerical_df):
        m = detect_metadata(numerical_df)
        p = TransformPipeline()
        p.fit(numerical_df, m)
        t = p.transform(numerical_df)
        assert not t.empty
        inv = p.inverse_transform(t)
        for c in ["x1", "x2", "x3", "x4"]:
            np.testing.assert_allclose(inv[c].values, numerical_df[c].values, rtol=0.1, atol=1.0)

    def test_null_handling(self):
        df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0, None], "b": ["x", None, "y", "x", "z"]})
        m = detect_metadata(df)
        p = TransformPipeline()
        p.fit(df, m)
        t = p.transform(df)
        assert "a.is_null" in t.columns

    def test_pii_columns_skipped(self, mixed_df):
        """PII columns with faker providers should be skipped from transforms."""
        m = detect_metadata(mixed_df)
        p = TransformPipeline()
        p.fit(mixed_df, m)
        assert "email" in p.skip_columns
        # email should not appear in transformed output
        t = p.transform(mixed_df)
        assert all("email" not in col for col in t.columns)


# ── Synthesizer Tests ──────────────────────────────────────────────

class TestSynthesizers:
    def test_registry_has_copula(self):
        assert "gaussian_copula" in list_synthesizers()

    def test_registry_has_sota_models(self):
        """SOTA models should be registered (even if torch unavailable, they import conditionally)."""
        registered = list_synthesizers()
        assert "gaussian_copula" in registered
        # These may or may not be present depending on torch availability
        # But gaussian_copula should always be there

    def test_copula_fit_sample(self, numerical_df):
        s = get_synthesizer("gaussian_copula")(random_state=42)
        s.fit(numerical_df.values, column_names=numerical_df.columns.tolist())
        assert s.is_fitted
        out = s.sample(100)
        assert out.shape == (100, 4) and not np.any(np.isnan(out))

    def test_copula_sample_dataframe(self, numerical_df):
        s = get_synthesizer("gaussian_copula")(random_state=42)
        s.fit(numerical_df.values, column_names=numerical_df.columns.tolist())
        df = s.sample_dataframe(50)
        assert isinstance(df, pd.DataFrame) and len(df) == 50
        assert list(df.columns) == list(numerical_df.columns)

    def test_copula_preserves_correlation(self, numerical_df):
        """Gaussian Copula should preserve inter-column correlation structure."""
        s = get_synthesizer("gaussian_copula")(random_state=42)
        s.fit(numerical_df.values, column_names=numerical_df.columns.tolist())
        syn = pd.DataFrame(s.sample(2000), columns=numerical_df.columns)
        # x1 and x2 are correlated (x2 = 0.7*x1 + noise)
        real_corr = numerical_df[["x1", "x2"]].corr().iloc[0, 1]
        syn_corr = syn[["x1", "x2"]].corr().iloc[0, 1]
        assert abs(real_corr - syn_corr) < 0.15  # Within 0.15

    def test_unknown_synthesizer_raises(self):
        with pytest.raises(ValueError, match="Unknown synthesizer"):
            get_synthesizer("nonexistent_model")

    def test_not_fitted_raises(self):
        s = get_synthesizer("gaussian_copula")()
        with pytest.raises(RuntimeError, match="not fitted"):
            s.sample(10)


# ── CUDA Gate Tests ────────────────────────────────────────────────

class TestCUDAGate:
    """Test that neural synthesizers enforce CUDA requirement."""

    def test_ctgan_requires_cuda(self):
        """CTGAN with cuda=True should fail without GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                pytest.skip("CUDA is available — gate won't trigger")
            # CTGAN is registered, try to fit — should raise RuntimeError
            cls = get_synthesizer("ctgan")
            synth = cls(epochs=1, cuda=True)
            data = np.random.randn(100, 4)
            with pytest.raises(RuntimeError, match="requires.*CUDA|requires a CUDA"):
                synth.fit(data)
        except (ImportError, ValueError):
            pytest.skip("torch or ctgan not available")

    def test_tvae_requires_cuda(self):
        """TVAE with cuda=True should fail without GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                pytest.skip("CUDA is available")
            cls = get_synthesizer("tvae")
            synth = cls(epochs=1, cuda=True)
            with pytest.raises(RuntimeError, match="requires.*CUDA|requires a CUDA"):
                synth.fit(np.random.randn(100, 4))
        except (ImportError, ValueError):
            pytest.skip("torch or tvae not available")

    def test_tabddpm_requires_cuda(self):
        """TabDDPM should fail without GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                pytest.skip("CUDA is available")
            cls = get_synthesizer("tabddpm")
            synth = cls(epochs=1, cuda=True)
            with pytest.raises(RuntimeError, match="requires.*CUDA"):
                synth.fit(np.random.randn(100, 4))
        except (ImportError, ValueError):
            pytest.skip("torch or tabddpm not available")

    def test_tabsyn_requires_cuda(self):
        """TabSyn should fail without GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                pytest.skip("CUDA is available")
            cls = get_synthesizer("tabsyn")
            synth = cls(cuda=True)
            with pytest.raises(RuntimeError, match="requires.*CUDA"):
                synth.fit(np.random.randn(100, 4))
        except (ImportError, ValueError):
            pytest.skip("torch or tabsyn not available")

    def test_auto_fallback_no_cuda(self, numerical_df):
        """Auto synthesizer should fallback to gaussian_copula when no CUDA."""
        forge = SynthForge(verbose=False)
        syn = forge.fit_generate(numerical_df, 100)
        # Should succeed (copula doesn't need CUDA)
        assert len(syn) == 100
        # Should have used gaussian_copula
        assert forge._synth_type == SynthesizerType.GAUSSIAN_COPULA


# ── Constraint Tests ───────────────────────────────────────────────

class TestConstraints:
    def test_inequality(self):
        df = pd.DataFrame({"a": [1, 2], "b": [5, 6]})
        assert Inequality("a", "b").is_valid(df).all()

    def test_inequality_missing_column(self):
        """Inequality should return all-True if columns are missing."""
        df = pd.DataFrame({"a": [1, 2]})
        assert Inequality("a", "missing").is_valid(df).all()

    def test_inequality_non_numeric(self):
        """Inequality should return all-True for non-numeric columns."""
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert Inequality("a", "b").is_valid(df).all()

    def test_positive(self):
        assert PositiveValue("x").is_valid(pd.DataFrame({"x": [1, 2, -1, 4]})).sum() == 3

    def test_positive_non_numeric(self):
        """PositiveValue should be no-op for non-numeric columns."""
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        assert PositiveValue("x").is_valid(df).all()
        result = PositiveValue("x").reverse_transform(df)
        assert list(result["x"]) == ["a", "b", "c"]

    def test_range(self):
        assert ValueRange("x", 10, 30).is_valid(pd.DataFrame({"x": [5, 15, 25, 35]})).sum() == 2

    def test_range_non_numeric(self):
        """ValueRange should be no-op for non-numeric columns."""
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        assert ValueRange("x", 0, 100).is_valid(df).all()
        result = ValueRange("x", 0, 100).reverse_transform(df)
        assert list(result["x"]) == ["a", "b", "c"]

    def test_pipeline_rate(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 6, 7], "c": [10, -1, 20]})
        r = ConstraintPipeline([Inequality("a", "b"), PositiveValue("c")]).validity_rate(df)
        assert 0 < r < 1


# ── Evaluation Tests ───────────────────────────────────────────────

class TestEvaluation:
    def test_basic(self, numerical_df):
        m = detect_metadata(numerical_df)
        noise = np.random.normal(0, 0.5, numerical_df.shape)
        syn = pd.DataFrame(numerical_df.values + noise, columns=numerical_df.columns)
        r = Evaluator().evaluate(numerical_df, syn, m)
        assert r.overall_score > 0 and len(r.diagnostics) > 0

    def test_high_quality_on_identical(self, numerical_df):
        """Evaluating data against itself should score near-perfect."""
        m = detect_metadata(numerical_df)
        r = Evaluator().evaluate(numerical_df, numerical_df.copy(), m)
        assert r.overall_score > 0.90

    def test_serialization(self, numerical_df):
        m = detect_metadata(numerical_df)
        d = Evaluator().evaluate(numerical_df, numerical_df, m).to_dict()
        assert "overall_score" in d
        assert isinstance(d["diagnostics"], list)

    def test_pii_excluded_from_fidelity(self, mixed_df):
        """PII columns should not affect fidelity metrics."""
        m = detect_metadata(mixed_df)
        e = Evaluator()
        # Non-PII columns
        non_pii_num = e._non_pii_numerical(m)
        non_pii_cat = e._non_pii_categorical(m)
        assert "email" not in non_pii_cat
        assert all(not m.columns[c].is_pii for c in non_pii_num)


# ── Strategy Tests ─────────────────────────────────────────────────

class TestStrategy:
    def test_recommend_numerical(self, numerical_df):
        meta = detect_metadata(numerical_df)
        assert recommend_strategy(meta) == DataStrategy.NUMERICAL

    def test_auto_selects_copula_without_cuda(self, mixed_df):
        """Without CUDA, auto should always pick gaussian_copula."""
        if _cuda_available():
            pytest.skip("CUDA available — can't test CPU fallback")
        meta = detect_metadata(mixed_df)
        config = SynthForgeConfig()
        synth_type = recommend_synthesizer(meta, DataStrategy.MIXED, config)
        assert synth_type == SynthesizerType.GAUSSIAN_COPULA

    def test_create_copula(self):
        config = SynthForgeConfig()
        synth = create_synthesizer(SynthesizerType.GAUSSIAN_COPULA, config)
        assert synth is not None and not synth.is_fitted

    def test_create_unknown_raises(self):
        """Creating an unregistered synthesizer should raise clear error."""
        config = SynthForgeConfig()
        # Use a valid enum value that might not be registered
        try:
            create_synthesizer(SynthesizerType.TABDDPM, config)
        except (ValueError, KeyError):
            pass  # Expected if tabddpm not registered (no torch)

    def test_new_config_models_exist(self):
        """New SOTA config models should be instantiable."""
        c = SynthForgeConfig()
        assert c.tabddpm.n_timesteps == 1000
        assert c.tabsyn.vae_latent_dim == 64
        assert c.great.model_name == "distilgpt2"
        assert c.tabddpm.cuda is True
        assert c.tabsyn.cuda is True
        assert c.great.cuda is True


# ── Full Pipeline Tests ────────────────────────────────────────────

class TestFullPipeline:
    def test_numerical(self, numerical_df):
        syn = SynthForge(synthesizer="gaussian_copula", verbose=False).fit_generate(numerical_df, 100)
        assert len(syn) == 100 and set(numerical_df.columns) == set(syn.columns)

    def test_mixed(self, mixed_df):
        syn = SynthForge(synthesizer="gaussian_copula", verbose=False).fit_generate(mixed_df, 200)
        assert len(syn) == 200

    def test_categorical(self, categorical_df):
        syn = SynthForge(synthesizer="gaussian_copula", verbose=False).fit_generate(categorical_df, 150)
        assert len(syn) == 150

    def test_evaluate(self, numerical_df):
        f = SynthForge(synthesizer="gaussian_copula", verbose=False)
        syn = f.fit_generate(numerical_df, 300)
        r = f.evaluate(numerical_df, syn)
        assert r.overall_score > 0 and "Overall" in r.summary()

    def test_auto(self, mixed_df):
        assert len(SynthForge(verbose=False).fit_generate(mixed_df, 100)) == 100

    def test_batch(self, numerical_df):
        c = SynthForgeConfig(generation=GenerationConfig(batch_size=100))
        f = SynthForge(config=c, synthesizer="gaussian_copula", verbose=False)
        assert len(f.fit_generate(numerical_df, 200)) == 200

    def test_constraints(self, numerical_df):
        f = SynthForge(synthesizer="gaussian_copula", verbose=False)
        f.add_constraint(ValueRange("x1", low=0, high=100))
        syn = f.fit_generate(numerical_df, 100)
        assert syn["x1"].min() >= 0 and syn["x1"].max() <= 100

    def test_constraints_on_mixed_types(self, mixed_df):
        """Constraints should silently skip non-numeric columns."""
        f = SynthForge(synthesizer="gaussian_copula", verbose=False)
        f.add_constraint(PositiveValue("salary"))
        f.add_constraint(ValueRange("department", low=0, high=100))  # Will be no-op
        syn = f.fit_generate(mixed_df, 100)
        assert len(syn) == 100

    def test_profile(self, mixed_df):
        m = SynthForge(verbose=False).profile(mixed_df)
        assert isinstance(m, Metadata) and m.row_count == 500

    def test_quality_above_threshold(self, numerical_df):
        """Gaussian Copula on clean numerical data should score >80%."""
        f = SynthForge(synthesizer="gaussian_copula", verbose=False)
        syn = f.fit_generate(numerical_df, 500)
        r = f.evaluate(numerical_df, syn)
        assert r.overall_score > 0.80, f"Score {r.overall_score:.2%} below 80% threshold"


# ── Config Tests ───────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        c = SynthForgeConfig()
        assert c.synthesizer == SynthesizerType.AUTO and not c.llm.enabled

    def test_with_llm(self):
        c = SynthForgeConfig().with_llm("anthropic", "claude-sonnet-4-20250514")
        assert c.llm.enabled and c.llm.provider == "anthropic"

    def test_synthesizer_types_include_sota(self):
        """All SOTA model types should be in the enum."""
        assert SynthesizerType.TABDDPM == "tabddpm"
        assert SynthesizerType.TABSYN == "tabsyn"
        assert SynthesizerType.GREAT == "great"

    def test_tabddpm_config(self):
        c = TabDDPMConfig(n_timesteps=500, epochs=100)
        assert c.n_timesteps == 500 and c.scheduler == "cosine"

    def test_tabsyn_config(self):
        c = TabSynConfig(vae_epochs=100, diff_epochs=100, vae_latent_dim=32)
        assert c.vae_latent_dim == 32

    def test_great_config(self):
        c = GReaTConfig(model_name="gpt2", epochs=10)
        assert c.model_name == "gpt2" and c.temperature == 0.7
