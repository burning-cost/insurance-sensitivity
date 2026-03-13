"""Tests for insurance-sensitivity."""

import numpy as np
import pytest
from insurance_sensitivity import SobolAnalysis, ShapleyEffects


def linear_model(X):
    """Simple linear model: f(x) = 3*x0 + 2*x1 + x2."""
    return 3 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2]


def interaction_model(X):
    """Model with interaction: f(x) = x0*x1 + x2."""
    return X[:, 0] * X[:, 1] + X[:, 2]


def make_uniform_bounds(p, lo=-1, hi=1):
    return [(lo, hi)] * p


def test_sobol_sum_first_order_no_interactions():
    """For linear model with independent uniform inputs, S_i sum should be near 1."""
    sa = SobolAnalysis(model=linear_model, n_samples=2000, random_state=42)
    result = sa.fit(X_bounds=make_uniform_bounds(3))
    # For linear model, sum of first-order = 1 (no interactions)
    assert 0.85 < result.sum_first_order <= 1.1, (
        f"Sum S_i = {result.sum_first_order:.3f}, expected near 1.0"
    )


def test_sobol_ranks_features_correctly():
    """x0 (coef=3) should have higher Sobol index than x1 (coef=2) > x2 (coef=1)."""
    sa = SobolAnalysis(model=linear_model, n_samples=3000, random_state=42)
    result = sa.fit(X_bounds=make_uniform_bounds(3))
    s = result.first_order
    assert s[0] > s[1] > s[2], f"Expected S[0]>S[1]>S[2], got {s}"


def test_sobol_detects_interaction():
    """Interaction model: total-order S should be > first-order S for x0, x1."""
    sa = SobolAnalysis(model=interaction_model, n_samples=2000, random_state=42)
    result = sa.fit(X_bounds=make_uniform_bounds(3))
    # x0 and x1 interact, so T_i > S_i for both
    assert result.total_order[0] > result.first_order[0], "T_0 should > S_0 for interaction"
    assert result.total_order[1] > result.first_order[1], "T_1 should > S_1 for interaction"


def test_shapley_efficiency():
    """Sum of Shapley effects should approximately equal total variance."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.uniform(-1, 1, (n, 3))
    se = ShapleyEffects(model=linear_model, n_samples=100, n_perms=20, random_state=42)
    result = se.fit(X)
    total_var = float(np.var(linear_model(X)))
    # Efficiency: sum of Shapley effects ≈ Var(Y)
    assert abs(result.sum_shapley - total_var) / max(total_var, 1e-8) < 0.3, (
        f"Shapley sum {result.sum_shapley:.4f} far from Var(Y) {total_var:.4f}"
    )


def test_shapley_ranks_features():
    """Shapley effects should rank x0 (coef=3) highest."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (1000, 3))
    se = ShapleyEffects(model=linear_model, n_samples=200, n_perms=30, random_state=42)
    result = se.fit(X)
    assert result.shapley_effects[0] > result.shapley_effects[2], (
        "x0 (coef=3) should have higher Shapley effect than x2 (coef=1)"
    )


def test_sobol_to_dataframe():
    sa = SobolAnalysis(model=linear_model, n_samples=500, feature_names=["x0","x1","x2"])
    result = sa.fit(X_bounds=make_uniform_bounds(3))
    df = result.to_dataframe()
    assert set(df.columns) >= {"feature", "sobol_first", "sobol_total"}
    assert len(df) == 3


def test_shapley_to_dataframe():
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (200, 2))
    se = ShapleyEffects(model=lambda X: X[:, 0] + X[:, 1], n_samples=50, n_perms=10)
    result = se.fit(X)
    df = result.to_dataframe()
    assert "shapley_effect" in df.columns
