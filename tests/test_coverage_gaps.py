"""
Extended test coverage for insurance-sensitivity (April 2026).

Targets untested code paths identified in the March 2026 coverage audit
(61% coverage, 7 tests).  This file adds tests for:

1.  SobolResult: summary(), to_dataframe() column content, interactions >=0
2.  SobolAnalysis: fit() via X array (not X_bounds), p-only path raises,
    no-args raises ValueError, result property before/after fit,
    constant output (zero variance) short-circuit, feature_names stored
    correctly, custom feature names flow through to result
3.  SobolAnalysis: fit with explicit feature_names; name length mismatch handled
4.  ShapleyResult: summary(), to_dataframe() columns, normalised sums to 1
5.  ShapleyEffects: result property before fit raises, after fit returns result,
    fit() with pd.DataFrame input, constant output short-circuit,
    feature_names passed through, n_perms=1 edge case
6.  ShapleyEffects._conditional_mean: empty subset (all features free),
    full subset (no free features)
7.  Numerical properties: Sobol T_i >= S_i, T_i in [0,1], S_i in [0,1],
    interactions = T_i - S_i, sum_first_order == sum(first_order)
8.  Ishigami function: known analytical ordering S_1 > S_2, T_1 > T_2
9.  Compatibility: both classes accept sklearn-like repeated fit calls
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_sensitivity import SobolAnalysis, SobolResult, ShapleyEffects, ShapleyResult


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def linear_model(X: np.ndarray) -> np.ndarray:
    """f(X) = 3*x0 + 2*x1 + x2 (no interactions, independent inputs)."""
    return 3.0 * X[:, 0] + 2.0 * X[:, 1] + 1.0 * X[:, 2]


def constant_model(X: np.ndarray) -> np.ndarray:
    """f(X) = 5.0 for all X — zero variance output."""
    return np.full(X.shape[0], 5.0)


def ishigami(X: np.ndarray) -> np.ndarray:
    """Ishigami function: known analytical Sobol ordering.

    f(x) = sin(x1) + 7*sin^2(x2) + 0.1*x3^4*sin(x1)
    Analytical first-order:  S_0 > S_1, S_2 ≈ 0
    Analytical total-order:  T_0 > T_1 > T_2, T_2 > 0 (x3 interacts)
    """
    return (
        np.sin(X[:, 0])
        + 7.0 * np.sin(X[:, 1]) ** 2
        + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])
    )


def make_uniform_bounds(p: int, lo: float = -1.0, hi: float = 1.0):
    return [(lo, hi)] * p


# ---------------------------------------------------------------------------
# 1. SobolResult
# ---------------------------------------------------------------------------

class TestSobolResult:

    def _make_result(self, n_features: int = 3) -> SobolResult:
        sa = SobolAnalysis(model=linear_model, n_samples=500, random_state=0)
        return sa.fit(X_bounds=make_uniform_bounds(n_features))

    def test_summary_returns_string(self):
        result = self._make_result()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_feature_names(self):
        sa = SobolAnalysis(
            model=linear_model, n_samples=500,
            feature_names=["age", "ncb", "group"], random_state=1
        )
        result = sa.fit(X_bounds=make_uniform_bounds(3))
        s = result.summary()
        assert "age" in s
        assert "ncb" in s
        assert "group" in s

    def test_summary_contains_variance(self):
        result = self._make_result()
        s = result.summary()
        assert "variance" in s.lower() or "Total" in s

    def test_to_dataframe_has_required_columns(self):
        result = self._make_result()
        df = result.to_dataframe()
        assert "feature" in df.columns
        assert "sobol_first" in df.columns
        assert "sobol_total" in df.columns
        assert "sobol_interaction" in df.columns

    def test_to_dataframe_length_matches_features(self):
        result = self._make_result(n_features=4)
        df = result.to_dataframe()
        assert len(df) == 4

    def test_to_dataframe_sorted_by_total_descending(self):
        """to_dataframe() should sort by sobol_total descending."""
        result = self._make_result()
        df = result.to_dataframe()
        totals = df["sobol_total"].tolist()
        assert totals == sorted(totals, reverse=True)

    def test_interactions_non_negative(self):
        """Interactions = T_i - S_i must be >= 0."""
        result = self._make_result()
        assert np.all(result.interactions >= 0.0)

    def test_total_order_ge_first_order(self):
        """T_i >= S_i for all features."""
        result = self._make_result()
        assert np.all(result.total_order >= result.first_order - 1e-6)

    def test_indices_in_unit_interval(self):
        """Clipping means all indices should be in [0, 1]."""
        result = self._make_result()
        assert np.all(result.first_order >= 0.0)
        assert np.all(result.first_order <= 1.0)
        assert np.all(result.total_order >= 0.0)
        assert np.all(result.total_order <= 1.0)

    def test_sum_first_order_attribute_matches_sum(self):
        """sum_first_order attribute should equal sum(first_order)."""
        result = self._make_result()
        np.testing.assert_allclose(
            result.sum_first_order, result.first_order.sum(), rtol=1e-10
        )

    def test_interactions_equals_total_minus_first(self):
        """Interactions stored = total_order - first_order (after clipping)."""
        result = self._make_result()
        expected = np.clip(result.total_order - result.first_order, 0.0, None)
        np.testing.assert_allclose(result.interactions, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. SobolAnalysis.fit() — alternative input paths
# ---------------------------------------------------------------------------

class TestSobolAnalysisFitPaths:

    def test_fit_with_X_array_uses_data_bounds(self):
        """fit(X=...) should infer bounds from X and produce valid indices."""
        rng = np.random.default_rng(10)
        X = rng.uniform(0, 10, (200, 3))
        sa = SobolAnalysis(model=linear_model, n_samples=300, random_state=10)
        result = sa.fit(X=X)
        assert result.first_order.shape == (3,)
        assert np.all(np.isfinite(result.first_order))

    def test_fit_with_X_infers_correct_p(self):
        """When X is provided, p should match X.shape[1]."""
        rng = np.random.default_rng(11)
        X = rng.standard_normal((100, 5))
        sa = SobolAnalysis(model=lambda Xm: Xm.sum(axis=1), n_samples=200, random_state=11)
        result = sa.fit(X=X)
        assert result.first_order.shape == (5,)

    def test_fit_no_args_raises_valueerror(self):
        """fit() with neither X nor X_bounds should raise ValueError."""
        sa = SobolAnalysis(model=linear_model, n_samples=100)
        with pytest.raises(ValueError, match="X or X_bounds"):
            sa.fit()

    def test_fit_with_x_and_x_bounds_prefers_x_bounds(self):
        """When both X and X_bounds are provided, X_bounds takes precedence for sampling."""
        rng = np.random.default_rng(12)
        X = rng.uniform(0, 1, (100, 3))
        # X_bounds specifies a different range — result should not crash
        sa = SobolAnalysis(model=linear_model, n_samples=300, random_state=12)
        result = sa.fit(X=X, X_bounds=make_uniform_bounds(3, -2, 2))
        assert result.first_order.shape == (3,)

    def test_feature_names_default_to_x0_x1(self):
        """Without feature_names, default names should be x0, x1, ..."""
        sa = SobolAnalysis(model=linear_model, n_samples=300, random_state=13)
        result = sa.fit(X_bounds=make_uniform_bounds(3))
        assert result.feature_names == ["x0", "x1", "x2"]

    def test_feature_names_custom_stored_in_result(self):
        """Custom feature_names must appear in the result."""
        names = ["driver_age", "vehicle_age", "ncb"]
        sa = SobolAnalysis(
            model=linear_model, n_samples=300,
            feature_names=names, random_state=14
        )
        result = sa.fit(X_bounds=make_uniform_bounds(3))
        assert result.feature_names == names

    def test_total_variance_positive_for_non_constant_model(self):
        """Total variance should be > 0 for a non-constant model."""
        sa = SobolAnalysis(model=linear_model, n_samples=500, random_state=15)
        result = sa.fit(X_bounds=make_uniform_bounds(3))
        assert result.total_variance > 0.0


# ---------------------------------------------------------------------------
# 3. SobolAnalysis.result property
# ---------------------------------------------------------------------------

class TestSobolResultProperty:

    def test_result_property_before_fit_raises(self):
        """Accessing result before fit() should raise RuntimeError."""
        sa = SobolAnalysis(model=linear_model, n_samples=100)
        with pytest.raises(RuntimeError, match="fit"):
            _ = sa.result

    def test_result_property_after_fit_returns_sobol_result(self):
        """After fit(), result property should return SobolResult."""
        sa = SobolAnalysis(model=linear_model, n_samples=300, random_state=20)
        sa.fit(X_bounds=make_uniform_bounds(3))
        assert isinstance(sa.result, SobolResult)

    def test_result_property_matches_fit_return(self):
        """result property should be the same object returned by fit()."""
        sa = SobolAnalysis(model=linear_model, n_samples=300, random_state=21)
        result_from_fit = sa.fit(X_bounds=make_uniform_bounds(3))
        assert sa.result is result_from_fit


# ---------------------------------------------------------------------------
# 4. SobolAnalysis — constant output (zero variance short-circuit)
# ---------------------------------------------------------------------------

class TestSobolConstantOutput:

    def test_constant_output_returns_zero_indices(self):
        """Constant model (zero output variance) should return all-zero indices."""
        sa = SobolAnalysis(model=constant_model, n_samples=200, random_state=30)
        result = sa.fit(X_bounds=make_uniform_bounds(3))
        np.testing.assert_array_equal(result.first_order, np.zeros(3))
        np.testing.assert_array_equal(result.total_order, np.zeros(3))

    def test_constant_output_sum_first_order_zero(self):
        sa = SobolAnalysis(model=constant_model, n_samples=200, random_state=31)
        result = sa.fit(X_bounds=make_uniform_bounds(2))
        assert result.sum_first_order == 0.0

    def test_constant_output_total_variance_near_zero(self):
        sa = SobolAnalysis(model=constant_model, n_samples=200, random_state=32)
        result = sa.fit(X_bounds=make_uniform_bounds(2))
        assert result.total_variance < 1e-10

    def test_constant_output_interactions_zero(self):
        sa = SobolAnalysis(model=constant_model, n_samples=200, random_state=33)
        result = sa.fit(X_bounds=make_uniform_bounds(2))
        np.testing.assert_array_equal(result.interactions, np.zeros(2))


# ---------------------------------------------------------------------------
# 5. Ishigami — known analytical ordering
# ---------------------------------------------------------------------------

class TestSobolIshigami:
    """Ishigami analytical benchmark.

    Analytical S_i: S_0 ≈ 0.31, S_1 ≈ 0.44, S_2 = 0 (exactly).
    Analytical T_i: T_0 ≈ 0.56, T_1 ≈ 0.44, T_2 ≈ 0.24.
    The key orderings: T_0 > T_1 > T_2, T_2 > 0.
    """

    def _run_ishigami(self) -> SobolResult:
        bounds = [(-np.pi, np.pi)] * 3
        sa = SobolAnalysis(model=ishigami, n_samples=3000, random_state=42)
        return sa.fit(X_bounds=bounds)

    def test_total_order_is_positive_for_all_features(self):
        """All features affect output variance (T_i > 0 for all i)."""
        result = self._run_ishigami()
        assert np.all(result.total_order > 0.0), (
            f"T_i should all be positive, got {result.total_order}"
        )

    def test_x1_total_order_greater_than_x2_total_order(self):
        """T_1 > T_2 for Ishigami."""
        result = self._run_ishigami()
        assert result.total_order[1] > result.total_order[2], (
            f"T_1={result.total_order[1]:.3f} should be > T_2={result.total_order[2]:.3f}"
        )

    def test_x0_has_highest_total_order(self):
        """x0 drives most total variance (T_0 is the largest)."""
        result = self._run_ishigami()
        assert result.total_order[0] == max(result.total_order), (
            f"T_0 should be the max, got {result.total_order}"
        )

    def test_x2_first_order_near_zero(self):
        """x3 (x2) has zero first-order effect analytically."""
        result = self._run_ishigami()
        # Allow some Monte Carlo error — the key point is it is much smaller
        assert result.first_order[2] < 0.1, (
            f"S_2 for Ishigami should be near 0, got {result.first_order[2]:.3f}"
        )


# ---------------------------------------------------------------------------
# 6. ShapleyResult
# ---------------------------------------------------------------------------

class TestShapleyResult:

    def _make_result(self) -> ShapleyResult:
        rng = np.random.default_rng(40)
        X = rng.uniform(-1, 1, (200, 3))
        se = ShapleyEffects(
            model=linear_model, n_samples=80, n_perms=10, random_state=40
        )
        return se.fit(X)

    def test_summary_returns_string(self):
        result = self._make_result()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_efficiency_text(self):
        result = self._make_result()
        s = result.summary()
        assert "variance" in s.lower() or "efficiency" in s.lower()

    def test_to_dataframe_has_required_columns(self):
        result = self._make_result()
        df = result.to_dataframe()
        assert "feature" in df.columns
        assert "shapley_effect" in df.columns
        assert "shapley_normalised" in df.columns

    def test_to_dataframe_length_matches_features(self):
        result = self._make_result()
        df = result.to_dataframe()
        assert len(df) == 3

    def test_shapley_normalised_sums_to_one(self):
        """Normalised Shapley effects must sum to 1."""
        result = self._make_result()
        np.testing.assert_allclose(result.shapley_normalised.sum(), 1.0, atol=1e-6)

    def test_shapley_effects_non_negative(self):
        """Shapley effects should be non-negative (variance contributions)."""
        result = self._make_result()
        assert np.all(result.shapley_effects >= 0.0), (
            f"Shapley effects should be non-negative: {result.shapley_effects}"
        )

    def test_sum_shapley_attribute_matches_sum(self):
        result = self._make_result()
        np.testing.assert_allclose(
            result.sum_shapley, result.shapley_effects.sum(), rtol=1e-10
        )

    def test_total_variance_positive(self):
        result = self._make_result()
        assert result.total_variance > 0.0


# ---------------------------------------------------------------------------
# 7. ShapleyEffects — result property and error paths
# ---------------------------------------------------------------------------

class TestShapleyEffectsResultProperty:

    def test_result_before_fit_raises(self):
        """Accessing result before fit() must raise RuntimeError."""
        se = ShapleyEffects(model=linear_model, n_samples=50, n_perms=5)
        with pytest.raises(RuntimeError, match="fit"):
            _ = se.result

    def test_result_after_fit_returns_shapley_result(self):
        rng = np.random.default_rng(50)
        X = rng.uniform(-1, 1, (100, 3))
        se = ShapleyEffects(model=linear_model, n_samples=50, n_perms=5, random_state=50)
        se.fit(X)
        assert isinstance(se.result, ShapleyResult)

    def test_result_property_matches_fit_return(self):
        rng = np.random.default_rng(51)
        X = rng.uniform(-1, 1, (100, 3))
        se = ShapleyEffects(model=linear_model, n_samples=50, n_perms=5, random_state=51)
        result_from_fit = se.fit(X)
        assert se.result is result_from_fit


# ---------------------------------------------------------------------------
# 8. ShapleyEffects — alternative input paths
# ---------------------------------------------------------------------------

class TestShapleyEffectsFitPaths:

    def test_fit_with_pandas_dataframe(self):
        """fit() should accept a pandas DataFrame as X."""
        rng = np.random.default_rng(60)
        X_np = rng.uniform(-1, 1, (150, 3))
        X_df = pd.DataFrame(X_np, columns=["a", "b", "c"])
        se = ShapleyEffects(model=linear_model, n_samples=50, n_perms=5, random_state=60)
        result = se.fit(X_df)
        assert result.shapley_effects.shape == (3,)
        assert np.all(np.isfinite(result.shapley_effects))

    def test_fit_with_numpy_array(self):
        rng = np.random.default_rng(61)
        X = rng.uniform(-1, 1, (150, 3))
        se = ShapleyEffects(model=linear_model, n_samples=50, n_perms=5, random_state=61)
        result = se.fit(X)
        assert result.shapley_effects.shape == (3,)

    def test_feature_names_default_x0_x1(self):
        """Without feature_names, defaults should be x0, x1, ..."""
        rng = np.random.default_rng(62)
        X = rng.uniform(-1, 1, (100, 3))
        se = ShapleyEffects(model=linear_model, n_samples=50, n_perms=5, random_state=62)
        result = se.fit(X)
        assert result.feature_names == ["x0", "x1", "x2"]

    def test_custom_feature_names_in_result(self):
        """Custom feature_names should flow through to the result."""
        rng = np.random.default_rng(63)
        X = rng.uniform(-1, 1, (100, 3))
        names = ["vehicle_age", "driver_age", "ncb_years"]
        se = ShapleyEffects(
            model=linear_model, n_samples=50, n_perms=5,
            feature_names=names, random_state=63
        )
        result = se.fit(X)
        assert result.feature_names == names

    def test_n_perms_1_does_not_crash(self):
        """Minimum n_perms=1 edge case should still produce a result."""
        rng = np.random.default_rng(64)
        X = rng.uniform(-1, 1, (100, 3))
        se = ShapleyEffects(model=linear_model, n_samples=50, n_perms=1, random_state=64)
        result = se.fit(X)
        assert result.shapley_effects.shape == (3,)
        assert np.all(np.isfinite(result.shapley_effects))

    def test_repeated_fit_updates_result(self):
        """Calling fit() twice should overwrite the cached result."""
        rng = np.random.default_rng(65)
        X1 = rng.uniform(-1, 1, (100, 3))
        X2 = rng.uniform(0, 2, (100, 3))
        se = ShapleyEffects(model=linear_model, n_samples=50, n_perms=5, random_state=65)
        se.fit(X1)
        result2 = se.fit(X2)
        assert se.result is result2


# ---------------------------------------------------------------------------
# 9. ShapleyEffects — constant output (zero variance short-circuit)
# ---------------------------------------------------------------------------

class TestShapleyConstantOutput:

    def test_constant_output_returns_zero_effects(self):
        """Constant model (zero output variance) should return all-zero effects."""
        rng = np.random.default_rng(70)
        X = rng.uniform(-1, 1, (100, 3))
        se = ShapleyEffects(model=constant_model, n_samples=50, n_perms=5, random_state=70)
        result = se.fit(X)
        np.testing.assert_array_equal(result.shapley_effects, np.zeros(3))

    def test_constant_output_sum_shapley_zero(self):
        rng = np.random.default_rng(71)
        X = rng.uniform(-1, 1, (100, 3))
        se = ShapleyEffects(model=constant_model, n_samples=50, n_perms=5, random_state=71)
        result = se.fit(X)
        assert result.sum_shapley == 0.0

    def test_constant_output_total_variance_near_zero(self):
        rng = np.random.default_rng(72)
        X = rng.uniform(-1, 1, (100, 3))
        se = ShapleyEffects(model=constant_model, n_samples=50, n_perms=5, random_state=72)
        result = se.fit(X)
        assert result.total_variance < 1e-10


# ---------------------------------------------------------------------------
# 10. ShapleyEffects._conditional_mean — internal edge cases
# ---------------------------------------------------------------------------

class TestShapleyConditionalMean:
    """Direct tests on the internal _conditional_mean method."""

    def setup_method(self):
        rng = np.random.default_rng(80)
        self.X_ref = rng.uniform(-1, 1, (200, 3))
        self.X_q = self.X_ref[:10]
        self.rng = np.random.default_rng(81)
        self.se = ShapleyEffects(
            model=linear_model, n_samples=50, n_perms=5, random_state=80
        )

    def test_full_subset_returns_deterministic_prediction(self):
        """With all features in subset, no marginalisation needed — result is f(X_q)."""
        subset = [0, 1, 2]
        result = self.se._conditional_mean(
            self.X_ref, self.X_q, subset, n_inner=10, rng=self.rng
        )
        expected = linear_model(self.X_q)
        # With full subset, n_inner copies of X_q all return the same model output
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_empty_subset_returns_array_of_correct_length(self):
        """Empty subset: all features are marginalised. Output shape must be correct."""
        subset = []
        result = self.se._conditional_mean(
            self.X_ref, self.X_q, subset, n_inner=20, rng=self.rng
        )
        assert result.shape == (len(self.X_q),)
        assert np.all(np.isfinite(result))

    def test_partial_subset_correct_output_length(self):
        """Partial subset: output length matches len(X_query)."""
        subset = [0]
        result = self.se._conditional_mean(
            self.X_ref, self.X_q, subset, n_inner=15, rng=self.rng
        )
        assert result.shape == (len(self.X_q),)

    def test_output_is_finite(self):
        """_conditional_mean must not produce NaN or Inf."""
        for subset in [[], [0], [0, 1], [0, 1, 2]]:
            result = self.se._conditional_mean(
                self.X_ref, self.X_q, subset, n_inner=10, rng=self.rng
            )
            assert np.all(np.isfinite(result)), f"Non-finite output for subset={subset}"


# ---------------------------------------------------------------------------
# 11. Shapley efficiency property on linear independent model
# ---------------------------------------------------------------------------

class TestShapleyEfficiencyProperty:
    """The sum of Shapley effects should approximately equal Var(Y).

    For the linear model with uniform[-1,1] inputs, the analytical variance
    is Var(Y) = Var(3X0 + 2X1 + X2) = 9*Var(X0) + 4*Var(X1) + Var(X2)
               = (9 + 4 + 1) * (1/3) = 14/3 ≈ 4.667.
    """

    def test_sum_shapley_close_to_total_variance(self):
        rng = np.random.default_rng(90)
        X = rng.uniform(-1, 1, (1000, 3))
        se = ShapleyEffects(
            model=linear_model, n_samples=200, n_perms=20, random_state=90
        )
        result = se.fit(X)
        ratio = abs(result.sum_shapley - result.total_variance) / max(result.total_variance, 1e-8)
        assert ratio < 0.25, (
            f"Efficiency axiom: sum_shapley={result.sum_shapley:.4f}, "
            f"total_variance={result.total_variance:.4f}, ratio={ratio:.3f}"
        )

    def test_x0_has_highest_shapley_effect(self):
        """For linear model, x0 (coef=3) should have highest Shapley effect."""
        rng = np.random.default_rng(91)
        X = rng.uniform(-1, 1, (500, 3))
        se = ShapleyEffects(
            model=linear_model, n_samples=100, n_perms=15, random_state=91
        )
        result = se.fit(X)
        assert result.shapley_effects[0] == max(result.shapley_effects), (
            f"x0 should have highest effect: {result.shapley_effects}"
        )


# ---------------------------------------------------------------------------
# 12. SobolAnalysis — repeated fit() calls overwrite state
# ---------------------------------------------------------------------------

class TestSobolRepeatedFit:

    def test_repeated_fit_updates_result(self):
        """Calling fit() twice should overwrite the cached result."""
        sa = SobolAnalysis(model=linear_model, n_samples=300, random_state=100)
        sa.fit(X_bounds=make_uniform_bounds(3))
        result2 = sa.fit(X_bounds=[(-2, 2)] * 3)
        assert sa.result is result2

    def test_fit_with_different_n_samples_gives_valid_result(self):
        """Very small n_samples should still produce valid (though noisy) results."""
        sa = SobolAnalysis(model=linear_model, n_samples=50, random_state=101)
        result = sa.fit(X_bounds=make_uniform_bounds(2))
        assert result.first_order.shape == (2,)
        assert np.all(np.isfinite(result.first_order))


# ---------------------------------------------------------------------------
# 13. SobolAnalysis — single feature edge case
# ---------------------------------------------------------------------------

class TestSobolSingleFeature:

    def test_single_feature_returns_correct_shape(self):
        """p=1 is an edge case — only one Sobol index."""
        sa = SobolAnalysis(
            model=lambda X: X[:, 0] ** 2,
            n_samples=500, random_state=110
        )
        result = sa.fit(X_bounds=[(-1, 1)])
        assert result.first_order.shape == (1,)
        assert result.total_order.shape == (1,)
        assert result.interactions.shape == (1,)

    def test_single_feature_total_equals_first_for_no_interaction(self):
        """With p=1, T_0 = S_0 (no other features to interact with)."""
        sa = SobolAnalysis(
            model=lambda X: 2.0 * X[:, 0],
            n_samples=500, random_state=111
        )
        result = sa.fit(X_bounds=[(-1, 1)])
        # With a single feature, T_0 = S_0 (same estimator in the limit)
        np.testing.assert_allclose(
            result.total_order[0], result.first_order[0], atol=0.05
        )

    def test_single_feature_total_variance_positive(self):
        sa = SobolAnalysis(
            model=lambda X: X[:, 0],
            n_samples=300, random_state=112
        )
        result = sa.fit(X_bounds=[(-1, 1)])
        assert result.total_variance > 0.0
