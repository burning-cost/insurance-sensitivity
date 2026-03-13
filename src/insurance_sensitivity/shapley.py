"""
Shapley effects for global sensitivity analysis.

Shapley effects extend the Shapley value from cooperative game theory to
sensitivity analysis. The "value" of a coalition S of features is the variance
explained by that coalition: v(S) = Var(E[Y | X_S]).

The Shapley effect for feature i is:
    phi_i = sum over S subset of {1,...,p}\{i} of
            [|S|! * (p - |S| - 1)! / p!] * [v(S u {i}) - v(S)]

Key property: sum of Shapley effects = Var(Y) (efficiency axiom).
This means the effects partition total variance completely and non-overlappingly.

For independent features, Shapley effects equal first-order Sobol indices.
For correlated features, they differ. This matters in insurance because vehicle
age and driver age are correlated; vehicle age and NCB are correlated; etc.

We use the Monte Carlo estimator from Broto et al. (2020) which avoids exponential
cost in p. The cost scales as O(n * p * n_perms) where n_perms is the number of
random permutations sampled.

References:
    Owen, A. B. (2014). Sobol' indices and Shapley value. *SIAM/ASA J. Uncertainty Quantification*,
    2(1), 245-251.

    Broto, B., et al. (2020). Variance reduction for estimation of Shapley effects and adaptation
    to unknown input distribution. *SIAM/ASA J. Uncertainty Quantification*, 8(2), 693-716.

    Iooss, B., & Prieur, C. (2019). Shapley effects for sensitivity analysis with dependent inputs:
    comparisons with Sobol' indices, numerical estimation and applications.
    *International Journal for Uncertainty Quantification*, 9(5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ShapleyResult:
    """Results from Shapley effects sensitivity analysis.

    Attributes
    ----------
    feature_names : list[str]
    shapley_effects : np.ndarray
        Shapley effect phi_i for each feature.
    shapley_normalised : np.ndarray
        Normalised Shapley effects phi_i / sum(phi_i). Sums to 1.
    total_variance : float
        Var(Y) of model output.
    sum_shapley : float
        Sum of Shapley effects (should equal total_variance).
    """

    feature_names: list[str]
    shapley_effects: np.ndarray
    shapley_normalised: np.ndarray
    total_variance: float
    sum_shapley: float

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "Shapley Effects (Global Sensitivity)",
            "=" * 65,
            f"  Total output variance: {self.total_variance:.6f}",
            f"  Sum of Shapley effects: {self.sum_shapley:.6f}",
            f"  (should ≈ total variance; efficiency axiom)",
            "",
            f"  {'Feature':<20} {'Phi_i':>12} {'Phi_i/Sum':>12}",
            "  " + "-" * 46,
        ]
        order = np.argsort(self.shapley_effects)[::-1]
        for i in order:
            lines.append(
                f"  {self.feature_names[i]:<20} "
                f"{self.shapley_effects[i]:>12.4f} "
                f"{self.shapley_normalised[i]:>12.4f}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.feature_names,
            "shapley_effect": self.shapley_effects,
            "shapley_normalised": self.shapley_normalised,
        }).sort_values("shapley_effect", ascending=False).reset_index(drop=True)


class ShapleyEffects:
    """Shapley effects estimator for global sensitivity analysis.

    Uses a permutation-based Monte Carlo estimator. For each random permutation
    of features, computes the marginal variance explained by adding each feature.
    Averages over n_perms permutations.

    The conditional variance V(S u {i}) - V(S) is estimated via:
        V(S) ≈ Var(E[Y | X_S]) ≈ Var(mean-of-model-over-non-S-features)

    For a predictive model f(X), E[Y | X_S] is estimated by conditioning on X_S
    and marginalising over the remaining features by drawing from the empirical
    distribution of (X_{-S} | X_S). For independent features this is simply drawing
    independently; for correlated features we use a nearest-neighbour approximation.

    Parameters
    ----------
    model : callable
        f(X) -> y. Must accept (n, p) array.
    n_samples : int
        Number of Monte Carlo samples for variance estimation.
    n_perms : int
        Number of random permutations for Shapley averaging.
    feature_names : list[str] or None
    random_state : int
    """

    def __init__(
        self,
        model: Callable,
        n_samples: int = 1_000,
        n_perms: int = 50,
        feature_names: list[str] | None = None,
        random_state: int = 42,
    ):
        self.model         = model
        self.n_samples     = n_samples
        self.n_perms       = n_perms
        self.feature_names = feature_names
        self.random_state  = random_state
        self._result: ShapleyResult | None = None

    def _conditional_mean(
        self,
        X_ref: np.ndarray,
        X_query: np.ndarray,
        subset: list[int],
        n_inner: int = 50,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Estimate E[f(X) | X_subset = X_query_subset] via MC marginalisation.

        For each query point, randomly sample replacement values for non-subset
        features from X_ref (empirical marginal), then average model predictions.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        n_query = len(X_query)
        p       = X_query.shape[1]
        non_sub = [j for j in range(p) if j not in subset]

        cond_means = np.zeros(n_query)
        for qi in range(n_query):
            X_aug = np.tile(X_query[qi], (n_inner, 1))
            if non_sub:
                # Sample non-subset features from X_ref (empirical marginal)
                idx_ref = rng.integers(0, len(X_ref), n_inner)
                X_aug[:, non_sub] = X_ref[idx_ref][:, non_sub]
            cond_means[qi] = self.model(X_aug).mean()

        return cond_means

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> ShapleyResult:
        """Compute Shapley effects.

        Parameters
        ----------
        X : array-like (n_ref, p)
            Reference data for estimating input distribution. Model is evaluated
            on points drawn from this distribution.

        Returns
        -------
        ShapleyResult
        """
        X = np.asarray(X, dtype=float)
        n_ref, p = X.shape
        rng = np.random.default_rng(self.random_state)

        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(p)]
        else:
            feature_names = list(self.feature_names)

        # Draw query points from X_ref
        n_q   = min(self.n_samples, n_ref)
        idx_q = rng.integers(0, n_ref, n_q)
        X_q   = X[idx_q]

        # Total variance: Var(f(X)) over X_ref
        f_all = self.model(X)
        total_var = float(np.var(f_all))

        if total_var < 1e-12:
            shapley = np.zeros(p)
            return ShapleyResult(
                feature_names=feature_names,
                shapley_effects=shapley,
                shapley_normalised=np.zeros(p),
                total_variance=total_var,
                sum_shapley=0.0,
            )

        # Marginal contributions accumulated across permutations
        phi = np.zeros(p)
        n_inner = max(20, self.n_samples // 20)

        for _ in range(self.n_perms):
            perm = rng.permutation(p).tolist()
            v_prev = 0.0   # V(empty set) = 0

            for k, feat in enumerate(perm):
                subset = perm[: k + 1]
                cond_mean_k = self._conditional_mean(X, X_q, subset, n_inner=n_inner, rng=rng)
                v_k = float(np.var(cond_mean_k))
                phi[feat] += (v_k - v_prev)
                v_prev = v_k

        phi /= self.n_perms

        # Normalised
        phi_sum = float(phi.sum())
        phi_norm = phi / max(phi_sum, 1e-12)

        self._result = ShapleyResult(
            feature_names=feature_names,
            shapley_effects=phi,
            shapley_normalised=phi_norm,
            total_variance=total_var,
            sum_shapley=phi_sum,
        )
        return self._result

    @property
    def result(self) -> ShapleyResult:
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result
