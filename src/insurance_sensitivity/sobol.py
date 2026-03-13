"""
Sobol sensitivity indices via the Saltelli (2010) estimator.

The Saltelli estimator uses two independent sample matrices A and B (n x p),
and p "re-mixed" matrices A_B^(i) (column i from B, all others from A) and
B_A^(i) (column i from A, all others from B).

First-order index:
    S_i = V_i / Var(Y)
    where V_i = (1/n) sum_j [f(B)_j * (f(A_B^(i))_j - f(A)_j)]
    (Saltelli 2010, Eq. 2)

Total-order index:
    T_i = (1/(2n)) sum_j [(f(A)_j - f(A_B^(i))_j)^2] / Var(Y)
    (Saltelli 2010, Eq. 4)

The interaction indices are T_i - S_i. The sum of first-order indices <= 1,
with equality only when there are no interactions.

References:
    Saltelli, A., et al. (2010). Variance based sensitivity analysis of model output.
    Design and estimator for the total sensitivity index. *Computer Physics Communications*,
    181(2), 259-270.

    Sobol', I. M. (1993). Sensitivity analysis for non-linear mathematical models.
    *Mathematical Modelling and Computational Experiments*, 1(4), 407-414.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class SobolResult:
    """Results from Sobol sensitivity analysis.

    Attributes
    ----------
    feature_names : list[str]
    first_order : np.ndarray
        First-order Sobol indices S_i.
    total_order : np.ndarray
        Total-order Sobol indices T_i.
    interactions : np.ndarray
        Interaction indices T_i - S_i.
    total_variance : float
        Var(Y) of model output.
    sum_first_order : float
        Sum of first-order indices (= 1 if no interactions).
    """

    feature_names: list[str]
    first_order: np.ndarray
    total_order: np.ndarray
    interactions: np.ndarray
    total_variance: float
    sum_first_order: float

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "Sobol Sensitivity Analysis",
            "=" * 65,
            f"  Total output variance: {self.total_variance:.6f}",
            f"  Sum of S_i (first-order): {self.sum_first_order:.4f}",
            f"  (< 1 indicates interactions; = 1 means no interactions)",
            "",
            f"  {'Feature':<20} {'S_i (1st)':>12} {'T_i (total)':>12} {'T_i-S_i (inter)':>16}",
            "  " + "-" * 62,
        ]
        order = np.argsort(self.total_order)[::-1]
        for i in order:
            lines.append(
                f"  {self.feature_names[i]:<20} {self.first_order[i]:>12.4f} "
                f"{self.total_order[i]:>12.4f} {self.interactions[i]:>16.4f}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.feature_names,
            "sobol_first": self.first_order,
            "sobol_total": self.total_order,
            "sobol_interaction": self.interactions,
        }).sort_values("sobol_total", ascending=False).reset_index(drop=True)


class SobolAnalysis:
    """Sobol sensitivity analysis for insurance pricing models.

    Estimates first-order and total-order Sobol indices using the Saltelli (2010)
    Monte Carlo estimator. Uses 2*(p+2)*n model evaluations.

    Parameters
    ----------
    model : callable
        Model function: f(X) -> y. Must accept a 2D array and return 1D array.
    n_samples : int
        Base sample size n. Total evaluations = 2*(p+2)*n.
    feature_names : list[str] or None
    random_state : int

    Examples
    --------
    >>> from insurance_sensitivity import SobolAnalysis
    >>> import numpy as np
    >>> # Ishigami test function: f(x) = sin(x1) + 7*sin^2(x2) + 0.1*x3^4*sin(x1)
    >>> def ishigami(X):
    ...     return np.sin(X[:,0]) + 7*np.sin(X[:,1])**2 + 0.1*X[:,2]**4*np.sin(X[:,0])
    >>> sa = SobolAnalysis(model=ishigami, n_samples=1000)
    >>> result = sa.fit(X_bounds=[(-np.pi, np.pi)]*3)
    >>> print(result.summary())
    """

    def __init__(
        self,
        model: Callable,
        n_samples: int = 2_000,
        feature_names: list[str] | None = None,
        random_state: int = 42,
    ):
        self.model         = model
        self.n_samples     = n_samples
        self.feature_names = feature_names
        self.random_state  = random_state
        self._result: SobolResult | None = None

    def fit(
        self,
        X: np.ndarray | None = None,
        X_bounds: list[tuple[float, float]] | None = None,
        p: int | None = None,
    ) -> SobolResult:
        """Compute Sobol indices.

        Parameters
        ----------
        X : array-like (n_ref, p) or None
            Reference data to estimate input distribution. Used if X_bounds is None.
        X_bounds : list of (lo, hi) tuples or None
            Uniform sampling bounds for each feature. If None, inferred from X.
        p : int or None
            Number of features. Required if X is None and X_bounds is None.

        Returns
        -------
        SobolResult
        """
        rng = np.random.default_rng(self.random_state)
        n   = self.n_samples

        if X is not None:
            X = np.asarray(X, dtype=float)
            p = X.shape[1]
            if X_bounds is None:
                X_bounds = [(X[:, j].min(), X[:, j].max()) for j in range(p)]
        elif X_bounds is not None:
            p = len(X_bounds)
        else:
            raise ValueError("Provide either X or X_bounds.")

        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(p)]
        else:
            feature_names = list(self.feature_names)

        # Sample matrices A and B from uniform bounds
        A = np.column_stack([
            rng.uniform(lo, hi, n) for lo, hi in X_bounds
        ])
        B = np.column_stack([
            rng.uniform(lo, hi, n) for lo, hi in X_bounds
        ])

        # Model evaluations on A and B
        f_A = self.model(A)
        f_B = self.model(B)

        total_var = float(np.var(np.concatenate([f_A, f_B])))
        if total_var < 1e-12:
            # Constant output — all indices zero
            return SobolResult(
                feature_names=feature_names,
                first_order=np.zeros(p),
                total_order=np.zeros(p),
                interactions=np.zeros(p),
                total_variance=total_var,
                sum_first_order=0.0,
            )

        S_i = np.zeros(p)
        T_i = np.zeros(p)

        for j in range(p):
            # A_B^(j): column j from B, rest from A
            A_Bj = A.copy()
            A_Bj[:, j] = B[:, j]

            f_ABj = self.model(A_Bj)

            # Saltelli (2010) estimators
            # First-order: V_i = mean(f_B * (f_ABj - f_A)) / Var(Y)
            v_i   = float(np.mean(f_B * (f_ABj - f_A)))
            S_i[j] = v_i / total_var

            # Total-order: T_i = mean((f_A - f_ABj)^2) / (2 * Var(Y))
            T_i[j] = float(np.mean((f_A - f_ABj) ** 2)) / (2 * total_var)

        # Clip to [0, 1] (numerical issues can push slightly outside)
        S_i = np.clip(S_i, 0.0, 1.0)
        T_i = np.clip(T_i, 0.0, 1.0)
        interactions = np.clip(T_i - S_i, 0.0, None)

        self._result = SobolResult(
            feature_names=feature_names,
            first_order=S_i,
            total_order=T_i,
            interactions=interactions,
            total_variance=total_var,
            sum_first_order=float(S_i.sum()),
        )
        return self._result

    @property
    def result(self) -> SobolResult:
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result
