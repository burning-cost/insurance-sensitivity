"""
insurance-sensitivity: Sensitivity analysis for insurance pricing models.

Global sensitivity analysis quantifies how much each input feature contributes to
the variance of model outputs. This is different from local sensitivity (SHAP values)
which measures the contribution of a feature to a single prediction. Global sensitivity
measures how much total output variance would be reduced if we knew a feature exactly.

Two methods are implemented:

**Sobol Indices** (variance decomposition):
    - First-order S_i: fraction of output variance due to feature i alone
    - Total-order T_i: fraction of output variance due to feature i including interactions
    - Interaction: T_i - S_i = variance due to interactions involving feature i
    - Problem: first-order Sobol misses interactions between features

**Shapley Effects** (cooperative game theory):
    - Phi_i: each feature's fair share of total output variance
    - Accounts for all interactions, allocated fairly using Shapley values
    - Phi_i = sum over all subsets S of [V(S u {i}) - V(S)] / (p choose |S|)
    - The sum of Shapley effects always equals total output variance (efficiency axiom)

In insurance pricing, Shapley effects are preferred when:
- Features are correlated (e.g. vehicle age and NCB are correlated)
- You need a complete, non-overlapping decomposition of variance
- You want to identify which features to collect/monitor for model governance

Key classes:
    SobolAnalysis      — First and total Sobol indices via Monte Carlo
    ShapleyEffects     — Shapley effects for global sensitivity
    SensitivityResult  — Result container

Typical usage::

    from insurance_sensitivity import ShapleyEffects

    se = ShapleyEffects(model=fitted_model, n_samples=5000)
    result = se.fit(X)
    print(result.summary())
"""

from insurance_sensitivity.sobol import SobolAnalysis, SobolResult
from insurance_sensitivity.shapley import ShapleyEffects, ShapleyResult

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-sensitivity")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    "SobolAnalysis",
    "SobolResult",
    "ShapleyEffects",
    "ShapleyResult",
    "__version__",
]
