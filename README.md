# insurance-sensitivity

Global sensitivity analysis for insurance pricing models: Sobol indices and Shapley effects.

## The problem

Your actuarial review asks: "Which features drive the most variance in your technical rate?"
You compute feature importances from your GBM. But GBM importances are local (split-based
or permutation-based) and don't tell you how much of the *total output variance* each feature
accounts for. They also don't tell you which features interact.

For model governance under FCA Consumer Duty and Lloyd's SBS, you need a rigorous,
principled answer to "what drives this model?" — one that accounts for interactions and
correlated features, and sums to 100% of explained variance.

Global sensitivity analysis provides this. Two complementary approaches are implemented:

**Sobol indices** decompose total output variance into contributions from individual
features and their interactions. First-order S_i measures direct contributions; total-order
T_i includes interactions. Difference T_i - S_i reveals how much of a feature's total
contribution comes from interactions with other features.

**Shapley effects** apply Shapley values from cooperative game theory to variance
decomposition. Each feature's Shapley effect is its fair share of total variance across
all possible coalitions of features. Key property: Shapley effects always sum to Var(Y)
(efficiency axiom) — a complete, non-overlapping partition of total output variance.

## When they differ

For additive models with independent features, Sobol first-order indices equal
normalised Shapley effects. They diverge when:

1. **Features interact** — Sobol S_i underestimates interacting features; Shapley allocates
   interaction variance fairly among the interacting features.
2. **Features are correlated** — Sobol assumes independent inputs (sampling from
   the product of marginals); Shapley uses the actual joint distribution.

Both cases are common in insurance: driver age and NCB are correlated; vehicle age and
geographic risk interact; young driver and urban area interact.

## Installation

```bash
pip install git+https://github.com/burning-cost/insurance-sensitivity.git
```

## Usage

```python
from insurance_sensitivity import SobolAnalysis, ShapleyEffects

# Sobol indices (fast, assumes independent features)
sa = SobolAnalysis(
    model=my_model,      # callable: f(X) -> y
    n_samples=5000,
    feature_names=["driver_age", "ncb", "vehicle_age", "region"],
)
result = sa.fit(X_bounds=[(0, 80), (0, 5), (0, 15), (0, 3)])
print(result.summary())
df = result.to_dataframe()   # feature, sobol_first, sobol_total, sobol_interaction

# Shapley effects (complete variance decomposition, handles correlations)
se = ShapleyEffects(
    model=my_model,
    n_samples=500,    # inner MC samples per query point
    n_perms=100,      # permutations for Shapley averaging
    feature_names=["driver_age", "ncb", "vehicle_age", "region"],
)
result = se.fit(X_train)    # uses actual joint distribution from training data
print(result.summary())
```

## Performance

Benchmarked against Sobol first-order indices on synthetic motor insurance pricing
models with known true sensitivities. See `notebooks/benchmark_sensitivity.py` for methodology.

- **On additive models with independent features, Sobol and Shapley agree** — the
  additional cost of Shapley computation is not justified here.
- **On models with genuine interactions** (young driver * urban, vehicle value * claim type),
  first-order Sobol underestimates importance of interacting features by 20-40%.
  Total-order Sobol (T_i) detects interaction but double-counts. Shapley effects
  correctly attribute interaction variance to the contributing features.
- **On correlated features** (driver_age and NCB, rho=0.7), Sobol indices under-attribute
  total variance because they sample from independent marginals. Shapley effects using
  the empirical joint distribution give the correct allocation.
- **Shapley effects always sum to Var(Y)** — essential for model governance where you
  need percentages that add to 100%.
- **Runtime tradeoff**: Sobol is 3-5x faster for the same sample size (n_samples=5000).
  Shapley with n_perms=100, n_inner=50 takes 3-5x longer but provides correct attribution
  under interactions. For p=4-6 features and n=10k reference data, both complete in <2 minutes.

## References

- Sobol', I. M. (1993). Sensitivity analysis for non-linear mathematical models. *MMCE*, 1(4).
- Saltelli, A., et al. (2010). Variance based sensitivity analysis of model output. *CPC*, 181(2).
- Owen, A. B. (2014). Sobol' indices and Shapley value. *SIAM/ASA J. Uncertainty Quantification*, 2(1).
- Broto, B., et al. (2020). Variance reduction for estimation of Shapley effects. *SIAM/ASA JUQ*, 8(2).
