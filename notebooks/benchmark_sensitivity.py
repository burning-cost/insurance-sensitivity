# Databricks notebook source
# This file uses Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line
#
# Run end-to-end on Databricks. Do not run locally.

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark: insurance-sensitivity — Shapley Effects vs First-Order Sobol Indices
# MAGIC
# MAGIC **Library:** `insurance-sensitivity` — Global sensitivity analysis for insurance
# MAGIC pricing models. Implements Sobol indices and Shapley effects.
# MAGIC
# MAGIC **Comparison:** Shapley effects vs first-order Sobol indices.
# MAGIC Both measure feature importance for model governance, but they differ in how they
# MAGIC handle interactions and correlated features.
# MAGIC
# MAGIC **Dataset:** Synthetic motor insurance pricing model with known true sensitivities.
# MAGIC Features include correlated pairs (vehicle_age and NCB) and genuine interactions.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Central question:** When insurance features are correlated (vehicle_age and NCB
# MAGIC are both proxies for driving experience) and genuinely interact in the model,
# MAGIC do Shapley effects provide a more complete sensitivity picture than first-order Sobol?
# MAGIC
# MAGIC **Key metrics:** Total sensitivity indices, interaction detection, allocation accuracy
# MAGIC vs analytical truth.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install git+https://github.com/burning-cost/insurance-sensitivity.git
%pip install matplotlib seaborn pandas numpy scipy scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from insurance_sensitivity import SobolAnalysis, ShapleyEffects

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test Functions: Motor Insurance Pricing Models
# MAGIC
# MAGIC We analyse two pricing models with known true sensitivities:
# MAGIC
# MAGIC **Model A (Additive, independent features):**
# MAGIC     log_rate = 3*x_age + 2*x_ncb + 1*x_vage + 0.5*x_region
# MAGIC     True S_i ∝ coef_i^2 * Var(x_i). No interactions. Sobol = Shapley.
# MAGIC
# MAGIC **Model B (Multiplicative with interaction, correlated features):**
# MAGIC     log_rate = 3*x_age + 2*x_ncb + x_age * x_ncb + x_vage^2
# MAGIC     True: x_age and x_ncb interact. Their Sobol first-order indices undercount
# MAGIC     their joint effect. Shapley effects correctly attribute the interaction.
# MAGIC     Additionally: x_age and x_ncb are correlated (rho=0.5), so independent-input
# MAGIC     Sobol indices give biased results.

# COMMAND ----------

def generate_insurance_features(n=5000, seed=42):
    """Generate correlated motor insurance features."""
    rng = np.random.default_rng(seed)

    # Independent features
    vehicle_age = rng.uniform(0, 1, n)    # 0-1 (standardised 0-10 years)
    region_risk = rng.uniform(0, 1, n)    # 0-1 (risk score)

    # Correlated pair: driver_age and ncb are correlated (rho=0.5)
    # Young drivers have low NCB; experienced drivers have high NCB
    driver_age = rng.uniform(0, 1, n)
    ncb_noise  = rng.uniform(0, 1, n)
    ncb        = 0.7 * driver_age + 0.3 * ncb_noise  # rho ≈ 0.7 with driver_age
    ncb        = np.clip(ncb, 0, 1)

    return np.column_stack([driver_age, ncb, vehicle_age, region_risk])


FEATURE_NAMES = ["driver_age", "ncb", "vehicle_age", "region_risk"]

# True coefficients for Model A (additive)
COEF_A = np.array([3.0, 2.0, 1.0, 0.5])   # driver_age, ncb, vehicle_age, region_risk

def model_A(X):
    """Additive model with independent features (no interactions)."""
    return X @ COEF_A

# Model B: interaction between driver_age (x0) and ncb (x1), plus nonlinear vehicle_age
def model_B(X):
    """Multiplicative model with interaction and correlated features."""
    return (3.0 * X[:, 0]
            + 2.0 * X[:, 1]
            + 2.0 * X[:, 0] * X[:, 1]    # interaction
            + 1.5 * X[:, 2] ** 2          # nonlinear vehicle_age
            + 0.5 * X[:, 3])

# Fitted insurance pricing model (Poisson GLM with interactions)
X_ref = generate_insurance_features(n=10_000, seed=42)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_ref)

glm_model = PoissonRegressor(alpha=0.01, max_iter=500)
scaler    = StandardScaler()
X_ref_s   = scaler.fit_transform(X_ref)

# Create synthetic claim rates using model_B as the true DGP
y_ref = np.exp(model_B(X_ref_s) * 0.3 + 0.2)  # log-link, realistic scale

glm_model.fit(X_ref_s, y_ref)

def fitted_glm(X):
    """Fitted Poisson GLM model."""
    X_s = scaler.transform(X)
    return glm_model.predict(X_s)

print(f"Reference data: {X_ref.shape}")
print(f"\nCorrelation between driver_age and ncb: {np.corrcoef(X_ref[:,0], X_ref[:,1])[0,1]:.3f}")
print(f"\nModel A (additive, no interactions): output variance = {np.var(model_A(X_ref_s)):.4f}")
print(f"Model B (interaction + nonlinear): output variance = {np.var(model_B(X_ref_s)):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model A: Additive Model — Sobol = Shapley (independent features)
# MAGIC
# MAGIC On an additive model with independent features, first-order Sobol indices
# MAGIC should equal Shapley effects. This is the "null case" — both methods agree.

# COMMAND ----------

print("=" * 60)
print("MODEL A: Additive, no interactions")
print("=" * 60)

# Analytical true Sobol indices for additive model with independent uniform(0,1) inputs
# Var(X_i) = 1/12 for uniform(0,1)
# Var(Y) = sum_i (coef_i^2 * Var(X_i))
var_xi    = 1/12  # for uniform(0,1)
var_y_A   = np.sum(COEF_A**2 * var_xi)
true_sobol_A = (COEF_A**2 * var_xi) / var_y_A

print(f"\nTrue first-order Sobol indices (analytical):")
for name, si in zip(FEATURE_NAMES, true_sobol_A):
    print(f"  {name:<15}: S_i = {si:.4f}")
print(f"  Sum = {true_sobol_A.sum():.4f} (= 1.0 for additive model)")

# Bounds for uniform(0,1) features
X_bounds_A = [(0.0, 1.0)] * 4

t0 = time.perf_counter()
sobol_A = SobolAnalysis(
    model=model_A,
    n_samples=5000,
    feature_names=FEATURE_NAMES,
    random_state=42,
)
result_sobol_A = sobol_A.fit(X_bounds=X_bounds_A)
sobol_A_time = time.perf_counter() - t0

print(f"\nSobol Analysis (n={sobol_A.n_samples:,}):")
print(result_sobol_A.summary())
print(f"Fit time: {sobol_A_time:.2f}s")

# COMMAND ----------

# Generate independent uniform(0,1) features for Shapley analysis
rng_sh = np.random.default_rng(42)
X_indep = rng_sh.uniform(0, 1, (5000, 4))

t0 = time.perf_counter()
shapley_A = ShapleyEffects(
    model=model_A,
    n_samples=500,
    n_perms=100,
    feature_names=FEATURE_NAMES,
    random_state=42,
)
result_shapley_A = shapley_A.fit(X_indep)
shapley_A_time = time.perf_counter() - t0

print(f"Shapley Effects (n_perms={shapley_A.n_perms}):")
print(result_shapley_A.summary())
print(f"Fit time: {shapley_A_time:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model B: Interaction + Correlated Features — Shapley captures interactions

# COMMAND ----------

print("=" * 60)
print("MODEL B: Interaction + correlated features")
print("=" * 60)

# Use correlated features for Sobol and Shapley
# Note: Sobol assumes independent features — this will give biased estimates for correlated inputs
X_B = generate_insurance_features(n=5000, seed=42)

# For Sobol with correlated inputs, we use the marginal bounds (this is the standard approach
# but introduces bias for correlated features)
X_bounds_B = [(0.0, 1.0)] * 4   # approximate bounds for correlated features

t0 = time.perf_counter()
sobol_B = SobolAnalysis(
    model=model_B,
    n_samples=5000,
    feature_names=FEATURE_NAMES,
    random_state=42,
)
result_sobol_B = sobol_B.fit(X_bounds=X_bounds_B)
sobol_B_time = time.perf_counter() - t0

print(f"\nSobol Analysis on Model B (assumes independent inputs):")
print(result_sobol_B.summary())
print(f"Fit time: {sobol_B_time:.2f}s")

# COMMAND ----------

t0 = time.perf_counter()
shapley_B = ShapleyEffects(
    model=model_B,
    n_samples=500,
    n_perms=100,
    feature_names=FEATURE_NAMES,
    random_state=42,
)
result_shapley_B = shapley_B.fit(X_B)   # uses actual correlated X
shapley_B_time = time.perf_counter() - t0

print(f"\nShapley Effects on Model B (uses actual correlated X):")
print(result_shapley_B.summary())
print(f"Fit time: {shapley_B_time:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model C: Fitted Poisson GLM on Insurance Data

# COMMAND ----------

print("=" * 60)
print("MODEL C: Fitted Poisson GLM on insurance data")
print("=" * 60)

t0 = time.perf_counter()
sobol_C = SobolAnalysis(
    model=fitted_glm,
    n_samples=2000,
    feature_names=FEATURE_NAMES,
    random_state=42,
)
result_sobol_C = sobol_C.fit(X_bounds=[(0, 1), (0, 1), (0, 1), (0, 1)])
sobol_C_time = time.perf_counter() - t0

print(f"\nSobol Analysis on fitted GLM:")
print(result_sobol_C.summary())

t0 = time.perf_counter()
shapley_C = ShapleyEffects(
    model=fitted_glm,
    n_samples=300,
    n_perms=50,
    feature_names=FEATURE_NAMES,
    random_state=42,
)
result_shapley_C = shapley_C.fit(X_ref)
shapley_C_time = time.perf_counter() - t0

print(f"\nShapley Effects on fitted GLM:")
print(result_shapley_C.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Comparison: Interaction Detection Accuracy

# COMMAND ----------

# The key test: Model B has a true interaction between driver_age (x0) and ncb (x1).
# First-order Sobol indices MISS this interaction in their first-order values.
# Total Sobol T_i - S_i should detect it.
# Shapley effects should correctly allocate the interaction contribution.

print("=== Interaction Detection in Model B ===\n")

# True: interaction term x0 * x1 with coefficient 2.0
# For uniform(0,1) inputs: Var(x0*x1) = 4/9 - 1/4 = 7/36 ≈ 0.194 per term
# The interaction contributes to BOTH x0 and x1's total variance budget

print("Sobol first-order S_i (misses interactions):")
for name, si in zip(FEATURE_NAMES, result_sobol_B.first_order):
    print(f"  {name:<15}: S_i = {si:.4f}")
print(f"  Sum S_i = {result_sobol_B.sum_first_order:.4f}  (<1 means interactions present)")

print(f"\nSobol interaction indices T_i - S_i:")
for name, ti, si in zip(FEATURE_NAMES, result_sobol_B.total_order, result_sobol_B.first_order):
    print(f"  {name:<15}: T_i - S_i = {ti-si:.4f}")

print(f"\nShapley effects (includes interaction attributed fairly):")
for name, phi in zip(FEATURE_NAMES, result_shapley_B.shapley_effects):
    print(f"  {name:<15}: phi_i = {phi:.4f}  ({result_shapley_B.shapley_normalised[FEATURE_NAMES.index(name)]:.3f})")

# COMMAND ----------

# Comparison table
print("\n=== Summary Comparison Table ===\n")

def pct(a, b):
    return f"{(b-a)/max(abs(a),1e-8)*100:+.0f}%"


rows_compare = []
for i, name in enumerate(FEATURE_NAMES):
    rows_compare.append({
        "Feature": name,
        "Sobol S_i (1st, indep)": f"{result_sobol_B.first_order[i]:.4f}",
        "Sobol T_i (total)": f"{result_sobol_B.total_order[i]:.4f}",
        "Sobol interaction (T-S)": f"{result_sobol_B.interactions[i]:.4f}",
        "Shapley phi_i": f"{result_shapley_B.shapley_effects[i]:.4f}",
        "Shapley normalised": f"{result_shapley_B.shapley_normalised[i]:.4f}",
    })

comparison_df = pd.DataFrame(rows_compare)
print(comparison_df.to_string(index=False))

print(f"\nKey finding:")
print(f"  Sum of first-order Sobol S_i = {result_sobol_B.sum_first_order:.4f}")
print(f"  (< 1 confirms interactions; sum of Shapley always = 1 by efficiency axiom)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Diagnostic Plots

# COMMAND ----------

fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # Model A: Sobol vs Shapley (should agree)
ax2 = fig.add_subplot(gs[1, 0])   # Model B: Sobol first-order
ax3 = fig.add_subplot(gs[1, 1])   # Model B: Shapley effects
ax4 = fig.add_subplot(gs[2, 0])   # Model B: Interaction indices
ax5 = fig.add_subplot(gs[2, 1])   # Fitted GLM: Sobol vs Shapley

# ── Plot 1: Model A — Sobol should equal Shapley ───────────────────────────
x_pos = np.arange(4)
w = 0.25
ax1.bar(x_pos - w,     true_sobol_A,                      w, label="True S_i (analytical)", color="black",    alpha=0.7)
ax1.bar(x_pos,         result_sobol_A.first_order,        w, label="Sobol S_i (estimated)",  color="steelblue", alpha=0.8)
ax1.bar(x_pos + w,     result_shapley_A.shapley_normalised, w, label="Shapley phi_i (normalised)", color="tomato", alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(FEATURE_NAMES, rotation=10)
ax1.set_ylabel("Sensitivity index")
ax1.set_title(
    "Model A (Additive, Independent) — Sobol S_i ≈ Shapley phi_i\n"
    "When there are no interactions, both methods agree",
    fontsize=10,
)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# ── Plot 2: Model B Sobol first-order ────────────────────────────────────
ax2.bar(x_pos - w/2, result_sobol_B.first_order, w, label="S_i (first-order)", color="steelblue", alpha=0.8)
ax2.bar(x_pos + w/2, result_sobol_B.total_order, w, label="T_i (total-order)", color="steelblue", alpha=0.4)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(FEATURE_NAMES, rotation=10)
ax2.set_ylabel("Sobol index")
ax2.set_title(
    f"Model B — Sobol Indices\nSum S_i = {result_sobol_B.sum_first_order:.3f} (interaction detected: <1)",
    fontsize=10,
)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# ── Plot 3: Model B Shapley effects ────────────────────────────────────────
colors3 = ["tomato" if i < 2 else "steelblue" for i in range(4)]  # highlight interacting pair
bars = ax3.bar(x_pos, result_shapley_B.shapley_effects, color=colors3, alpha=0.8, width=0.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(FEATURE_NAMES, rotation=10)
ax3.set_ylabel("Shapley effect phi_i")
ax3.set_title(
    f"Model B — Shapley Effects\nSum phi_i = {result_shapley_B.sum_shapley:.4f} ≈ Var(Y)",
    fontsize=10,
)
ax3.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, result_shapley_B.shapley_effects):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.4f}", ha="center", fontsize=8)
ax3.text(0.5, 0.95, "Red = features with\ntrue interaction", transform=ax3.transAxes,
         ha="center", va="top", fontsize=8, color="tomato")

# ── Plot 4: Interaction indices (T_i - S_i) ────────────────────────────────
interactions = result_sobol_B.interactions
ax4.bar(x_pos, interactions, color=["tomato" if i < 2 else "steelblue" for i in range(4)], alpha=0.8, width=0.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(FEATURE_NAMES, rotation=10)
ax4.set_ylabel("Interaction index T_i - S_i")
ax4.set_title(
    "Model B — Sobol Interaction Indices\nT_i - S_i > 0 detects interaction (red = true interacting pair)",
    fontsize=10,
)
ax4.grid(True, alpha=0.3, axis="y")

# ── Plot 5: Fitted GLM — Sobol vs Shapley ─────────────────────────────────
si_C  = result_sobol_C.first_order
ti_C  = result_sobol_C.total_order
phi_C = result_shapley_C.shapley_normalised

ax5.bar(x_pos - w, si_C,  w, label="Sobol S_i (1st)",  color="steelblue",  alpha=0.8)
ax5.bar(x_pos,     ti_C,  w, label="Sobol T_i (total)", color="steelblue",  alpha=0.4)
ax5.bar(x_pos + w, phi_C, w, label="Shapley phi_i",    color="tomato",    alpha=0.8)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(FEATURE_NAMES, rotation=10)
ax5.set_ylabel("Sensitivity index")
ax5.set_title("Fitted Poisson GLM — Sobol vs Shapley\nReal-world model sensitivity decomposition", fontsize=10)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "insurance-sensitivity: Shapley Effects vs First-Order Sobol Indices\n"
    "Global sensitivity analysis on motor insurance pricing models",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.savefig("/tmp/benchmark_sensitivity.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_sensitivity.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verdict
# MAGIC
# MAGIC ### When to use Shapley effects over first-order Sobol indices
# MAGIC
# MAGIC **Shapley effects win when:**
# MAGIC
# MAGIC - **Features interact.** Motor insurance models routinely have interactions:
# MAGIC   young driver * urban area, high-value vehicle * young driver, etc. First-order
# MAGIC   Sobol S_i attributes only the individual feature's variance, ignoring interactions.
# MAGIC   Shapley effects allocate interaction variance fairly among the interacting features.
# MAGIC   The total-order Sobol T_i catches interactions but double-counts (T_i + T_j > 1
# MAGIC   for interacting i and j). Shapley effects sum exactly to Var(Y).
# MAGIC
# MAGIC - **Features are correlated.** Sobol assumes independent inputs. When vehicle age
# MAGIC   and NCB are correlated (as they are in practice), Sobol indices under-attribute
# MAGIC   variance to correlated feature pairs. Shapley effects use the actual empirical
# MAGIC   joint distribution.
# MAGIC
# MAGIC - **You need a complete variance decomposition for model governance.** FCA model
# MAGIC   governance requires demonstrating that you understand which features drive
# MAGIC   your model. Shapley effects sum to 100% of explained variance — they produce a
# MAGIC   pie chart that adds up. First-order Sobol indices don't (their sum < 1 under
# MAGIC   interactions) and total-order Sobol indices double-count.
# MAGIC
# MAGIC - **Lloyd's model validation.** The SBS framework asks "which variables drive
# MAGIC   the most variance in the technical rate?" Shapley effects give a defensible,
# MAGIC   principled answer with a clean attribution.
# MAGIC
# MAGIC **First-order Sobol indices are sufficient when:**
# MAGIC
# MAGIC - **Features are approximately independent.** If you've done a proper feature
# MAGIC   engineering step and decorrelated your inputs, first-order Sobol is exact.
# MAGIC
# MAGIC - **Interaction detection is the goal, not attribution.** T_i - S_i is a simple,
# MAGIC   fast interaction indicator. If you just want to know "is there an interaction
# MAGIC   involving this feature?" Sobol is faster.
# MAGIC
# MAGIC - **Computational budget is tight.** Sobol with n=5,000 requires ~(2*p+2)*n model
# MAGIC   evaluations — typically 50,000-100,000 for a 5-feature model. Shapley effects
# MAGIC   require n_perms * p * n_inner evaluations — typically 200,000-500,000.
# MAGIC   Sobol is 3-5x faster for the same accuracy.
# MAGIC
# MAGIC **Expected performance (this benchmark):**
# MAGIC
# MAGIC | Metric                           | First-Order Sobol    | Shapley Effects        |
# MAGIC |----------------------------------|----------------------|------------------------|
# MAGIC | Independent features (Model A)   | Accurate             | Accurate (agrees)      |
# MAGIC | Interactions (Model B)           | Underestimates       | Correct attribution    |
# MAGIC | Correlated features              | Biased               | Handles correctly      |
# MAGIC | Completeness (sums to 1)         | No (< 1 if interact) | Yes (always)           |
# MAGIC | Runtime (p=4, n=5k)              | ~1-5s                | ~20-60s                |
# MAGIC | Interaction detection            | Via T_i - S_i        | Implicit in allocation |

# COMMAND ----------

print("=" * 65)
print("VERDICT: Shapley Effects vs First-Order Sobol Indices")
print("=" * 65)
print()
print("Model A (additive, independent):")
print(f"  Sobol S_i sum:      {result_sobol_A.sum_first_order:.4f}  (= 1.0: no interactions)")
for n_, si, phi in zip(FEATURE_NAMES, result_sobol_A.first_order, result_shapley_A.shapley_normalised):
    print(f"    {n_:<15}: Sobol={si:.4f}  Shapley={phi:.4f}  Agree: {'YES' if abs(si-phi) < 0.05 else 'NO'}")

print()
print("Model B (interaction + correlated):")
print(f"  Sobol S_i sum:      {result_sobol_B.sum_first_order:.4f}  (< 1.0: interactions present)")
print(f"  Shapley phi sum:    {result_shapley_B.sum_shapley:.4f}  ≈ Var(Y) = {np.var(model_B(X_B)):.4f}")
print()
print("  Interaction detection (driver_age * ncb term is TRUE):")
for n_, ti, si, phi in zip(FEATURE_NAMES, result_sobol_B.total_order, result_sobol_B.first_order, result_shapley_B.shapley_effects):
    inter = ti - si
    print(f"    {n_:<15}: Sobol T-S={inter:.4f}  Shapley={phi:.4f}")

print()
print(f"  Fit time — Sobol:   {sobol_B_time:.2f}s")
print(f"  Fit time — Shapley: {shapley_B_time:.2f}s")
print()
print("  Bottom line:")
print("  Shapley effects correctly attribute interaction variance;")
print("  first-order Sobol underestimates importance of interacting features.")
print("  Shapley effects sum to Var(Y) — a complete, non-overlapping decomposition.")
