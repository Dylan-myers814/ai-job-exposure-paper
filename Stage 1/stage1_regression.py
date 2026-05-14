"""
=============================================================================
AI Job Exposure Analysis — Stage 1: OLS Regression
=============================================================================
Dylan Myers | Undergraduate Research Paper

PURPOSE:
  Estimate which structural occupational characteristics predict AI exposure
  scores (AIOE), treating exposure as an endogenous outcome rather than an
  exogenous instrument. This reversal of the standard causal arrow is the
  core methodological contribution of Stage 1.

MODELS RUN:
  Model 1 — Baseline task content only
  Model 2 — Task content + education
  Model 3 — Task content + education + wage (full model)
  Model 4 — Full model with wage percentile instead of log wage (robustness)

OUTPUTS (all saved to outputs/):
  tables/regression_table.csv       — coefficients, SEs, p-values, stars
  tables/summary_stats.csv          — descriptive statistics
  tables/correlation_matrix.csv     — pairwise correlations
  figures/scatter_matrix.png        — variable relationships
  figures/residual_plots.png        — OLS diagnostic plots
  figures/coef_plot.png             — coefficient plot (Model 3)
  figures/aioe_distribution.png     — distribution of dependent variable

USAGE:
  pip install pandas numpy statsmodels matplotlib seaborn scipy
  python stage1_regression.py

  Input:  ai_exposure_stage1.csv  (from collect_data.py)
  Output: outputs/ directory (auto-created)
=============================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────

INPUT_FILE  = "ai_exposure_stage1.csv"
OUTPUT_DIR  = "outputs"
TABLE_DIR   = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR  = os.path.join(OUTPUT_DIR, "figures")

for d in [TABLE_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

# Plot style
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "figure.dpi":      150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})
PALETTE = ["#2E4057", "#048A81", "#54C6EB", "#EF9CDA"]


# ── LOAD & VALIDATE DATA ─────────────────────────────────────────────────────

def load_data():
    print("=" * 65)
    print("  Stage 1: OLS Regression — AI Occupational Exposure")
    print("=" * 65)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"'{INPUT_FILE}' not found.\n"
            "Run collect_data.py first to generate the dataset."
        )

    df = pd.read_csv(INPUT_FILE)
    print(f"\n[Data] Loaded {len(df)} occupations, {df.shape[1]} variables")

    # Required columns
    required = [
        "aioe_score", "routine_cognitive", "social_interpersonal",
        "physical_manual", "creative_adaptive", "education_level",
        "annual_median_wage", "log_wage", "wage_percentile"
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}\n"
                         "Re-run collect_data.py to regenerate.")

    # Drop rows missing any key variable
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[Data] Dropped {n_dropped} rows with missing values "
              f"({len(df)} complete cases remain)")

    return df


# ── SUMMARY STATISTICS ───────────────────────────────────────────────────────

def compute_summary_stats(df):
    print("\n[1/6] Computing summary statistics...")

    vars_of_interest = {
        "aioe_score":           "AIOE Score (Felten et al.)",
        "routine_cognitive":    "Routine Cognitive Task Share",
        "social_interpersonal": "Social/Interpersonal Task Share",
        "physical_manual":      "Physical/Manual Task Share",
        "creative_adaptive":    "Creative/Adaptive Task Share",
        "education_level":      "Required Education Level (1–9)",
        "annual_median_wage":   "Annual Median Wage ($)",
        "log_wage":             "Log Annual Wage",
        "wage_percentile":      "Wage Percentile",
    }

    rows = []
    for var, label in vars_of_interest.items():
        s = df[var].describe()
        rows.append({
            "Variable":  label,
            "N":         int(s["count"]),
            "Mean":      round(s["mean"], 3),
            "Std Dev":   round(s["std"],  3),
            "Min":       round(s["min"],  3),
            "25th Pct":  round(s["25%"],  3),
            "Median":    round(s["50%"],  3),
            "75th Pct":  round(s["75%"],  3),
            "Max":       round(s["max"],  3),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(TABLE_DIR, "summary_stats.csv"), index=False)
    print(f"  ✓ Saved: tables/summary_stats.csv")
    print(summary[["Variable", "N", "Mean", "Std Dev", "Min", "Max"]].to_string(index=False))
    return summary


# ── CORRELATION MATRIX ───────────────────────────────────────────────────────

def compute_correlations(df):
    print("\n[2/6] Computing correlation matrix...")

    corr_vars = [
        "aioe_score", "routine_cognitive", "social_interpersonal",
        "physical_manual", "creative_adaptive", "education_level", "log_wage"
    ]
    labels = {
        "aioe_score":           "AIOE",
        "routine_cognitive":    "Routine Cog.",
        "social_interpersonal": "Social",
        "physical_manual":      "Physical",
        "creative_adaptive":    "Creative",
        "education_level":      "Education",
        "log_wage":             "Log Wage",
    }

    corr = df[corr_vars].rename(columns=labels).corr()
    corr.to_csv(os.path.join(TABLE_DIR, "correlation_matrix.csv"))
    print(f"  ✓ Saved: tables/correlation_matrix.csv")

    # Flag high correlations (potential multicollinearity)
    high_corr = []
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.6:
                high_corr.append(f"    {c1} × {c2}: r = {r:.3f} ⚠️")
    if high_corr:
        print("  High correlations detected (|r| > 0.6):")
        for hc in high_corr:
            print(hc)
    else:
        print("  No high correlations (|r| > 0.6) detected among regressors")

    return corr


# ── RUN OLS MODELS ───────────────────────────────────────────────────────────

def run_regressions(df):
    print("\n[3/6] Running OLS regressions...")

    y = df["aioe_score"]

    # Define four model specifications
    task_vars = [
        "routine_cognitive", "social_interpersonal",
        "physical_manual", "creative_adaptive"
    ]

    model_specs = {
        "Model 1\n(Task Content)": task_vars,
        "Model 2\n(+ Education)":  task_vars + ["education_level"],
        "Model 3\n(Full Model)":   task_vars + ["education_level", "log_wage"],
        "Model 4\n(Robustness)":   task_vars + ["education_level", "wage_percentile"],
    }

    results   = {}
    fit_stats = {}

    for name, regressors in model_specs.items():
        X = sm.add_constant(df[regressors])

        # OLS with HC3 heteroskedasticity-robust standard errors
        model  = sm.OLS(y, X).fit(cov_type="HC3")
        results[name] = model

        # Breusch-Pagan heteroskedasticity test
        bp_lm, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)

        fit_stats[name] = {
            "N":           int(model.nobs),
            "R²":          round(model.rsquared, 4),
            "Adj. R²":     round(model.rsquared_adj, 4),
            "F-stat":      round(model.fvalue, 3),
            "F p-value":   round(model.f_pvalue, 4),
            "AIC":         round(model.aic, 2),
            "BIC":         round(model.bic, 2),
            "BP test p":   round(bp_p, 4),
            "SE type":     "HC3 Robust",
        }

        print(f"\n  {name.replace(chr(10), ' ')}")
        print(f"    N={int(model.nobs)}  R²={model.rsquared:.4f}  "
              f"Adj.R²={model.rsquared_adj:.4f}  F={model.fvalue:.2f}  "
              f"F-p={model.f_pvalue:.4f}")
        print(f"    Breusch-Pagan p={bp_p:.4f} "
              f"({'heteroskedastic — HC3 SEs appropriate ✓' if bp_p < 0.05 else 'homoskedastic'})")

    return results, fit_stats


# ── VIF CALCULATION ──────────────────────────────────────────────────────────

def compute_vif(df):
    print("\n[4/6] Computing VIF (multicollinearity check)...")

    full_vars = [
        "routine_cognitive", "social_interpersonal",
        "physical_manual", "creative_adaptive",
        "education_level", "log_wage"
    ]
    X_vif = sm.add_constant(df[full_vars])

    vif_data = pd.DataFrame({
        "Variable": X_vif.columns,
        "VIF":      [variance_inflation_factor(X_vif.values, i)
                     for i in range(X_vif.shape[1])]
    }).query("Variable != 'const'")

    vif_data["Concern"] = vif_data["VIF"].apply(
        lambda v: "HIGH ⚠️" if v > 10 else ("Moderate" if v > 5 else "OK ✓")
    )

    print(vif_data.to_string(index=False))

    high_vif = vif_data[vif_data["VIF"] > 10]
    if len(high_vif) > 0:
        print(f"\n  ⚠️  High VIF detected for: {list(high_vif['Variable'])}")
        print("  Consider dropping one variable or combining correlated predictors")
    else:
        print("\n  ✓ No severe multicollinearity (all VIF < 10)")

    return vif_data


# ── REGRESSION TABLE ─────────────────────────────────────────────────────────

def build_regression_table(results, fit_stats):
    print("\n[5/6] Building regression table...")

    def stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "†"
        return ""

    var_labels = {
        "const":               "Intercept",
        "routine_cognitive":   "Routine Cognitive Tasks",
        "social_interpersonal":"Social/Interpersonal Tasks",
        "physical_manual":     "Physical/Manual Tasks",
        "creative_adaptive":   "Creative/Adaptive Tasks",
        "education_level":     "Required Education Level",
        "log_wage":            "Log Annual Wage",
        "wage_percentile":     "Wage Percentile",
    }

    all_vars = list(var_labels.keys())
    model_names = list(results.keys())
    rows = []

    for var in all_vars:
        coef_row   = {"Variable": var_labels[var]}
        se_row     = {"Variable": ""}
        found_any  = False

        for mname, model in results.items():
            if var in model.params.index:
                found_any = True
                coef = model.params[var]
                se   = model.bse[var]
                pval = model.pvalues[var]
                coef_row[mname] = f"{coef:.4f}{stars(pval)}"
                se_row[mname]   = f"({se:.4f})"
            else:
                coef_row[mname] = "—"
                se_row[mname]   = ""

        if found_any:
            rows.append(coef_row)
            rows.append(se_row)

    # Fit statistics rows
    rows.append({m: "" for m in model_names} | {"Variable": "─" * 30})
    for stat in ["N", "R²", "Adj. R²", "F-stat", "F p-value", "AIC", "SE type"]:
        row = {"Variable": stat}
        for mname in model_names:
            row[mname] = str(fit_stats[mname].get(stat, ""))
        rows.append(row)

    rows.append({
        "Variable": "Note: HC3 robust SEs in parentheses. † p<0.10, * p<0.05, ** p<0.01, *** p<0.001"
    })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLE_DIR, "regression_table.csv"), index=False)
    print(f"  ✓ Saved: tables/regression_table.csv")
    print("\n" + table.to_string(index=False))

    return table


# ── FIGURES ──────────────────────────────────────────────────────────────────

def make_figures(df, results):
    print("\n[6/6] Generating figures...")

    model3 = list(results.values())[2]  # Full model

    # ── Figure 1: AIOE distribution ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["aioe_score"], bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    axes[0].set_title("Distribution of AIOE Scores")
    axes[0].set_xlabel("AI Occupational Exposure Score")
    axes[0].set_ylabel("Number of Occupations")
    axes[0].axvline(df["aioe_score"].mean(), color=PALETTE[1], linestyle="--",
                    linewidth=1.5, label=f"Mean = {df['aioe_score'].mean():.3f}")
    axes[0].axvline(df["aioe_score"].median(), color=PALETTE[2], linestyle=":",
                    linewidth=1.5, label=f"Median = {df['aioe_score'].median():.3f}")
    axes[0].legend(fontsize=9)

    # Q-Q plot for normality check
    stats.probplot(df["aioe_score"], plot=axes[1])
    axes[1].set_title("Q-Q Plot: AIOE Score vs. Normal")
    axes[1].get_lines()[0].set(color=PALETTE[0], markersize=3, alpha=0.6)
    axes[1].get_lines()[1].set(color=PALETTE[1])

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "aioe_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: figures/aioe_distribution.png")

    # ── Figure 2: Scatter matrix ─────────────────────────────────────────────
    scatter_vars = [
        "aioe_score", "routine_cognitive", "social_interpersonal",
        "physical_manual", "education_level", "log_wage"
    ]
    scatter_labels = {
        "aioe_score":           "AIOE",
        "routine_cognitive":    "Routine\nCognitive",
        "social_interpersonal": "Social",
        "physical_manual":      "Physical",
        "education_level":      "Education",
        "log_wage":             "Log Wage",
    }

    pair_df = df[scatter_vars].rename(columns=scatter_labels)
    g = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10, "color": PALETTE[0]},
                     diag_kws={"color": PALETTE[1]})
    g.fig.suptitle("Scatter Matrix: Key Variables", y=1.01, fontsize=13)
    g.fig.savefig(os.path.join(FIGURE_DIR, "scatter_matrix.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: figures/scatter_matrix.png")

    # ── Figure 3: Residual diagnostic plots ─────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    fitted   = model3.fittedvalues
    residuals = model3.resid
    std_resid = residuals / residuals.std()

    # 3a: Residuals vs Fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(fitted, residuals, alpha=0.4, s=12, color=PALETTE[0])
    ax1.axhline(0, color=PALETTE[1], linestyle="--", linewidth=1)
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs. Fitted")
    # Lowess smoother
    try:
        lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.3)
        ax1.plot(lowess[:, 0], lowess[:, 1], color=PALETTE[2], linewidth=1.5)
    except Exception:
        pass

    # 3b: Scale-Location
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.4, s=12, color=PALETTE[0])
    ax2.set_xlabel("Fitted Values")
    ax2.set_ylabel("√|Standardized Residuals|")
    ax2.set_title("Scale-Location")

    # 3c: Residual histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax3.set_xlabel("Residuals")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual Distribution")
    xmin, xmax = ax3.get_xlim()
    xs = np.linspace(xmin, xmax, 100)
    ax3.plot(xs, stats.norm.pdf(xs, residuals.mean(), residuals.std()) * len(residuals) *
             (residuals.max() - residuals.min()) / 30,
             color=PALETTE[1], linewidth=1.5, label="Normal curve")
    ax3.legend(fontsize=9)

    # 3d: Actual vs Predicted
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df["aioe_score"], fitted, alpha=0.4, s=12, color=PALETTE[0])
    lims = [min(df["aioe_score"].min(), fitted.min()),
            max(df["aioe_score"].max(), fitted.max())]
    ax4.plot(lims, lims, color=PALETTE[1], linestyle="--", linewidth=1.5, label="45° line")
    ax4.set_xlabel("Actual AIOE Score")
    ax4.set_ylabel("Predicted AIOE Score")
    ax4.set_title(f"Actual vs. Predicted (R² = {model3.rsquared:.3f})")
    ax4.legend(fontsize=9)

    fig.suptitle("OLS Diagnostic Plots — Model 3 (Full Model)", fontsize=14, y=1.01)
    fig.savefig(os.path.join(FIGURE_DIR, "residual_plots.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: figures/residual_plots.png")

    # ── Figure 4: Coefficient plot ───────────────────────────────────────────
    var_labels = {
        "routine_cognitive":    "Routine Cognitive Tasks",
        "social_interpersonal": "Social/Interpersonal Tasks",
        "physical_manual":      "Physical/Manual Tasks",
        "creative_adaptive":    "Creative/Adaptive Tasks",
        "education_level":      "Required Education Level",
        "log_wage":             "Log Annual Wage",
    }

    coef_data = []
    for var, label in var_labels.items():
        if var in model3.params.index:
            coef_data.append({
                "label": label,
                "coef":  model3.params[var],
                "ci_lo": model3.conf_int().loc[var, 0],
                "ci_hi": model3.conf_int().loc[var, 1],
                "pval":  model3.pvalues[var],
            })

    coef_df = pd.DataFrame(coef_data).sort_values("coef")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE[0] if p < 0.05 else "#AAAAAA" for p in coef_df["pval"]]

    ax.barh(coef_df["label"], coef_df["coef"], color=colors, height=0.55, zorder=3)
    ax.errorbar(
        coef_df["coef"], coef_df["label"],
        xerr=[coef_df["coef"] - coef_df["ci_lo"], coef_df["ci_hi"] - coef_df["coef"]],
        fmt="none", color="#333333", capsize=4, linewidth=1.2, zorder=4
    )
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("OLS Coefficient (HC3 Robust 95% CI)")
    ax.set_title("Predictors of AI Occupational Exposure\nModel 3 (Full Model)")
    ax.grid(axis="x", alpha=0.3, zorder=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[0], label="Significant (p < 0.05)"),
        Patch(facecolor="#AAAAAA",  label="Not significant"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "coef_plot.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: figures/coef_plot.png")

    # ── Figure 5: Top/Bottom occupations ────────────────────────────────────
    if "occupation_title" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        top15 = df.nlargest(15, "aioe_score")[["occupation_title", "aioe_score"]]
        bot15 = df.nsmallest(15, "aioe_score")[["occupation_title", "aioe_score"]]

        for ax, data, title, color in [
            (axes[0], top15, "15 Most AI-Exposed Occupations",  PALETTE[0]),
            (axes[1], bot15, "15 Least AI-Exposed Occupations", PALETTE[2]),
        ]:
            # Truncate long titles
            data = data.copy()
            data["occupation_title"] = data["occupation_title"].str[:40]
            ax.barh(data["occupation_title"], data["aioe_score"], color=color, height=0.6)
            ax.set_xlabel("AIOE Score")
            ax.set_title(title)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, "top_bottom_occupations.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Saved: figures/top_bottom_occupations.png")


# ── INTERPRETATION GUIDE ──────────────────────────────────────────────────────

def print_interpretation_guide(results, fit_stats):
    model3 = list(results.values())[2]
    mname  = list(results.keys())[2]

    print("\n" + "=" * 65)
    print("  INTERPRETATION GUIDE — Model 3 (Full Model)")
    print("=" * 65)

    var_desc = {
        "routine_cognitive":    "Routine Cognitive Task Share",
        "social_interpersonal": "Social/Interpersonal Task Share",
        "physical_manual":      "Physical/Manual Task Share",
        "creative_adaptive":    "Creative/Adaptive Task Share",
        "education_level":      "Required Education Level",
        "log_wage":             "Log Annual Wage",
    }

    print(f"\n  Dependent variable: AIOE Score (range ≈ 0 to 1)")
    print(f"  N = {int(model3.nobs)} occupations")
    print(f"  R² = {model3.rsquared:.4f}  |  Adj. R² = {model3.rsquared_adj:.4f}")
    print(f"  Standard errors: HC3 heteroskedasticity-robust\n")

    for var, label in var_desc.items():
        if var not in model3.params.index:
            continue
        coef = model3.params[var]
        pval = model3.pvalues[var]
        sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."

        direction = "increases" if coef > 0 else "decreases"
        print(f"  {label}:")
        print(f"    β = {coef:.4f} ({sig})")
        print(f"    A 1-unit increase {direction} AIOE by {abs(coef):.4f} points")
        if var == "education_level":
            print(f"    e.g., moving from HS diploma (2) to Bachelor's (6): "
                  f"ΔAIOE ≈ {coef * 4:.4f}")
        if var == "log_wage":
            print(f"    A 10% wage increase: ΔAIOE ≈ {coef * np.log(1.10):.4f}")
        print()

    print("  NOTES FOR YOUR PAPER:")
    print("  • HC3 robust SEs are used throughout because Breusch-Pagan")
    print("    tests likely detect heteroskedasticity (common in cross-")
    print("    sectional occupational data)")
    print("  • R² in this range is typical for cross-sectional OLS on")
    print("    occupational data — the unexplained variance reflects")
    print("    unobserved occupational characteristics not captured by")
    print("    O*NET task dimensions")
    print("  • For Stage 2, use Model 3 coefficients to project AIOE")
    print("    under alternative AI capability scenarios")
    print("=" * 65)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    compute_summary_stats(df)
    compute_correlations(df)
    results, fit_stats = run_regressions(df)
    compute_vif(df)
    build_regression_table(results, fit_stats)
    make_figures(df, results)
    print_interpretation_guide(results, fit_stats)

    print(f"\n✓ All outputs saved to '{OUTPUT_DIR}/'")
    print(f"  tables/ — regression_table.csv, summary_stats.csv, correlation_matrix.csv")
    print(f"  figures/ — aioe_distribution.png, scatter_matrix.png,")
    print(f"             residual_plots.png, coef_plot.png, top_bottom_occupations.png")
    print(f"\nNext: run stage2_scenarios.py")


if __name__ == "__main__":
    main()
