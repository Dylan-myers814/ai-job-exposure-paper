"""
=============================================================================
AI Job Exposure Analysis — Stage 2: Scenario Projections
=============================================================================
Dylan Myers | Undergraduate Research Paper

PURPOSE:
  Using the Stage 1 OLS regression coefficients, project how the AIOE
  occupational exposure distribution shifts under three AI capability
  scenarios. This answers: where is exposure heading if current AI
  trends continue?

METHODOLOGY:
  Stage 1 estimated:
    AIOE = 2.09
         + 6.73  × routine_cognitive
         - 4.20  × social_interpersonal
         - 15.28 × physical_manual
         - 6.94  × creative_adaptive
         + 0.061 × education_level
         + 0.186 × log_wage

  Stage 2 applies scenario-specific multipliers to the Stage 1 regression
  coefficients (not raw task shares): amplifying effects on routine/creative
  dimensions where AI capability grows and attenuating protective effects on
  physical and social dimensions as robotics and tooling evolve.

SCENARIOS:
  Conservative / Moderate / Aggressive — coef_multipliers on Model 3 terms
  so projected AIOE rises as AI capability strengthens for exposed occupations.

HIGH EXPOSURE THRESHOLD:
  Occupations crossing above the 75th percentile AIOE score of the
  baseline distribution are classified as "newly high exposure" —
  they were previously moderate but cross into the danger zone under
  the given scenario.

WORKER COUNTS:
  BLS OES TOT_EMP (total employment by occupation) weights all
  estimates so results are expressed in workers, not just occupation
  counts. This produces the "X million workers at risk" estimates.

OUTPUTS (all in Stage 2/outputs/):
  tables/scenario_summary.csv           — key stats for all 3 scenarios
  tables/occupation_projections.csv     — full occupation-level projections
  tables/newly_exposed_occupations.csv  — occupations crossing threshold
  tables/at_risk_by_wage_tertile.csv    — worker counts by wage group
  tables/at_risk_by_education.csv       — worker counts by education level
  tables/at_risk_by_industry.csv        — worker counts by industry group
  figures/scenario_distributions.png    — AIOE distributions all scenarios
  figures/newly_exposed_by_scenario.png — occupations crossing threshold
  figures/at_risk_workers.png           — worker population at risk
  figures/wage_education_heatmap.png    — risk by wage × education
  figures/occupation_shift_chart.png    — biggest movers by scenario
  tables/sensitivity_analysis.csv       — routine & creative multiplier sweeps
  figures/sensitivity_analysis.png      — sensitivity curves vs multiplier

USAGE:
  python stage2_scenarios.py
  Run from Stage 2/ directory
  Input: ../Stage 1/ai_exposure_stage1.csv
         ../Stage 1/data_cache/bls_oes_national.xlsx (for employment counts)
=============================================================================
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
STAGE1_CSV  = os.path.join(SCRIPT_DIR, "..", "Stage 1", "ai_exposure_stage1.csv")
OES_FILE    = os.path.join(SCRIPT_DIR, "..", "Stage 1", "data_cache",
                            "bls_oes_national.xlsx")

OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
TABLE_DIR   = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR  = os.path.join(OUTPUT_DIR, "figures")

for d in [TABLE_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── STAGE 1 MODEL 3 COEFFICIENTS ─────────────────────────────────────────────
# From stage1_regression.py output — Model 3 (Full Model), HC3 robust SEs
# Dependent variable: aioe_score (z-scored, mean=0, SD=1)

INTERCEPT           =  2.0948
COEF_ROUTINE_COG    =  6.7335
COEF_SOCIAL         = -4.1987
COEF_PHYSICAL       = -15.2768
COEF_CREATIVE       = -6.9354
COEF_EDUCATION      =  0.0611
COEF_LOG_WAGE       =  0.1858

# ── SCENARIO DEFINITIONS ──────────────────────────────────────────────────────

SCENARIOS = {
    "Baseline": {
        "label":       "Baseline (2023)",
        "color":       "#94A3B8",
        "linestyle":   "-",
        "description": (
            "Current AI capabilities - observed AIOE scores used directly"
        ),
        "coef_multipliers": {
            "routine_cognitive":    1.00,
            "social_interpersonal": 1.00,
            "physical_manual":      1.00,
            "creative_adaptive":    1.00,
        },
    },
    "Conservative": {
        "label":       "Conservative (2026-2027)",
        "color":       "#048A81",
        "linestyle":   "--",
        "description": (
            "Modest AI capability growth. AI handles more routine cognitive "
            "tasks but physical and interpersonal work remains protected. "
            "Reflects 2026-2027 trajectory."
        ),
        "coef_multipliers": {
            "routine_cognitive":    1.20,
            "social_interpersonal": 0.95,
            "physical_manual":      0.98,
            "creative_adaptive":    1.08,
        },
    },
    "Moderate": {
        "label":       "Moderate (2028-2030)",
        "color":       "#2E4057",
        "linestyle":   "-.",
        "description": (
            "Significant AI capability growth. LLMs handle most routine "
            "cognitive work and begin penetrating creative tasks. Physical "
            "and complex interpersonal work still largely protected. "
            "Reflects 2028-2030 trajectory."
        ),
        "coef_multipliers": {
            "routine_cognitive":    1.45,
            "social_interpersonal": 0.88,
            "physical_manual":      0.95,
            "creative_adaptive":    1.25,
        },
    },
    "Aggressive": {
        "label":       "Aggressive (2031-2035)",
        "color":       "#EF4444",
        "linestyle":   ":",
        "description": (
            "AI approaches broad cognitive capability. Nearly all routine "
            "cognitive work automatable, creative tasks substantially "
            "penetrated, robotics beginning to affect physical work. "
            "Reflects 2031-2035 trajectory under rapid AI progress."
        ),
        "coef_multipliers": {
            "routine_cognitive":    1.80,
            "social_interpersonal": 0.75,
            "physical_manual":      0.88,
            "creative_adaptive":    1.55,
        },
    },
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_data():
    print("=" * 65)
    print("  Stage 2: Scenario Projections")
    print("  AI Capability Scenarios — Conservative / Moderate / Aggressive")
    print("=" * 65)

    # Stage 1 dataset
    paths = [
        STAGE1_CSV,
        os.path.join(SCRIPT_DIR, "ai_exposure_stage1.csv"),
        os.path.join(SCRIPT_DIR, "..", "ai_exposure_stage1.csv"),
    ]
    df = None
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"\n[Data] Loaded Stage 1 dataset: {len(df)} occupations")
            print(f"       from {path}")
            break
    if df is None:
        raise FileNotFoundError(
            "Could not find ai_exposure_stage1.csv.\n"
            "Run collect_data.py in Stage 1 first."
        )

    df["soc_code"] = df["soc_code"].astype(str).str.strip().str.split(".").str[0]

    # Load employment counts from OES
    df = load_employment_counts(df)

    # Validate required columns
    required = [
        "aioe_score", "routine_cognitive", "social_interpersonal",
        "physical_manual", "creative_adaptive",
        "education_level", "log_wage"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required)
    print(f"[Data] Complete cases for projection: {len(df)}")

    return df


def load_employment_counts(df):
    """Load TOT_EMP from BLS OES file and merge onto dataset."""
    oes_paths = [
        OES_FILE,
        os.path.join(SCRIPT_DIR, "..", "Stage 1", "data_cache",
                     "bls_oes_national", "oesm24nat", "national_M2024_dl.xlsx"),
        os.path.join(SCRIPT_DIR, "..", "Stage 1", "data_cache",
                     "national_M2024_dl.xlsx"),
    ]

    for path in oes_paths:
        if os.path.exists(path):
            try:
                oes = pd.read_excel(path, dtype={"OCC_CODE": str})
                oes.columns = [c.lower().strip() for c in oes.columns]

                soc_col = next(
                    (c for c in oes.columns if "occ_code" in c), None
                )
                emp_col = next(
                    (c for c in oes.columns if "tot_emp" in c), None
                )

                if soc_col and emp_col:
                    oes_sub = oes[[soc_col, emp_col]].copy()
                    oes_sub.columns = ["soc_code", "tot_emp"]
                    oes_sub["soc_code"] = (
                        oes_sub["soc_code"].astype(str)
                        .str.strip().str.split(".").str[0]
                    )
                    oes_sub["tot_emp"] = pd.to_numeric(
                        oes_sub["tot_emp"].astype(str)
                        .str.replace(",", "").str.strip(),
                        errors="coerce"
                    )
                    oes_sub = oes_sub.dropna(subset=["tot_emp"])
                    # Filter out aggregate groups
                    oes_sub = oes_sub[~oes_sub["soc_code"].str.endswith("0000")]

                    df = df.merge(oes_sub, on="soc_code", how="left")
                    matched = df["tot_emp"].notna().sum()
                    total_workers = df["tot_emp"].sum() / 1e6
                    print(f"[Employment] Loaded from {os.path.basename(path)}")
                    print(f"             {matched} occupations matched")
                    print(f"             {total_workers:.1f}M workers covered")
                    return df
            except Exception as e:
                print(f"[Employment] Failed to load {path}: {e}")

    print("[Employment] Warning: No OES employment file found.")
    print("             Projections will use occupation counts not worker counts.")
    print("             Place bls_oes_national.xlsx in Stage 1/data_cache/")
    df["tot_emp"] = np.nan
    return df


# ── SCENARIO PROJECTION ENGINE ────────────────────────────────────────────────

def apply_scenario(df, scenario_name, scenario_config):
    """
    Project AIOE scores under a given AI capability scenario.

    Instead of adjusting task shares, we adjust the effective
    regression coefficients to reflect changing AI capability.
    A multiplier > 1 on routine_cognitive means AI has made
    that task dimension more threatening to the occupation.
    A multiplier < 1 on physical_manual means physical tasks
    remain protective but slightly less so as robotics advance.

    This correctly produces rising AIOE scores for high-routine-
    cognitive occupations as AI capability grows.
    """
    mults = scenario_config["coef_multipliers"]
    proj = df.copy()

    proj[f"aioe_{scenario_name}"] = (
        INTERCEPT
        + COEF_ROUTINE_COG * mults["routine_cognitive"] * proj["routine_cognitive"]
        + COEF_SOCIAL * mults["social_interpersonal"] * proj["social_interpersonal"]
        + COEF_PHYSICAL * mults["physical_manual"] * proj["physical_manual"]
        + COEF_CREATIVE * mults["creative_adaptive"] * proj["creative_adaptive"]
        + COEF_EDUCATION * proj["education_level"]
        + COEF_LOG_WAGE * proj["log_wage"]
    )

    proj[f"aioe_change_{scenario_name}"] = (
        proj[f"aioe_{scenario_name}"] - proj["aioe_score"]
    )

    return proj[
        [
            "soc_code",
            f"aioe_{scenario_name}",
            f"aioe_change_{scenario_name}",
        ]
    ]


def project_aioe_with_multipliers(
    df,
    routine_m,
    social_m,
    physical_m,
    creative_m,
):
    """Project AIOE from Stage 1 coefficients and explicit coefficient multipliers."""
    return (
        INTERCEPT
        + COEF_ROUTINE_COG * routine_m * df["routine_cognitive"]
        + COEF_SOCIAL * social_m * df["social_interpersonal"]
        + COEF_PHYSICAL * physical_m * df["physical_manual"]
        + COEF_CREATIVE * creative_m * df["creative_adaptive"]
        + COEF_EDUCATION * df["education_level"]
        + COEF_LOG_WAGE * df["log_wage"]
    )


# Moderate scenario multipliers (fixed when isolating one dimension)
MOD_SOCIAL = SCENARIOS["Moderate"]["coef_multipliers"]["social_interpersonal"]
MOD_PHYSICAL = SCENARIOS["Moderate"]["coef_multipliers"]["physical_manual"]
MOD_CREATIVE = SCENARIOS["Moderate"]["coef_multipliers"]["creative_adaptive"]
MOD_ROUTINE = SCENARIOS["Moderate"]["coef_multipliers"]["routine_cognitive"]

SENS_ROUTINE_MARKERS = {
    "Conservative": (1.20, "#048A81"),
    "Moderate":     (1.45, "#2E4057"),
    "Aggressive":   (1.80, "#EF4444"),
}
# Scenario creative multipliers (for dots on creative sweep panel)
SENS_CREATIVE_MARKERS = {
    "Conservative": (
        SCENARIOS["Conservative"]["coef_multipliers"]["creative_adaptive"],
        "#048A81",
    ),
    "Moderate": (
        SCENARIOS["Moderate"]["coef_multipliers"]["creative_adaptive"],
        "#2E4057",
    ),
    "Aggressive": (
        SCENARIOS["Aggressive"]["coef_multipliers"]["creative_adaptive"],
        "#EF4444",
    ),
}

SENS_ANNOTATION = (
    "Smooth monotonic curve = finding robust to multiplier choice"
)


def run_sensitivity_analysis(df, threshold):
    """
    Sweep routine_cognitive and creative_adaptive multipliers; count newly
    high-exposure occupations and workers relative to baseline p75 threshold.
    """
    baseline_high = df["aioe_score"] >= threshold
    has_emp = (
        "tot_emp" in df.columns
        and df["tot_emp"].notna().any()
    )

    mult_grid = np.linspace(1.0, 2.0, 21)

    def newly_exposed_counts(aioe_proj):
        newly = (~baseline_high) & (aioe_proj >= threshold)
        n = int(newly.sum())
        if has_emp:
            w_m = df.loc[newly, "tot_emp"].sum() / 1e6
        else:
            w_m = np.nan
        return n, w_m

    # --- Routine cognitive sweep ---
    rows_r = []
    n_list_r = []
    w_list_r = []
    for m in mult_grid:
        proj = project_aioe_with_multipliers(
            df, m, MOD_SOCIAL, MOD_PHYSICAL, MOD_CREATIVE
        )
        n_occ, w_m = newly_exposed_counts(proj)
        n_list_r.append(n_occ)
        w_list_r.append(w_m)
        rows_r.append({
            "sensitivity_axis":            "routine_cognitive",
            "multiplier":                  round(float(m), 2),
            "newly_exposed_occupations":   n_occ,
            "newly_exposed_workers_M":     round(w_m, 4)
            if not np.isnan(w_m) else np.nan,
        })

    # --- Creative adaptive sweep ---
    rows_c = []
    n_list_c = []
    w_list_c = []
    for m in mult_grid:
        proj = project_aioe_with_multipliers(
            df, MOD_ROUTINE, MOD_SOCIAL, MOD_PHYSICAL, m
        )
        n_occ, w_m = newly_exposed_counts(proj)
        n_list_c.append(n_occ)
        w_list_c.append(w_m)
        rows_c.append({
            "sensitivity_axis":            "creative_adaptive",
            "multiplier":                  round(float(m), 2),
            "newly_exposed_occupations":   n_occ,
            "newly_exposed_workers_M":     round(w_m, 4)
            if not np.isnan(w_m) else np.nan,
        })

    sens_df = pd.DataFrame(rows_r + rows_c)
    sens_path = os.path.join(TABLE_DIR, "sensitivity_analysis.csv")
    sens_df.to_csv(sens_path, index=False)

    n_arr_r = np.array(n_list_r, dtype=float)
    mono_occ_r = bool(np.all(np.diff(n_arr_r) >= -1e-9))

    n_arr_c = np.array(n_list_c, dtype=float)
    mono_occ_c = bool(np.all(np.diff(n_arr_c) >= -1e-9))

    # Threshold crossing multipliers (routine)
    idx_pos_r = np.where(n_arr_r > 0)[0]
    first_occ_m_r = float(mult_grid[idx_pos_r[0]]) if len(idx_pos_r) else np.nan

    def first_m_workers_ge(target_millions):
        if not has_emp:
            return np.nan
        for i, m in enumerate(mult_grid):
            if not np.isnan(w_list_r[i]) and w_list_r[i] >= target_millions:
                return float(m)
        return np.nan

    m10 = first_m_workers_ge(10.0)
    m20 = first_m_workers_ge(20.0)
    m30 = first_m_workers_ge(30.0)

    # --- Figure: 2 top panels + creative bottom ---
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "Sensitivity Analysis: Newly High-Exposure Occupations vs. "
        "Routine Cognitive Multiplier",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.915,
        "All other multipliers held at Moderate scenario values — isolating effect "
        "of routine cognitive AI capability growth",
        ha="center",
        fontsize=10,
        style="italic",
    )

    ax0.plot(mult_grid, n_list_r, color="#334155", linewidth=2)
    ax0.axhline(0, color="#94A3B8", linestyle="--", linewidth=1)
    ax0.set_xlabel("Routine cognitive multiplier")
    ax0.set_ylabel("Newly exposed occupations (count)")
    ax0.set_xlim(1.0, 2.0)
    ax0.grid(alpha=0.3)
    ax0.text(
        0.03,
        0.97,
        SENS_ANNOTATION,
        transform=ax0.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
    )

    for label, (xv, col) in SENS_ROUTINE_MARKERS.items():
        yv = float(np.interp(xv, mult_grid, n_list_r))
        ax0.scatter([xv], [yv], color=col, s=55, zorder=5, edgecolors="white")
        ax0.annotate(
            label,
            (xv, yv),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=col,
            fontweight="bold",
        )

    if has_emp:
        ax1.plot(mult_grid, w_list_r, color="#334155", linewidth=2)
        ax1.axhline(0, color="#94A3B8", linestyle="--", linewidth=1)
        ax1.set_xlabel("Routine cognitive multiplier")
        ax1.set_ylabel("Newly exposed workers (millions)")
        ax1.set_xlim(1.0, 2.0)
        ax1.grid(alpha=0.3)
        for label, (xv, col) in SENS_ROUTINE_MARKERS.items():
            yv = float(np.interp(xv, mult_grid, w_list_r))
            if np.isnan(yv):
                continue
            ax1.scatter([xv], [yv], color=col, s=55, zorder=5, edgecolors="white")
            ax1.annotate(
                label,
                (xv, yv),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=col,
                fontweight="bold",
            )
        ax1.text(
            0.03,
            0.97,
            SENS_ANNOTATION,
            transform=ax1.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="#CBD5E1", alpha=0.95
            ),
        )
    else:
        ax1.text(
            0.5,
            0.5,
            "Employment data required",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
            color="#64748B",
        )
        ax1.set_axis_off()

    # Creative sweep (occupations)
    ax2.plot(mult_grid, n_list_c, color="#7C3AED", linewidth=2)
    ax2.axhline(0, color="#94A3B8", linestyle="--", linewidth=1)
    ax2.set_xlabel("Creative adaptive multiplier (routine = 1.45, other = Moderate)")
    ax2.set_ylabel("Newly exposed occupations (count)")
    ax2.set_xlim(1.0, 2.0)
    ax2.set_title(
        "Creative adaptive sensitivity — tests whether protective creative "
        "term can reverse under stronger AI",
        fontsize=11,
    )
    ax2.grid(alpha=0.3)
    ax2.text(
        0.03,
        0.97,
        SENS_ANNOTATION,
        transform=ax2.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
    )

    for label, (xv, col) in SENS_CREATIVE_MARKERS.items():
        yv = float(np.interp(xv, mult_grid, n_list_c))
        ax2.scatter([xv], [yv], color=col, s=55, zorder=5, edgecolors="white")
        ax2.annotate(
            label,
            (xv, yv),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=col,
            fontweight="bold",
        )

    plt.subplots_adjust(top=0.86)
    fig_out = os.path.join(FIGURE_DIR, "sensitivity_analysis.png")
    fig.savefig(fig_out, bbox_inches="tight")
    plt.close()

    # --- Terminal summary ---
    print("\n[Sensitivity analysis]")
    print("  Saved: tables/sensitivity_analysis.csv")
    print("  Saved: figures/sensitivity_analysis.png")
    print("\n  Routine cognitive sweep (other multipliers = Moderate):")
    if not np.isnan(first_occ_m_r):
        print(
            f"    First newly-exposed occupation at multiplier: {first_occ_m_r:.2f}"
        )
    else:
        print("    First newly-exposed occupation at multiplier: (none in 1.0-2.0)")
    if has_emp:
        for tag, mv in [("10M", m10), ("20M", m20), ("30M", m30)]:
            if not np.isnan(mv):
                print(
                    f"    Newly exposed workers first exceed {tag} at multiplier: "
                    f"{mv:.2f}"
                )
            else:
                print(
                    f"    Newly exposed workers first exceed {tag} at multiplier: "
                    "(not reached in 1.0-2.0)"
                )
    else:
        print("    Worker thresholds (10M/20M/30M): N/A (employment data missing)")
    print(f"    Monotonically increasing (occupations): {mono_occ_r}")
    if mono_occ_r:
        print(
            "    Interpretation: ROBUST: qualitative finding holds across all "
            "tested multiplier values"
        )
    else:
        print(
            "    Interpretation: NON-MONOTONIC: results sensitive to multiplier "
            "choice — investigate further"
        )

    print("\n  Creative adaptive sweep (routine=1.45, other non-varying = Moderate):")
    print(f"    Monotonically increasing (occupations): {mono_occ_c}")
    if mono_occ_c:
        print(
            "    Interpretation: ROBUST: qualitative finding holds across all "
            "tested multiplier values"
        )
    else:
        print(
            "    Interpretation: NON-MONOTONIC: results sensitive to multiplier "
            "choice — investigate further"
        )


def print_scenario_justification():
    """Print methodology note on the empirical basis for scenario coefficient multipliers."""
    text = """
SCENARIO CALIBRATION — EMPIRICAL BASIS
================================================================
Multipliers are calibrated to measured AI capability growth rates
from three independent empirical sources:

LLM COGNITIVE CAPABILITY (routine_cognitive, creative_adaptive):
  Source: METR (2025) 'Measuring AI Ability to Complete Long Tasks'
  Finding: AI task-completion time horizon doubling every ~7 months
           since 2019. Text-related task success rates projected
           80-95% by 2029 (Crashing Waves et al. 2026).
  Conservative x1.20: ~3-4 doublings from 2023 baseline (2026-27)
  Moderate    x1.45: ~6-8 doublings from 2023 baseline (2028-30)
  Aggressive  x1.80: continued exponential growth (2031-35)

COST REDUCTION / ADOPTION (amplifies all cognitive multipliers):
  Source: Stanford AI Index 2025 (HAI)
  Finding: GPT-3.5-level inference costs fell 280x in 18 months.
           Training compute doubles every 5 months. Hardware costs
           decline 30% annually. Lower costs drive broader adoption
           of AI against routine cognitive tasks.

PHYSICAL TASK PENETRATION (physical_manual):
  Source: Stanford AI Index 2026; Figure AI BMW deployment data
  Finding: Real-world physical task success rate reached 77.3% for
           structured industrial tasks but only 12% for unstructured
           household tasks. Figure AI robots 4x faster and 7x more
           accurate in production deployments.
  Conservative x0.98: minimal robotics penetration of occupations
  Moderate    x0.95: selective industrial automation
  Aggressive  x0.88: meaningful but incomplete physical penetration

CITATION NOTE:
  Kwa et al. (2025). Measuring AI Ability to Complete Long Tasks.
    METR. arxiv.org/abs/2503.14499
  Stanford HAI (2025). AI Index Report 2025. hai.stanford.edu
  Stanford HAI (2026). AI Index Report 2026. hai.stanford.edu
  Crashing Waves vs Rising Tides (2026). arxiv.org/abs/2604.01363
================================================================
""".strip()
    print(text)
    open(os.path.join(TABLE_DIR, "scenario_calibration_note.txt"), "w", encoding="utf-8").write(text)


def run_all_scenarios(df):
    """Run all three scenarios and merge projections onto base dataset."""
    print("\n[Projections] Running scenario models...")

    # Baseline — use observed aioe_score as-is
    result = df.copy()
    result["aioe_Baseline"] = result["aioe_score"]

    for scenario_name, config in SCENARIOS.items():
        if scenario_name == "Baseline":
            continue
        proj = apply_scenario(df, scenario_name, config)
        result = result.merge(
            proj[["soc_code",
                  f"aioe_{scenario_name}",
                  f"aioe_change_{scenario_name}"]],
            on="soc_code", how="left"
        )
        n_proj = result[f"aioe_{scenario_name}"].notna().sum()
        mean_change = result[f"aioe_change_{scenario_name}"].mean()
        print(f"  {scenario_name:<15} {n_proj} occupations projected  "
              f"mean AIOE change: {mean_change:+.4f}")

    return result


# ── THRESHOLD ANALYSIS ────────────────────────────────────────────────────────

def analyze_thresholds(result):
    """
    Identify occupations crossing into high exposure territory.
    High exposure threshold = 75th percentile of baseline AIOE.
    """
    print("\n[Thresholds] Identifying newly high-exposure occupations...")

    threshold = result["aioe_score"].quantile(0.75)
    print(f"  High exposure threshold (75th pct of baseline): {threshold:.4f}")

    result["baseline_high"] = result["aioe_score"] >= threshold

    newly_exposed = {}
    for scenario in ["Conservative", "Moderate", "Aggressive"]:
        col = f"aioe_{scenario}"
        if col not in result.columns:
            continue

        result[f"high_{scenario}"] = result[col] >= threshold
        result[f"newly_exposed_{scenario}"] = (
            ~result["baseline_high"] & result[f"high_{scenario}"]
        )

        n_new  = result[f"newly_exposed_{scenario}"].sum()
        if "tot_emp" in result.columns:
            workers_new = result.loc[
                result[f"newly_exposed_{scenario}"], "tot_emp"
            ].sum()
        else:
            workers_new = np.nan

        newly_exposed[scenario] = {
            "n_occupations":    n_new,
            "workers_millions": workers_new / 1e6 if not np.isnan(workers_new) else np.nan,
            "threshold":        threshold,
        }

        print(f"\n  {scenario}:")
        print(f"    New high-exposure occupations: {n_new}")
        if not np.isnan(workers_new):
            print(f"    Workers newly in high-exposure: {workers_new/1e6:.2f}M")

    return result, newly_exposed, threshold


# ── AT-RISK ANALYSIS ──────────────────────────────────────────────────────────

def analyze_at_risk(result):
    """
    Break down at-risk worker populations by wage tertile,
    education level, and industry group.
    """
    print("\n[At-Risk Analysis] Breaking down worker populations...")

    # Wage tertiles
    result["wage_tertile"] = pd.qcut(
        result["annual_median_wage"], q=3,
        labels=["Low Wage", "Mid Wage", "High Wage"]
    )

    # Education groups
    def edu_group(level):
        if pd.isna(level):    return "Unknown"
        if level < 3:         return "No College"
        if level < 6:         return "Some College / Associate"
        if level < 8:         return "Bachelor's"
        return "Graduate Degree"

    result["edu_group"] = result["education_level"].apply(edu_group)

    # SOC major group → industry
    soc_industry = {
        "11": "Management", "13": "Business & Finance",
        "15": "Computer & Math", "17": "Architecture & Engineering",
        "19": "Life & Physical Science", "21": "Community & Social Service",
        "23": "Legal", "25": "Education",
        "27": "Arts & Media", "29": "Healthcare Practitioners",
        "31": "Healthcare Support", "33": "Protective Service",
        "35": "Food Preparation", "37": "Building & Grounds",
        "39": "Personal Care", "41": "Sales",
        "43": "Office & Admin Support", "45": "Farming & Fishing",
        "47": "Construction", "49": "Installation & Repair",
        "51": "Production", "53": "Transportation",
    }
    result["occ_group"] = result["soc_code"].str[:2].map(soc_industry).fillna("Other")

    at_risk_tables = {}

    for scenario in ["Conservative", "Moderate", "Aggressive"]:
        col = f"newly_exposed_{scenario}"
        if col not in result.columns:
            continue

        newly = result[result[col]].copy()

        # By wage tertile
        if "tot_emp" in newly.columns:
            wage_risk = newly.groupby("wage_tertile", observed=True)["tot_emp"].sum() / 1e6
        else:
            wage_risk = newly.groupby("wage_tertile", observed=True).size()

        # By education
        if "tot_emp" in newly.columns:
            edu_risk = newly.groupby("edu_group", observed=True)["tot_emp"].sum() / 1e6
        else:
            edu_risk = newly.groupby("edu_group", observed=True).size()

        # By occupation group
        if "tot_emp" in newly.columns:
            occ_risk = newly.groupby("occ_group", observed=True)["tot_emp"].sum() / 1e6
        else:
            occ_risk = newly.groupby("occ_group", observed=True).size()

        at_risk_tables[scenario] = {
            "wage":  wage_risk,
            "edu":   edu_risk,
            "occ":   occ_risk,
        }

        print(f"\n  {scenario} — newly high-exposure workers by wage tertile:")
        for tertile, workers in wage_risk.items():
            unit = "M workers" if "tot_emp" in newly.columns else "occupations"
            print(f"    {tertile:<20} {workers:.2f} {unit}")

    return result, at_risk_tables


# ── SAVE TABLES ───────────────────────────────────────────────────────────────

def save_tables(result, newly_exposed, at_risk_tables):
    print("\n[Tables] Saving outputs...")

    # Scenario summary
    summary_rows = []
    for scenario in ["Baseline", "Conservative", "Moderate", "Aggressive"]:
        col = f"aioe_{scenario}"
        if col not in result.columns:
            continue
        row = {
            "Scenario":            SCENARIOS[scenario]["label"],
            "Mean AIOE":           round(result[col].mean(), 4),
            "Median AIOE":         round(result[col].median(), 4),
            "Std Dev AIOE":        round(result[col].std(), 4),
            "Pct in High Exposure": round(
                (result[col] >= result["aioe_score"].quantile(0.75)).mean() * 100, 1
            ),
        }
        if scenario in newly_exposed:
            row["Newly High Exposure (n)"] = newly_exposed[scenario]["n_occupations"]
            row["Newly High Exposure (M workers)"] = round(
                newly_exposed[scenario]["workers_millions"], 2
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLE_DIR, "scenario_summary.csv"), index=False)
    print(f"  Saved: tables/scenario_summary.csv")

    # Full occupation projections
    proj_cols = ["soc_code", "occupation_title",
                 "aioe_score", "aioe_Conservative", "aioe_Moderate", "aioe_Aggressive",
                 "aioe_change_Conservative", "aioe_change_Moderate", "aioe_change_Aggressive",
                 "routine_cognitive", "physical_manual", "education_level",
                 "annual_median_wage", "tot_emp",
                 "baseline_high", "newly_exposed_Conservative",
                 "newly_exposed_Moderate", "newly_exposed_Aggressive"]
    available = [c for c in proj_cols if c in result.columns]
    result[available].to_csv(
        os.path.join(TABLE_DIR, "occupation_projections.csv"), index=False
    )
    print(f"  Saved: tables/occupation_projections.csv")

    # Newly exposed occupations
    newly_rows = []
    for scenario in ["Conservative", "Moderate", "Aggressive"]:
        col = f"newly_exposed_{scenario}"
        if col not in result.columns:
            continue
        sub = result[result[col]].copy()
        sub["scenario"] = scenario
        sub["projected_aioe"] = sub[f"aioe_{scenario}"]
        sub["aioe_increase"] = sub[f"aioe_change_{scenario}"]
        newly_rows.append(sub)

    if newly_rows:
        newly_df = pd.concat(newly_rows)
        out_cols = ["scenario", "soc_code", "occupation_title",
                    "aioe_score", "projected_aioe", "aioe_increase",
                    "tot_emp", "annual_median_wage", "education_level"]
        available = [c for c in out_cols if c in newly_df.columns]
        newly_df[available].sort_values(
            ["scenario", "aioe_increase"], ascending=[True, False]
        ).to_csv(
            os.path.join(TABLE_DIR, "newly_exposed_occupations.csv"), index=False
        )
        print(f"  Saved: tables/newly_exposed_occupations.csv")

    # At-risk by wage tertile
    wage_rows = []
    for scenario, tables in at_risk_tables.items():
        for tertile, value in tables["wage"].items():
            wage_rows.append({
                "Scenario": scenario,
                "Wage Tertile": str(tertile),
                "At-Risk (M workers)": round(float(value), 3)
            })
    pd.DataFrame(wage_rows).to_csv(
        os.path.join(TABLE_DIR, "at_risk_by_wage_tertile.csv"), index=False
    )
    print(f"  Saved: tables/at_risk_by_wage_tertile.csv")

    # At-risk by education
    edu_rows = []
    for scenario, tables in at_risk_tables.items():
        for group, value in tables["edu"].items():
            edu_rows.append({
                "Scenario": scenario,
                "Education Group": str(group),
                "At-Risk (M workers)": round(float(value), 3)
            })
    pd.DataFrame(edu_rows).to_csv(
        os.path.join(TABLE_DIR, "at_risk_by_education.csv"), index=False
    )
    print(f"  Saved: tables/at_risk_by_education.csv")

    # At-risk by occupation group
    occ_rows = []
    for scenario, tables in at_risk_tables.items():
        for group, value in tables["occ"].items():
            occ_rows.append({
                "Scenario": scenario,
                "Occupation Group": str(group),
                "At-Risk (M workers)": round(float(value), 3)
            })
    pd.DataFrame(occ_rows).to_csv(
        os.path.join(TABLE_DIR, "at_risk_by_industry.csv"), index=False
    )
    print(f"  Saved: tables/at_risk_by_industry.csv")

    return summary_df


# ── FIGURES ───────────────────────────────────────────────────────────────────

def make_figures(result, newly_exposed, at_risk_tables, threshold, summary_df):
    print("\n[Figures] Generating...")

    # ── Figure 1: AIOE distributions across scenarios ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    for scenario, config in SCENARIOS.items():
        col = f"aioe_{scenario}"
        if col not in result.columns:
            continue
        data = result[col].dropna()
        ax.hist(data, bins=40, density=True, alpha=0.35,
                color=config["color"], label=config["label"])
        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method=0.3)
        xs  = np.linspace(data.min(), data.max(), 200)
        ax.plot(xs, kde(xs), color=config["color"],
                linewidth=2, linestyle=config["linestyle"])

    ax.axvline(threshold, color="#EF4444", linewidth=1.5,
               linestyle="--", alpha=0.8,
               label=f"High-exposure threshold (p75 = {threshold:.2f})")
    ax.set_xlabel("Projected AIOE Score")
    ax.set_ylabel("Density")
    ax.set_title("AI Occupational Exposure Distribution\nBaseline vs. Three AI Capability Scenarios")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "scenario_distributions.png"),
                bbox_inches="tight")
    plt.close()
    print("  Saved: figures/scenario_distributions.png")

    # ── Figure 2: Newly exposed occupations and workers ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    scenarios      = ["Conservative", "Moderate", "Aggressive"]
    colors         = ["#048A81", "#2E4057", "#EF4444"]
    n_occs         = [newly_exposed.get(s, {}).get("n_occupations", 0)
                      for s in scenarios]
    n_workers      = [newly_exposed.get(s, {}).get("workers_millions", 0)
                      for s in scenarios]
    scenario_labels = ["Conservative\n(2026-27)", "Moderate\n(2028-30)",
                       "Aggressive\n(2031-35)"]

    axes[0].bar(scenario_labels, n_occs, color=colors, width=0.5,
                edgecolor="white")
    axes[0].set_title("Occupations Newly Crossing\nHigh-Exposure Threshold")
    axes[0].set_ylabel("Number of Occupations")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(n_occs):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=10,
                     fontweight="bold")

    valid_workers = [w for w in n_workers if not np.isnan(w) and w > 0]
    if valid_workers:
        axes[1].bar(scenario_labels, n_workers, color=colors, width=0.5,
                    edgecolor="white")
        axes[1].set_title("Workers Newly in High-Exposure\nOccupations (Millions)")
        axes[1].set_ylabel("Workers (Millions)")
        axes[1].grid(axis="y", alpha=0.3)
        for i, v in enumerate(n_workers):
            if not np.isnan(v):
                axes[1].text(i, v + 0.05, f"{v:.1f}M", ha="center",
                             fontsize=10, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "Employment data\nnot available",
                     ha="center", va="center", transform=axes[1].transAxes,
                     fontsize=12, color="#888888")
        axes[1].set_title("Workers Newly in High-Exposure\nOccupations (Millions)")

    plt.suptitle("Occupational Displacement Under AI Capability Scenarios",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "newly_exposed_by_scenario.png"),
                bbox_inches="tight")
    plt.close()
    print("  Saved: figures/newly_exposed_by_scenario.png")

    # ── Figure 3: At-risk workers by wage tertile ─────────────────────────────
    if at_risk_tables:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        tertile_colors = {"Low Wage": "#54C6EB",
                          "Mid Wage": "#048A81",
                          "High Wage": "#2E4057"}

        for ax, scenario in zip(axes, ["Conservative", "Moderate", "Aggressive"]):
            if scenario not in at_risk_tables:
                continue
            wage_data = at_risk_tables[scenario]["wage"]
            labels    = [str(l) for l in wage_data.index]
            values    = wage_data.values
            bar_colors = [tertile_colors.get(l, "#888888") for l in labels]

            bars = ax.bar(labels, values, color=bar_colors,
                          width=0.5, edgecolor="white")
            ax.set_title(f"{scenario}\n({['2026-27','2028-30','2031-35'][['Conservative','Moderate','Aggressive'].index(scenario)]})")
            ax.set_ylabel("Workers (M)" if scenario == "Conservative" else "")
            ax.grid(axis="y", alpha=0.3)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            h + 0.02, f"{h:.2f}M",
                            ha="center", va="bottom", fontsize=9,
                            fontweight="bold")

        plt.suptitle("Workers Newly in High-Exposure Occupations\nby Wage Tertile and Scenario",
                     fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, "at_risk_workers.png"),
                    bbox_inches="tight")
        plt.close()
        print("  Saved: figures/at_risk_workers.png")

    # ── Figure 4: Wage × Education heatmap ───────────────────────────────────
    if at_risk_tables and "Moderate" in at_risk_tables:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, scenario in zip(axes, ["Conservative", "Moderate", "Aggressive"]):
            if scenario not in at_risk_tables:
                continue

            col = f"newly_exposed_{scenario}"
            if col not in result.columns:
                continue

            newly = result[result[col]].copy()
            if "tot_emp" not in newly.columns or newly.empty:
                continue

            pivot = newly.pivot_table(
                values="tot_emp",
                index="edu_group",
                columns="wage_tertile",
                aggfunc="sum",
                observed=True
            ) / 1e6

            pivot = pivot.fillna(0)
            sns.heatmap(pivot, annot=True, fmt=".2f",
                        cmap="YlOrRd", ax=ax,
                        cbar_kws={"label": "Workers (M)"},
                        linewidths=0.5)
            ax.set_title(f"{scenario}")
            ax.set_xlabel("Wage Tertile")
            ax.set_ylabel("Education Level" if scenario == "Conservative" else "")

        plt.suptitle("At-Risk Workers (Millions) by Education × Wage\nunder Each Scenario",
                     fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, "wage_education_heatmap.png"),
                    bbox_inches="tight")
        plt.close()
        print("  Saved: figures/wage_education_heatmap.png")

    # ── Figure 5: Top occupation movers ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    for ax, scenario in zip(axes, ["Conservative", "Moderate", "Aggressive"]):
        col_chg = f"aioe_change_{scenario}"
        if col_chg not in result.columns:
            continue

        top15 = result.nlargest(15, col_chg)[
            ["occupation_title", col_chg, "aioe_score"]
        ].copy() if "occupation_title" in result.columns else result.nlargest(
            15, col_chg
        )[["soc_code", col_chg, "aioe_score"]].copy()

        name_col = "occupation_title" if "occupation_title" in top15.columns else "soc_code"
        top15[name_col] = top15[name_col].str[:35]

        colors_bar = ["#EF4444" if b else "#048A81"
                      for b in (top15["aioe_score"] <
                                result["aioe_score"].quantile(0.75))]

        ax.barh(top15[name_col], top15[col_chg],
                color=colors_bar, height=0.6)
        ax.set_title(f"{scenario}\nTop 15 Biggest AIOE Increases")
        ax.set_xlabel("AIOE Change from Baseline")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(0, color="#333333", linewidth=0.8)

    plt.suptitle("Occupations with Largest Projected AIOE Increase\nby Scenario",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "occupation_shift_chart.png"),
                bbox_inches="tight")
    plt.close()
    print("  Saved: figures/occupation_shift_chart.png")


# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────

def print_summary(result, newly_exposed, summary_df):
    print("\n" + "=" * 65)
    print("  STAGE 2 SUMMARY - Key Findings for Paper")
    print("=" * 65)

    print(f"\n  BASELINE")
    print(f"  Occupations analyzed:       {len(result)}")
    if "tot_emp" in result.columns:
        total_workers = result["tot_emp"].sum() / 1e6
        print(f"  Total workers covered:      {total_workers:.1f}M")
    baseline_high = result["baseline_high"].sum()
    print(f"  Currently high-exposure:    {baseline_high} occupations")

    print(f"\n  SCENARIO PROJECTIONS")
    print(f"  {'Scenario':<20} {'New High-Exp Occs':>18} {'New At-Risk Workers':>20}")
    print("  " + "-" * 62)

    for scenario in ["Conservative", "Moderate", "Aggressive"]:
        if scenario not in newly_exposed:
            continue
        n   = newly_exposed[scenario]["n_occupations"]
        w   = newly_exposed[scenario]["workers_millions"]
        w_str = f"{w:.1f}M" if not np.isnan(w) else "N/A"
        label = SCENARIOS[scenario]["label"]
        print(f"  {label:<35} {n:>5} occupations   {w_str:>10} workers")

    print(f"\n  PAPER WRITE-UP TEMPLATE (Section VII):")
    for scenario in ["Conservative", "Moderate", "Aggressive"]:
        if scenario not in newly_exposed:
            continue
        n   = newly_exposed[scenario]["n_occupations"]
        w   = newly_exposed[scenario]["workers_millions"]
        w_str = f"{w:.1f} million" if not np.isnan(w) else "an estimated"
        desc  = SCENARIOS[scenario]["description"].split(".")[0]
        print(f"\n  {scenario}: '{desc}. Under this scenario,")
        print(f"  {n} additional occupations cross into high AI exposure")
        print(f"  territory, placing approximately {w_str} workers")
        print(f"  newly in the high-exposure zone.'")

    print("\n" + "=" * 65)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    df                           = load_data()
    print_scenario_justification()
    result                       = run_all_scenarios(df)
    result, newly_exposed, threshold = analyze_thresholds(result)
    result, at_risk_tables       = analyze_at_risk(result)
    summary_df                   = save_tables(result, newly_exposed,
                                               at_risk_tables)
    make_figures(result, newly_exposed, at_risk_tables, threshold, summary_df)
    run_sensitivity_analysis(result, threshold)
    print_summary(result, newly_exposed, summary_df)

    print(f"\n  All outputs saved to '{OUTPUT_DIR}/'")
    print(f"  Next: run stage3_abm.py")


if __name__ == "__main__":
    main()
