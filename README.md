# AI Job Exposure in the U.S. Economy
### A Speculative Regression Framework with Validation, Case Studies, and Agent-Based Scenario Modeling

**Author:** Dylan Myers  
**Institution:** Undergraduate Research Paper  
**GitHub:** [github.com/Dylan-myers814](https://github.com/Dylan-myers814)  
**Status:** 🔬 Active Development — Stage 1 in progress

---

## What This Project Is

This paper asks a question that most AI-and-labor research gets backwards:

> Instead of *"does AI exposure cause unemployment?"*, we ask *"what structural features of an occupation make it exposed to AI in the first place?"*

That reversal is the core methodological contribution. We treat AI occupational exposure scores as an **endogenous outcome** predicted by task content, education requirements, and wage level — then validate those predictions against real labor market data and documented corporate layoffs before projecting forward under alternative AI capability scenarios.

---

## Paper Structure & Argument Flow

The paper builds its argument in five sequential components. Each earns the right to make the next claim.

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: What structurally predicts AI exposure?               │
│  OLS regression → AIOE score as dependent variable              │
│  Script: stage1_regression.py                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ coefficients + exposure rankings
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATION 1: Do high-exposure occupations show real           │
│  labor market deterioration?                                    │
│  AIOE scores × BLS employment projections + JOLTS trends        │
│  Script: validation.py                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ corroborating real-world evidence
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATION 2: Are the layoffs happening in the roles           │
│  our model flags as most exposed?                               │
│  Corporate layoff case studies × AIOE scores                    │
│  Script: case_studies.py                                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ grounded, credible exposure measure
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Where is exposure heading as AI capabilities grow?    │
│  Scenario projections (conservative / moderate / aggressive)    │
│  Script: stage2_scenarios.py                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │ projected exposure distributions
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: What does this mean for employment dynamics?          │
│  Agent-based model + international policy comparison            │
│  Script: stage3_abm.py                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Full Paper Outline

### I. Introduction
- The open question: what makes an occupation structurally vulnerable to AI?
- Why reversing the causal arrow matters
- Paper roadmap and contributions

### II. Literature Review
- Measurement: Felten et al. (2021), Eloundou et al. (2023)
- Theory: Acemoglu & Restrepo (2018, 2019, 2020)
- Empirical evidence: Goldman Sachs (2025), Dallas Fed (2026), Frank et al. (2025)
- Gaps motivating this analysis

### III. Data & Methods
- Data sources and merge procedure (SOC code linkage)
- Variable construction (task dimensions, education, wage)
- OLS specification and robustness approach

### IV. Stage 1 — Regression Results
- Model 1–4 coefficient tables
- Coefficient plot (Model 3)
- VIF and diagnostic discussion
- Interpretation: anatomy of occupational vulnerability

### V. Validation 1 — BLS Labor Market Corroboration
- BLS 10-year occupational employment projections × AIOE scores
- Correlation: does exposure predict projected employment decline?
- AIOE quartile × projected growth rate comparison table
- Subgroup: does the relationship differ by wage tertile?
- JOLTS job openings trends: high vs. low exposure industries post-ChatGPT (Nov 2022)

### VI. Validation 2 — Corporate Layoff Case Studies
- Methodology: mapping announced layoff roles to SOC codes via O\*NET
- **Tier 1 flagship cases** (deep treatment, one paragraph each):
  - Google/Alphabet 2023–2024 (~12,000 — recruiting, program management)
  - Microsoft 2023 (~10,000 — customer support, sales operations)
  - IBM 2023–2024 (~7,800 — HR, back-office, finance ops; CEO explicitly cited AI)
- **Tier 2 supporting cases** (summary table):
  - Goldman Sachs — junior analyst and research roles
  - Salesforce — sales operations, customer success
  - Duolingo 2024 — content creation contractors
  - BT Group — customer service, network operations (55,000 by 2030)
- **Aggregate finding:** % of displaced roles falling in top AIOE quartile
- Discussion: the model predicted these profiles from structural first principles

### VII. Stage 2 — Scenario Projections
- Scenario definitions (conservative / moderate / aggressive AI capability)
- Projection methodology using Stage 1 coefficients
- Results: how exposure distribution shifts under each scenario
- Occupational clusters crossing from moderate → high exposure
- At-risk population estimates by wage and education group

### VIII. Stage 3 — Agent-Based Employment Dynamics
- Model architecture: workers and firms as agents
- Displacement, reinstatement, and reskilling parameters
- Swarm optimization for equilibrium employment distributions
- International policy comparison:
  - Labor market flexibility (OECD indices)
  - UI generosity and active labor market spending
  - National AI strategy adoption
- Results: employment outcomes under each scenario × policy environment

### IX. Discussion
- The hollowing-out pattern: evidence and limits
- Optimist vs. pessimist framing — where does this paper land?
- Implications for workers in the blast radius
- Policy levers most effective at moderating disruption

### X. Limitations & Future Research
- Exposure ≠ displacement: the causal gap
- Projection assumptions and sensitivity
- ABM initialization and parameter uncertainty
- Data constraints (AIOE vintage, O\*NET task classification subjectivity)
- Case study role-to-SOC mapping involves judgment calls

### XI. Conclusion

### XII. References

---

## Scripts & Pipeline

| Script | Paper Section | Status | What it does |
|---|---|---|---|
| `collect_data.py` | III | ✅ Built | Downloads AIOE, O\*NET, BLS wages → `ai_exposure_stage1.csv` |
| `stage1_regression.py` | IV | ✅ Built | 4 OLS models, diagnostics, all tables and figures |
| `validation.py` | V | 🔜 Next | BLS projections × AIOE correlation, JOLTS trends |
| `case_studies.py` | VI | 🔜 Next | Layoff role → SOC → AIOE mapping, case tables and plot |
| `stage2_scenarios.py` | VII | 📋 Planned | Scenario projections using Stage 1 coefficients |
| `stage3_abm.py` | VIII | 📋 Planned | Agent-based employment dynamics + policy comparison |

---

## Project File Structure

```
ai_exposure_project/
│
├── README.md                        ← you are here
│
├── collect_data.py                  ← Stage 0: data pipeline
├── stage1_regression.py             ← Stage 1: OLS regression
├── validation.py                    ← Validation 1: BLS projections + JOLTS
├── case_studies.py                  ← Validation 2: corporate layoff cases
├── stage2_scenarios.py              ← Stage 2: scenario projections
├── stage3_abm.py                    ← Stage 3: agent-based model
│
├── data_cache/                      ← raw downloaded files (gitignore this)
│   ├── AIOE_DataAppendix.xlsx
│   ├── onet_work_activities.*
│   ├── onet_education.*
│   ├── bls_oes_national.xlsx
│   ├── bls_oep_projections.xlsx     ← BLS 10-year employment projections
│   └── jolts_industry.xlsx          ← BLS JOLTS job openings by industry
│
├── ai_exposure_stage1.csv           ← merged regression dataset
├── ai_exposure_validated.csv        ← stage1 + BLS projections merged
│
└── outputs/
    ├── tables/
    │   ├── summary_stats.csv
    │   ├── correlation_matrix.csv
    │   ├── regression_table.csv
    │   ├── validation_correlations.csv   ← AIOE × projected growth
    │   ├── aioe_quartile_growth.csv      ← quartile × growth comparison
    │   └── case_studies_table.csv        ← layoff cases with AIOE scores
    └── figures/
        ├── aioe_distribution.png
        ├── scatter_matrix.png
        ├── residual_plots.png
        ├── coef_plot.png
        ├── top_bottom_occupations.png
        ├── aioe_vs_projected_growth.png  ← validation scatter
        ├── quartile_growth_bars.png      ← quartile comparison
        ├── jolts_trends.png              ← job openings by exposure group
        └── case_studies_aioe.png         ← layoffs vs AIOE distribution
```

---

## Data Sources

| Source | Variables | Used In | Access |
|---|---|---|---|
| Felten, Raj & Seamans (2021) | `aioe_score` | Stage 1 DV | [GitHub](https://github.com/AIOE-Data/AIOE) — free |
| O\*NET Work Activities (v29) | `routine_cognitive`, `social_interpersonal`, `physical_manual`, `creative_adaptive` | Stage 1 regressors | [onetcenter.org](https://www.onetcenter.org/database.html) — free |
| O\*NET Education & Training (v29) | `education_level` | Stage 1 regressor | [onetcenter.org](https://www.onetcenter.org/database.html) — free |
| BLS OES National (2024) | `annual_median_wage`, `log_wage`, `wage_percentile` | Stage 1 regressor | [bls.gov](https://www.bls.gov/oes/) — free |
| BLS Occupational Employment Projections | `projected_growth_rate`, `projected_openings` | Validation 1 | [bls.gov/emp](https://www.bls.gov/emp/) — free |
| BLS JOLTS | `job_openings_by_industry` | Validation 1 | [bls.gov/jlt](https://www.bls.gov/jlt/) — free |
| Layoffs.fyi + public earnings calls | Layoff counts, role descriptions | Validation 2 | layoffs.fyi — free |

All occupation-level sources merged on **6-digit SOC codes**.  
Industry-level sources (JOLTS) merged on **NAICS industry codes**.

---

## Stage 1 Model Specifications

```
Model 1 (Baseline):    AIOE = β₀ + β₁(RoutineCog) + β₂(Social) + β₃(Physical) + β₄(Creative) + ε
Model 2 (+ Education): Model 1 + β₅(EducationLevel)
Model 3 (Full Model):  Model 2 + β₆(LogWage)          ← primary specification
Model 4 (Robustness):  Model 2 + β₆(WagePercentile)   ← sensitivity check
```

**Standard errors:** HC3 heteroskedasticity-robust throughout  
**Multicollinearity:** VIF flagged if > 10

### Expected Signs

| Variable | Expected β | Rationale |
|---|---|---|
| Routine Cognitive | **+** | Core AI target: data processing, information tasks |
| Social/Interpersonal | **−** | AI performs poorly on relational, emotional tasks |
| Physical/Manual | **−** | Moravec's Paradox — physical dexterity is hard for AI |
| Creative/Adaptive | **?** | Ambiguous — pre/post-LLM indices disagree |
| Education Level | **+** | Eloundou et al.: higher-ed jobs more LLM-exposed |
| Log Wage | **+** | Higher-wage cognitive occupations tend to be more exposed |

---

## Validation 2 — Case Study Reference Table

| Company | Year | Scale | Roles Cut | AI Attribution | Expected AIOE |
|---|---|---|---|---|---|
| Google/Alphabet | 2023–24 | ~12,000 | Recruiting, program management | Explicit — earnings calls | High |
| Microsoft | 2023 | ~10,000 | Customer support, sales ops | Explicit — Copilot launch | High |
| IBM | 2023–24 | ~7,800 | HR, back-office, finance ops | Explicit — CEO statement | High |
| Goldman Sachs | 2023–24 | Undisclosed | Junior analysts, research | Implicit — AI report generation | High |
| Salesforce | 2023 | ~8,000 | Sales ops, customer success | Explicit — Einstein AI | Medium-High |
| Duolingo | 2024 | ~10% contractors | Content creators | Explicit — press release | Medium |
| BT Group | 2023 | ~55,000 by 2030 | Customer service, network ops | Explicit — CEO statement | High |

---

## Execution Order

```bash
# Install all dependencies
pip install requests pandas openpyxl xlrd tqdm numpy statsmodels matplotlib seaborn scipy mesa

# 1. Collect and merge all data
python collect_data.py

# 2. Stage 1: OLS regression
python stage1_regression.py

# 3. Validation 1: BLS employment projections + JOLTS
python validation.py

# 4. Validation 2: corporate layoff case studies
python case_studies.py

# 5. Stage 2: scenario projections
python stage2_scenarios.py

# 6. Stage 3: agent-based employment dynamics
python stage3_abm.py
```

---

## Roadmap

**Complete ✅**
- [x] Literature review written
- [x] `collect_data.py` — full data pipeline
- [x] `stage1_regression.py` — OLS engine with diagnostics, tables, figures

**In Progress 🔬**
- [ ] Run Stage 1 on collected data, review results
- [ ] Interpret coefficients, write Section IV

**Up Next 🔜**
- [ ] `validation.py` — BLS projections × AIOE, JOLTS trends
- [ ] `case_studies.py` — corporate layoff mapping + AIOE scoring

**Planned 📋**
- [ ] `stage2_scenarios.py` — conservative/moderate/aggressive projections
- [ ] `stage3_abm.py` — agent-based model + policy comparison
- [ ] Final paper draft
- [ ] Presentation slides

---

## Citation

```
Myers, D. (2026). AI Job Exposure in the U.S. Economy: A Speculative Regression
Framework with Validation, Case Studies, and Agent-Based Scenario Modeling.
Undergraduate Research Paper.
```

**Primary data citations:**
- Felten, E., Raj, M., & Seamans, R. (2021). *Strategic Management Journal*, 42(12), 2195–2217.
- Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). *Science*, 384(6702), 1306–1308.
- Acemoglu, D., & Restrepo, P. (2020). *Journal of Political Economy*, 128(6), 2188–2244.

---

*Built with Python · Econometrics via statsmodels · Simulation via Mesa · Visualization via matplotlib/seaborn*
