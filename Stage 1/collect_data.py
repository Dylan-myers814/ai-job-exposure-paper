"""
=============================================================================
AI Job Exposure Analysis — Stage 1 Data Collection
=============================================================================
Dylan Myers | Undergraduate Research Paper

This script downloads and merges four data sources into a single clean CSV
ready for OLS regression. All sources are free and publicly available.

SOURCES:
  1. Felten, Raj & Seamans (2021) AIOE scores       [GitHub]
  2. O*NET Work Activities (task content)            [onetcenter.org]
  3. O*NET Education/Training requirements           [onetcenter.org]
  4. BLS OES National wage data                      [bls.gov]

OUTPUT:
  ai_exposure_stage1.csv  — one row per 6-digit SOC occupation

USAGE:
  pip install requests pandas openpyxl xlrd tqdm
  python collect_data.py

If any download fails, the script will print the manual download URL
and instructions so you can drop the file in manually.
=============================================================================
"""

import os
import io
import time
import zipfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────────

OUTPUT_FILE = "ai_exposure_stage1.csv"
CACHE_DIR   = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def download(url, filename, description):
    """Download a file with caching. Returns local path or None on failure."""
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        print(f"  [cached] {description}")
        return path
    print(f"  Downloading {description}...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"  ✓ {description} saved to {path}")
        return path
    except Exception as e:
        print(f"\n  ✗ FAILED: {description}")
        print(f"    Error: {e}")
        print(f"    Manual download URL: {url}")
        print(f"    Save file as: {path}\n")
        return None


def clean_soc(soc):
    """Standardize SOC codes to XX-XXXX format."""
    soc = str(soc).strip()
    soc = soc.replace(".", "-")
    if len(soc) == 7 and soc[2] != "-":
        soc = soc[:2] + "-" + soc[2:]
    return soc


# ── SOURCE 1: FELTEN AIOE SCORES ─────────────────────────────────────────────

def load_aioe():
    """
    Felten, Raj & Seamans (2021) AI Occupational Exposure scores.
    GitHub: https://github.com/AIOE-Data/AIOE

    MANUAL FALLBACK:
      Go to https://github.com/AIOE-Data/AIOE
      Download AIOE_DataAppendix.xlsx
      Place in data_cache/AIOE_DataAppendix.xlsx
    """
    print("\n[1/4] Loading AIOE scores (Felten et al. 2021)...")

    url  = "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx"
    path = download(url, "AIOE_DataAppendix.xlsx", "Felten AIOE data")

    if path is None:
        # Try the releases page as fallback
        url2 = "https://github.com/AIOE-Data/AIOE/raw/main/AIOE_DataAppendix.xlsx"
        path = download(url2, "AIOE_DataAppendix.xlsx", "Felten AIOE data (fallback)")

    if path is None:
        raise FileNotFoundError(
            "Could not download AIOE data. Please manually download from:\n"
            "  https://github.com/AIOE-Data/AIOE\n"
            "  and save as data_cache/AIOE_DataAppendix.xlsx"
        )

    # The file has multiple sheets — we want Appendix A (AIOE by occupation)
    xls = pd.ExcelFile(path)
    print(f"  Sheets found: {xls.sheet_names}")

    # Try to find the occupation-level sheet
    target_sheet = None
    for sheet in xls.sheet_names:
        if "append" in sheet.lower() or "occup" in sheet.lower() or "a" == sheet.strip().lower():
            target_sheet = sheet
            break
    if target_sheet is None:
        target_sheet = xls.sheet_names[0]

    df = pd.read_excel(path, sheet_name=target_sheet)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Identify SOC and AIOE columns flexibly
    soc_col = next(
        (c for c in df.columns if "soc" in c.lower() or "occ" in c.lower() or "code" in c.lower()),
        df.columns[0]
    )
    aioe_col = next(
        (c for c in df.columns if "aioe" in c.lower() or "exposure" in c.lower() or "score" in c.lower()),
        df.columns[1]
    )
    name_col = next(
        (c for c in df.columns if "title" in c.lower() or "name" in c.lower() or "occup" in c.lower()
         and c != soc_col),
        None
    )

    cols = {soc_col: "soc_code", aioe_col: "aioe_score"}
    if name_col:
        cols[name_col] = "occupation_title"

    df = df[list(cols.keys())].rename(columns=cols)
    df["soc_code"] = df["soc_code"].apply(clean_soc)
    df["aioe_score"] = pd.to_numeric(df["aioe_score"], errors="coerce")
    df = df.dropna(subset=["soc_code", "aioe_score"])

    print(f"  ✓ AIOE: {len(df)} occupations loaded")
    return df


# ── SOURCE 2: O*NET WORK ACTIVITIES ──────────────────────────────────────────

def load_onet_work_activities():
    """
    O*NET Work Activities — the backbone for constructing task content indices.
    We build four composite variables from the O*NET scale scores:
      - routine_cognitive    (information processing, routine analytical tasks)
      - social_interpersonal (communicating, coordinating, caring for others)
      - physical_manual      (handling equipment, physical work, spatial tasks)
      - creative_adaptive    (creative thinking, problem solving, adaptability)

    Download: https://www.onetcenter.org/database.html#all-files
    File: Work Activities.xlsx (or .txt in the flat file database)

    MANUAL FALLBACK:
      1. Go to https://www.onetcenter.org/database.html
      2. Download the "Work Activities" file (Excel or text)
      3. Save as data_cache/onet_work_activities.xlsx
    """
    print("\n[2/4] Loading O*NET Work Activities...")

    # Try the O*NET flat file database ZIP
    url  = "https://www.onetcenter.org/dl_files/database/db_29_0_excel/Work%20Activities.xlsx"
    path = download(url, "onet_work_activities.xlsx", "O*NET Work Activities")

    if path is None:
        # Try txt version
        url2 = "https://www.onetcenter.org/dl_files/database/db_29_0_text/Work%20Activities.txt"
        path2 = download(url2, "onet_work_activities.txt", "O*NET Work Activities (txt)")
        if path2:
            df_raw = pd.read_csv(path2, sep="\t", encoding="latin-1")
        else:
            raise FileNotFoundError(
                "Could not download O*NET Work Activities.\n"
                "Manual download:\n"
                "  1. Go to https://www.onetcenter.org/database.html\n"
                "  2. Download 'Work Activities' (Excel or txt)\n"
                "  3. Save as data_cache/onet_work_activities.xlsx or .txt"
            )
    else:
        df_raw = pd.read_excel(path)

    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Columns: {list(df_raw.columns[:6])}...")

    # Standardize column names
    df_raw.columns = [c.lower().replace(" ", "_").replace("/", "_") for c in df_raw.columns]

    soc_col   = next(c for c in df_raw.columns if "o*net" in c or "soc" in c or "onetsoc" in c.replace("-",""))
    elem_col  = next(c for c in df_raw.columns if "element" in c and "name" in c)
    scale_col = next(c for c in df_raw.columns if "scale" in c and "id" in c.lower())
    value_col = next(c for c in df_raw.columns if "data_value" in c or "value" in c.lower())

    df_raw = df_raw.rename(columns={
        soc_col: "soc_code", elem_col: "element", scale_col: "scale_id", value_col: "value"
    })

    # Keep only "Importance" scale (IM), which is most interpretable
    df_raw = df_raw[df_raw["scale_id"] == "IM"].copy()
    df_raw["soc_code"] = df_raw["soc_code"].apply(clean_soc)
    df_raw["value"]    = pd.to_numeric(df_raw["value"], errors="coerce")

    # ── Task content classification ──────────────────────────────────────────
    # Map O*NET work activity elements to four composite dimensions
    # Based on Acemoglu-Autor (2011) task taxonomy + AI exposure literature

    routine_cognitive_keywords = [
        "processing information", "analyzing data", "documenting", "recording",
        "scheduling", "data or information", "quantitative", "evaluating information",
        "organizing", "clerical", "computing"
    ]
    social_interpersonal_keywords = [
        "communicating", "coordinating", "assisting", "caring for", "coaching",
        "resolving conflicts", "selling", "persuading", "negotiating", "customer",
        "supervising", "training", "interpersonal", "helping"
    ]
    physical_manual_keywords = [
        "handling", "operating", "repairing", "installing", "maintaining",
        "controlling machines", "physical", "manual", "inspecting", "driving",
        "construction", "manufacturing"
    ]
    creative_adaptive_keywords = [
        "thinking creatively", "developing", "innovating", "problem solving",
        "strategizing", "identifying", "researching", "judging", "making decisions",
        "planning", "designing", "adapting"
    ]

    def classify_activity(element_name):
        e = element_name.lower()
        scores = {
            "routine_cognitive":    sum(k in e for k in routine_cognitive_keywords),
            "social_interpersonal": sum(k in e for k in social_interpersonal_keywords),
            "physical_manual":      sum(k in e for k in physical_manual_keywords),
            "creative_adaptive":    sum(k in e for k in creative_adaptive_keywords),
        }
        return scores

    # Expand element classifications
    elements = df_raw["element"].unique()
    el_map = pd.DataFrame([
        {"element": e, **classify_activity(e)} for e in elements
    ])

    df_raw = df_raw.merge(el_map, on="element", how="left")

    # Weight importance scores by classification membership, then aggregate by SOC
    for dim in ["routine_cognitive", "social_interpersonal", "physical_manual", "creative_adaptive"]:
        df_raw[f"weighted_{dim}"] = df_raw["value"] * df_raw[dim]

    agg = df_raw.groupby("soc_code").agg(
        total_importance=("value", "sum"),
        routine_cognitive=("weighted_routine_cognitive", "sum"),
        social_interpersonal=("weighted_social_interpersonal", "sum"),
        physical_manual=("weighted_physical_manual", "sum"),
        creative_adaptive=("weighted_creative_adaptive", "sum"),
        n_activities=("value", "count")
    ).reset_index()

    # Normalize by total importance to get shares (0–1)
    for dim in ["routine_cognitive", "social_interpersonal", "physical_manual", "creative_adaptive"]:
        agg[dim] = agg[dim] / agg["total_importance"].replace(0, np.nan)

    agg = agg.drop(columns=["total_importance"])
    print(f"  ✓ O*NET Work Activities: {len(agg)} occupations, 4 task dimensions built")
    return agg


# ── SOURCE 3: O*NET EDUCATION & TRAINING ─────────────────────────────────────

def load_onet_education():
    """
    O*NET Education, Training, and Experience — required education level.
    We extract the 'Required Level of Education' element for each occupation
    and construct a numeric education index (1=no HS, 6=doctoral degree).

    MANUAL FALLBACK:
      1. Go to https://www.onetcenter.org/database.html
      2. Download 'Education, Training, and Experience' file
      3. Save as data_cache/onet_education.xlsx or .txt
    """
    print("\n[3/4] Loading O*NET Education requirements...")

    url  = "https://www.onetcenter.org/dl_files/database/db_29_0_excel/Education%2C%20Training%2C%20and%20Experience.xlsx"
    path = download(url, "onet_education.xlsx", "O*NET Education/Training")

    if path is None:
        url2 = "https://www.onetcenter.org/dl_files/database/db_29_0_text/Education%2C%20Training%2C%20and%20Experience.txt"
        path2 = download(url2, "onet_education.txt", "O*NET Education/Training (txt)")
        if path2:
            df_raw = pd.read_csv(path2, sep="\t", encoding="latin-1")
        else:
            raise FileNotFoundError(
                "Could not download O*NET Education data.\n"
                "Manual download:\n"
                "  1. Go to https://www.onetcenter.org/database.html\n"
                "  2. Download 'Education, Training, and Experience'\n"
                "  3. Save as data_cache/onet_education.xlsx or .txt"
            )
    else:
        df_raw = pd.read_excel(path)

    df_raw.columns = [c.lower().replace(" ", "_").replace("/", "_") for c in df_raw.columns]

    soc_col   = next(c for c in df_raw.columns if "o*net" in c or "soc" in c or "onetsoc" in c.replace("-",""))
    elem_col  = next(c for c in df_raw.columns if "element" in c and "name" in c)
    scale_col = next(c for c in df_raw.columns if "scale" in c and "id" in c.lower())
    value_col = next(c for c in df_raw.columns if "data_value" in c or "value" in c.lower())

    df_raw = df_raw.rename(columns={
        soc_col: "soc_code", elem_col: "element",
        scale_col: "scale_id", value_col: "value"
    })

    # Filter to: element = "Required Level of Education", scale = RL (required level)
    edu = df_raw[
        (df_raw["element"].str.contains("Required Level of Education", case=False, na=False)) &
        (df_raw["scale_id"] == "RL")
    ].copy()

    edu["soc_code"] = edu["soc_code"].apply(clean_soc)
    edu["value"]    = pd.to_numeric(edu["value"], errors="coerce")

    # O*NET RL scale: 1=No HS, 2=HS/GED, 3=Post-sec cert, 4=Some college,
    #                 5=Associate, 6=Bachelor, 7=Post-BA cert, 8=Master, 9=Doctoral/Professional
    edu_agg = edu.groupby("soc_code")["value"].mean().reset_index()
    edu_agg.columns = ["soc_code", "education_level"]

    print(f"  ✓ O*NET Education: {len(edu_agg)} occupations")
    return edu_agg


# ── SOURCE 4: BLS OES WAGE DATA ───────────────────────────────────────────────

def load_bls_wages():
    """
    BLS Occupational Employment and Wage Statistics — annual wages by SOC.
    We use the national all-industries file to get median annual wage
    and compute a wage percentile rank across occupations.

    URL: https://www.bls.gov/oes/current/oes_nat.xlsx
    Backup: https://www.bls.gov/oes/current/national_M2023_dl.xlsx

    MANUAL FALLBACK:
      1. Go to https://www.bls.gov/oes/current/oes_nat.htm
      2. Download 'National' Excel file (oes_nat.xlsx)
      3. Save as data_cache/bls_oes_national.xlsx
    """
    print("\n[4/4] Loading BLS OES wage data...")

    urls = [
        "https://www.bls.gov/oes/current/oes_nat.xlsx",
        "https://www.bls.gov/oes/special.requests/oesnat24.xlsx",
    ]

    path = None
    for url in urls:
        path = download(url, "bls_oes_national.xlsx", f"BLS OES wages ({url.split('/')[-1]})")
        if path:
            break

    if path is None:
        raise FileNotFoundError(
            "Could not download BLS OES data.\n"
            "Manual download:\n"
            "  1. Go to https://www.bls.gov/oes/current/oes_nat.htm\n"
            "  2. Click 'Download' under National XLS files\n"
            "  3. Save as data_cache/bls_oes_national.xlsx"
        )

    df_raw = pd.read_excel(path, dtype={"OCC_CODE": str})
    df_raw.columns = [c.lower().strip() for c in df_raw.columns]

    print(f"  Columns: {list(df_raw.columns[:8])}...")

    soc_col  = next(c for c in df_raw.columns if "occ_code" in c or "soc" in c, None) or "occ_code"
    name_col = next((c for c in df_raw.columns if "occ_title" in c or "occ_nm" in c), None)
    wage_col = next((c for c in df_raw.columns if "a_median" in c or "ann_mean" in c or "annual" in c), None)

    if wage_col is None:
        print(f"  Available columns: {list(df_raw.columns)}")
        raise ValueError("Could not identify wage column in BLS data")

    df = df_raw[[soc_col, name_col, wage_col]].copy() if name_col else df_raw[[soc_col, wage_col]].copy()
    df.columns = ["soc_code", "occupation_title", wage_col] if name_col else ["soc_code", wage_col]
    df = df.rename(columns={wage_col: "annual_median_wage"})

    df["soc_code"] = df["soc_code"].apply(clean_soc)

    # BLS uses '*' and '#' for suppressed/unavailable values — replace with NaN
    df["annual_median_wage"] = pd.to_numeric(
        df["annual_median_wage"].astype(str).str.replace(",", "").str.strip(),
        errors="coerce"
    )

    # Filter out aggregate groups (SOC codes ending in 0000 are major groups)
    df = df[~df["soc_code"].str.endswith("0000")].copy()
    df = df.dropna(subset=["annual_median_wage"])

    # Compute wage percentile rank
    df["wage_percentile"] = df["annual_median_wage"].rank(pct=True) * 100

    print(f"  ✓ BLS OES: {len(df)} detailed occupations loaded")
    return df


# ── MERGE ALL SOURCES ─────────────────────────────────────────────────────────

def build_dataset():
    print("=" * 65)
    print("  AI Job Exposure — Stage 1 Data Collection")
    print("=" * 65)

    # Load all four sources
    df_aioe = load_aioe()
    df_wa   = load_onet_work_activities()
    df_edu  = load_onet_education()
    df_bls  = load_bls_wages()

    # ── Merge ────────────────────────────────────────────────────────────────
    print("\n[Merging] Joining all sources on SOC code...")

    # Start with AIOE as the anchor (it defines the universe of occupations)
    merged = df_aioe.copy()

    n0 = len(merged)
    merged = merged.merge(df_wa,  on="soc_code", how="left")
    print(f"  After O*NET Work Activities merge: {len(merged)} rows "
          f"({merged['routine_cognitive'].notna().sum()} matched)")

    merged = merged.merge(df_edu, on="soc_code", how="left")
    print(f"  After O*NET Education merge: {len(merged)} rows "
          f"({merged['education_level'].notna().sum()} matched)")

    merged = merged.merge(
        df_bls[["soc_code", "annual_median_wage", "wage_percentile"]],
        on="soc_code", how="left"
    )
    print(f"  After BLS OES merge: {len(merged)} rows "
          f"({merged['annual_median_wage'].notna().sum()} matched)")

    # ── Derived variables ────────────────────────────────────────────────────
    print("\n[Derived Variables] Computing additional regressors...")

    # Log wage (standard in labor economics)
    merged["log_wage"] = np.log(merged["annual_median_wage"].replace(0, np.nan))

    # Standardize AIOE score to z-score (useful as dependent variable)
    merged["aioe_zscore"] = (
        (merged["aioe_score"] - merged["aioe_score"].mean()) /
        merged["aioe_score"].std()
    )

    # Education dummy: college_required = 1 if education_level >= 6 (Bachelor's+)
    merged["college_required"] = (merged["education_level"] >= 6).astype(int)

    # ── Final cleaning ───────────────────────────────────────────────────────
    complete_cases = merged.dropna(subset=[
        "aioe_score", "routine_cognitive", "social_interpersonal",
        "physical_manual", "creative_adaptive", "education_level",
        "annual_median_wage"
    ])

    print(f"\n  Total rows: {len(merged)}")
    print(f"  Complete cases (all key vars): {len(complete_cases)}")
    print(f"  Missing AIOE: {merged['aioe_score'].isna().sum()}")
    print(f"  Missing task dims: {merged['routine_cognitive'].isna().sum()}")
    print(f"  Missing education: {merged['education_level'].isna().sum()}")
    print(f"  Missing wage: {merged['annual_median_wage'].isna().sum()}")

    # ── Export ───────────────────────────────────────────────────────────────
    col_order = [
        "soc_code", "occupation_title",
        # Dependent variable
        "aioe_score", "aioe_zscore",
        # Task content regressors (O*NET)
        "routine_cognitive", "social_interpersonal", "physical_manual", "creative_adaptive",
        "n_activities",
        # Education
        "education_level", "college_required",
        # Wage
        "annual_median_wage", "log_wage", "wage_percentile"
    ]
    col_order = [c for c in col_order if c in merged.columns]
    merged = merged[col_order]

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'=' * 65}")
    print(f"  ✓ Dataset saved: {OUTPUT_FILE}")
    print(f"  Rows: {len(merged)} occupations")
    print(f"  Columns: {len(merged.columns)}")
    print(f"{'=' * 65}")

    # ── Summary stats preview ────────────────────────────────────────────────
    print("\n── Summary Statistics Preview ──────────────────────────────")
    print(merged[[
        "aioe_score", "routine_cognitive", "social_interpersonal",
        "physical_manual", "creative_adaptive", "education_level",
        "annual_median_wage"
    ]].describe().round(3).to_string())

    print("\n── Top 10 Most AI-Exposed Occupations ──────────────────────")
    top10 = merged.nlargest(10, "aioe_score")[["soc_code", "occupation_title", "aioe_score"]]
    print(top10.to_string(index=False))

    print("\n── Bottom 10 Least AI-Exposed Occupations ──────────────────")
    bot10 = merged.nsmallest(10, "aioe_score")[["soc_code", "occupation_title", "aioe_score"]]
    print(bot10.to_string(index=False))

    return merged


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_dataset()
    print(f"\nNext step: Run stage1_regression.py on {OUTPUT_FILE}")
