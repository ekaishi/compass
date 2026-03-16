"""
data_fetcher.py — NYC K12 school data pipeline for the Gale Sales Compass.

Confirmed-working data sources (verified March 2026):
  1. NYC Open Data p6h4-mpyy  — School Locations (DBN, name, borough, grades, lat/lon)
  2. NYC Open Data vmmu-wj3w  — Demographic Snapshot 2020-21
                               (enrollment, ELL count/%, poverty %, Economic Need Index)
  3. NYC Open Data iebs-5yhr  — ELA Test Results 2013-2023
                               (% at Level 3+4 = proficient, school-level, 2023)
  4. NYSED cn.nysed.gov       — CEP Notification Report (Excel, "NYC DOE" sheet)
                               School column = DBN, direct match
  5. NYSED nysed.gov          — Title I Part A final allocations (HTML table, 2025-26)
                               Borough-level county totals distributed proportionally
                               by each school's poverty headcount
  6. Synthetic demo seed      — if ALL network fetches fail, shows realistic UI

All data cached in schools.db (SQLite, same directory as this file).
"""

import os
import re
import json
import sqlite3
import logging
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schools.db")
SODA_BASE = "https://data.cityofnewyork.us/resource"

BOROUGH_MAP = {
    "M": "Manhattan", "X": "Bronx", "K": "Brooklyn",
    "Q": "Queens",    "R": "Staten Island",
    "MANHATTAN": "Manhattan", "BRONX": "Bronx", "BROOKLYN": "Brooklyn",
    "QUEENS": "Queens", "STATEN ISLAND": "Staten Island",
}

# CEP Notification Report — NYSED Child Nutrition portal (confirmed working Apr 2025)
CEP_URL = "https://www.cn.nysed.gov/sites/cn/files/cepnotificationreport.xlsx"

# Title I Part A final allocations — NYSED (HTML table, county-level for NYC)
TITLE1_URL = "https://www.nysed.gov/essa/2025-26-final-allocations-title-i-part"

# NYC borough → county BEDS code prefix in the NYSED Title I table
# (first 4 digits of BEDS code identify county+district for county-wide entries)
BOROUGH_BEDS = {
    "Manhattan":     ["3000"],            # New York County (BEDS 300000010000)
    "Bronx":         ["3207"],           # Bronx County
    "Brooklyn":      ["3313"],           # Kings County
    "Queens":        ["3424"],           # Queens County
    "Staten Island": ["3531"],           # Richmond County
}

# ─── SQLite helpers ───────────────────────────────────────────────────────────

def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS schools (
                dbn TEXT PRIMARY KEY, school_name TEXT, borough TEXT,
                address TEXT, grade_band TEXT, lat REAL, lon REAL,
                total_enrollment INTEGER, ell_count INTEGER, ell_pct REAL,
                poverty_pct REAL, economic_need_index REAL,
                ela_proficiency REAL, cep INTEGER,
                title1_amount REAL, title3_amount REAL,
                priority_score REAL, tier TEXT
            )""")
        c.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")


def save_schools(df: pd.DataFrame):
    init_db()
    with _conn() as c:
        df.to_sql("schools", c, if_exists="replace", index=False)
        c.execute("INSERT OR REPLACE INTO meta VALUES ('last_updated', ?)",
                  (datetime.now().isoformat(),))


def load_schools():
    init_db()
    try:
        with _conn() as c:
            df = pd.read_sql("SELECT * FROM schools ORDER BY priority_score DESC", c)
            meta = pd.read_sql("SELECT value FROM meta WHERE key='last_updated'", c)
        last_updated = meta["value"].iloc[0] if not meta.empty else None
        return df, last_updated
    except Exception:
        return pd.DataFrame(), None


# ─── Generic Socrata fetch ────────────────────────────────────────────────────

def _soda(dataset_id: str, limit: int = 60000, **params) -> pd.DataFrame:
    url = f"{SODA_BASE}/{dataset_id}.json"
    p = {"$limit": limit, **params}
    try:
        r = requests.get(url, params=p, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return pd.DataFrame(data)
    except Exception as e:
        logger.warning(f"  Socrata [{dataset_id}] failed: {e}")
    return pd.DataFrame()


# ─── School directory ─────────────────────────────────────────────────────────

def fetch_school_directory() -> pd.DataFrame:
    """
    Dataset p6h4-mpyy — 2017-18 School Locations
    Fields used: ats_system_code (DBN), location_name, location_1 (lat/lon/address),
                 grades_final_text, location_category_description (borough implied).
    """
    df = _soda("p6h4-mpyy", limit=5000)
    if df.empty:
        logger.error("  Cannot fetch school directory")
        return pd.DataFrame()

    logger.info(f"  Raw rows: {len(df)}")

    # Extract DBN
    df["dbn"] = df["ats_system_code"].astype(str).str.strip()
    df = df[df["dbn"].str.len() >= 6].copy()

    # Extract school name
    df["school_name"] = df.get("location_name", df.get("school_name", "")).astype(str).str.strip().str.title()

    # Extract lat/lon from nested location_1 dict
    def _extract_coord(loc, key):
        if isinstance(loc, dict):
            return loc.get(key)
        if isinstance(loc, str):
            try:
                d = json.loads(loc)
                return d.get(key)
            except Exception:
                pass
        return None

    if "location_1" in df.columns:
        df["lat"] = df["location_1"].apply(lambda x: _extract_coord(x, "latitude"))
        df["lon"] = df["location_1"].apply(lambda x: _extract_coord(x, "longitude"))
        # Extract address from human_address field inside location_1
        def _extract_addr(loc):
            if isinstance(loc, dict):
                ha = loc.get("human_address", "")
                if ha:
                    try:
                        a = json.loads(ha)
                        return f"{a.get('address','')}, {a.get('city','')}"
                    except Exception:
                        pass
            return ""
        df["address"] = df["location_1"].apply(_extract_addr)
    else:
        df["lat"] = None
        df["lon"] = None
        df["address"] = ""

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Borough from DBN prefix (2-char district code → borough letter = 3rd char)
    # DBN format: 09X123 → district=09, borough=X, school=123
    def _boro(dbn):
        dbn = str(dbn).strip()
        if len(dbn) >= 3:
            code = dbn[2].upper()
            return BOROUGH_MAP.get(code, "Unknown")
        return "Unknown"
    df["borough"] = df["dbn"].apply(_boro)

    # Grade band
    grade_col = "grades_final_text" if "grades_final_text" in df.columns else "grades_text"
    if grade_col in df.columns:
        df["grade_band"] = df[grade_col].apply(_grade_band)
    else:
        df["grade_band"] = "Unknown"

    keep = ["dbn", "school_name", "borough", "address", "grade_band", "lat", "lon"]
    return df[keep].drop_duplicates("dbn").reset_index(drop=True)


def _grade_band(raw) -> str:
    if not raw or pd.isna(raw):
        return "Unknown"
    g = str(raw).upper()
    nums = [int(x) for x in re.findall(r"\d+", g) if int(x) <= 12]
    has_pk = "PK" in g or "PRE" in g
    has_k  = bool(re.search(r"(?<!\d)K(?!\d)|,0K|^K,|^K$", g))
    lo = min(nums) if nums else (0 if (has_k or has_pk) else 99)
    hi = max(nums) if nums else 0
    if hi >= 9:
        if lo <= 5 or has_k or has_pk: return "K-12"
        if lo <= 8:                     return "Middle-High"
        return "High School"
    if hi >= 6:
        if lo <= 5 or has_k or has_pk: return "K-8"
        return "Middle School"
    return "Elementary"


# ─── Demographics ─────────────────────────────────────────────────────────────

def fetch_demographics() -> pd.DataFrame:
    """
    Dataset vmmu-wj3w — DOE Demographic Snapshot 2020-21.
    Confirmed columns: english_language_learners (count), english_language_learners_1 (%),
                       poverty (count), poverty_1 (%), economic_need_index, total_enrollment.
    """
    # Try newest dataset first, then fall back
    for ds_id in ["vmmu-wj3w", "c7ru-d68s", "s52a-8aq6"]:
        df = _soda(ds_id, limit=60000)
        if not df.empty and "english_language_learners" in df.columns:
            logger.info(f"  Demographics from [{ds_id}]: {len(df):,} rows")
            break
        if not df.empty and len(df) > 1000:
            # Wrong dataset but has rows — try next
            logger.info(f"  [{ds_id}] lacks ELL columns, trying next…")

    if df.empty or "dbn" not in df.columns:
        logger.warning("  Demographics unavailable")
        return pd.DataFrame(columns=["dbn", "total_enrollment", "ell_count",
                                      "ell_pct", "poverty_pct", "economic_need_index"])

    # Keep most recent year per school
    if "year" in df.columns:
        df = df.sort_values("year", ascending=False)

    # Keep only "All Grades" summary rows
    # The snapshot has one row per school-grade combo; grade column varies by dataset
    for grade_col in ["grade", "grade_level"]:
        if grade_col in df.columns:
            all_g = df[df[grade_col].astype(str).str.lower().isin(
                ["all grades", "all", "total", "0", "00"])]
            if len(all_g) > 500:
                df = all_g
            break

    df = df.drop_duplicates("dbn", keep="first").copy()

    def _num(col, default=0):
        """Parse numeric, stripping trailing % signs (NYC Open Data quirk)."""
        if col not in df.columns:
            return pd.Series(default, index=df.index)
        raw = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
        return pd.to_numeric(raw, errors="coerce").fillna(default)

    out = pd.DataFrame()
    out["dbn"]              = df["dbn"].astype(str).str.strip()
    out["total_enrollment"] = _num("total_enrollment").astype(int)
    out["ell_count"]        = _num("english_language_learners").astype(int)

    raw_ell_pct = _num("english_language_learners_1")
    out["ell_pct"] = (raw_ell_pct * 100 if raw_ell_pct.max() <= 1.0 else raw_ell_pct).round(1)

    raw_pov = _num("poverty_1")
    out["poverty_pct"] = (raw_pov * 100 if raw_pov.max() <= 1.0 else raw_pov).round(1)

    raw_eni = _num("economic_need_index")
    # ENI comes as "88.2%" → 88.2 after % strip; sometimes as 0.882 → multiply
    out["economic_need_index"] = (raw_eni * 100 if raw_eni.max() <= 1.0 else raw_eni).round(1)

    return out


# ─── ELA proficiency ─────────────────────────────────────────────────────────

def fetch_ela_scores() -> pd.DataFrame:
    """
    Dataset iebs-5yhr — ELA Test Results 2013-2023.
    Query: school-level, All Grades, All Students, year=2023.
    geographic_subdivision = DBN, level_3_4_1 = % at/above proficiency.
    """
    df = _soda("iebs-5yhr", limit=5000,
               **{"$where": "year='2023' AND report_category='School' "
                             "AND grade='All Grades' AND category='All Students'",
                  "$select": "geographic_subdivision,school_name,level_3_4_1,number_tested"})

    if df.empty:
        logger.warning("  ELA data unavailable")
        return pd.DataFrame(columns=["dbn", "ela_proficiency"])

    logger.info(f"  ELA data: {len(df):,} school rows for 2023")

    out = pd.DataFrame()
    out["dbn"] = df["geographic_subdivision"].astype(str).str.strip()

    # 's' means suppressed (small N); treat as NaN
    ela_raw = df["level_3_4_1"].replace("s", np.nan)
    out["ela_proficiency"] = pd.to_numeric(ela_raw, errors="coerce").round(1)

    return out.dropna(subset=["ela_proficiency"]).drop_duplicates("dbn")


# ─── CEP status ───────────────────────────────────────────────────────────────

def fetch_cep_status(poverty_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Primary: NYSED CEP Notification Report (cn.nysed.gov), "NYC DOE" sheet.
    The `School` column in that sheet is the 6-char DBN (e.g. 01M015).
    Falls back to poverty_pct >= 40% proxy if the file is unreachable.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; gale-sales-tool/1.0)"}
    try:
        r = requests.get(CEP_URL, timeout=45, headers=headers)
        r.raise_for_status()
        xl = pd.ExcelFile(BytesIO(r.content))

        # "NYC DOE" sheet has School (DBN), School Name, SchoolType, ISP data
        sheet = "NYC DOE" if "NYC DOE" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet)
        logger.info(f"  CEP sheet '{sheet}': {len(df):,} rows, cols={list(df.columns[:5])}")

        # School column is the DBN
        dbn_col = next((c for c in df.columns
                        if c.strip().lower() in ["school", "dbn", "school code"]), None)
        if dbn_col:
            dbns = df[dbn_col].astype(str).str.strip()
            out = pd.DataFrame({"dbn": dbns, "cep": 1})
            logger.info(f"  ✓ {len(out):,} NYC CEP schools from NYSED")
            return out

    except Exception as e:
        logger.warning(f"  CEP fetch failed: {e}")

    # Fallback: poverty >= 40% proxy
    if poverty_df is not None and "poverty_pct" in poverty_df.columns:
        proxy = poverty_df[poverty_df["poverty_pct"] >= 40][["dbn"]].copy()
        proxy["cep"] = 1
        logger.info(f"  CEP proxy (poverty ≥ 40%): {len(proxy):,} schools estimated")
        return proxy

    logger.warning("  CEP unavailable")
    return pd.DataFrame()


# ─── Title I ─────────────────────────────────────────────────────────────────

def fetch_title_funding(schools_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    NYSED publishes Title I at county (borough) level for NYC DOE.
    Strategy:
      1. Scrape the NYSED HTML table → get borough-level county totals.
      2. For each school, estimate share = school_poverty_count / borough_poverty_total.
      3. title1_amount = share × borough_allocation.

    schools_df must have columns: dbn, borough, total_enrollment, poverty_pct.
    Returns DataFrame: dbn, title1_amount.
    """
    if schools_df is None or schools_df.empty:
        logger.warning("  Title I: no schools_df provided")
        return pd.DataFrame()

    headers = {"User-Agent": "Mozilla/5.0 (compatible; gale-sales-tool/1.0)"}
    try:
        r = requests.get(TITLE1_URL, timeout=30, headers=headers)
        r.raise_for_status()
        from io import StringIO
        tables = pd.read_html(StringIO(r.text))
        if not tables:
            raise ValueError("No HTML tables found")
        df = tables[0]
        logger.info(f"  Title I table: {len(df):,} LEA rows")

        # Normalise column names
        df.columns = [c.strip().upper() for c in df.columns]
        beds_col  = next(c for c in df.columns if "BEDS" in c)
        alloc_col = next(c for c in df.columns if "ALLOC" in c)
        df[alloc_col] = pd.to_numeric(df[alloc_col], errors="coerce").fillna(0)
        df[beds_col]  = df[beds_col].astype(str)

        # Extract borough-level allocations using BEDS code prefixes
        borough_alloc = {}
        for borough, prefixes in BOROUGH_BEDS.items():
            mask = df[beds_col].str[:4].isin(prefixes) & df[beds_col].str.endswith("10000")
            total = df.loc[mask, alloc_col].sum()
            if total > 0:
                borough_alloc[borough] = total
                logger.info(f"    {borough}: ${total:,.0f}")

        if not borough_alloc:
            logger.warning("  No borough allocations found in Title I table")
            return pd.DataFrame()

    except Exception as e:
        logger.warning(f"  Title I scrape failed: {e}")
        return pd.DataFrame()

    # Distribute proportionally by poverty headcount within each borough
    s = schools_df.copy()
    s["_pov_count"] = (
        pd.to_numeric(s["total_enrollment"], errors="coerce").fillna(0) *
        pd.to_numeric(s["poverty_pct"], errors="coerce").fillna(0) / 100
    )

    results = []
    for borough, alloc in borough_alloc.items():
        mask = s["borough"] == borough
        boro_df = s[mask].copy()
        total_pov = boro_df["_pov_count"].sum()
        if total_pov > 0:
            boro_df["title1_amount"] = (boro_df["_pov_count"] / total_pov * alloc).round(0)
        else:
            boro_df["title1_amount"] = 0.0
        results.append(boro_df[["dbn", "title1_amount"]])

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    logger.info(f"  ✓ Title I estimated for {len(out):,} schools  "
                f"(total ${out['title1_amount'].sum():,.0f})")
    return out


# ─── Scoring ─────────────────────────────────────────────────────────────────

def calculate_priority_scores(df: pd.DataFrame) -> pd.Series:
    """
    0–100 composite score.

    Weight breakdown:
      30  Economic Need Index (ENI) — primary poverty/Title I signal
      20  CEP status                — binary high-poverty bonus
      20  ELL %                     — multilingual = Gale In Context hook
       5  Title III or high ELL     — dedicated ELL funding bonus
      15  Low ELA proficiency       — unmet literacy need (inverted)
      10  Enrollment                — deal size proxy
    """
    s = pd.Series(0.0, index=df.index)

    def col(name, default=0):
        return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)),
                             errors="coerce").fillna(default)

    # Economic Need Index (0-100 scale in our DB; higher = more need = more points)
    eni = col("economic_need_index")
    if eni.max() > 1:          # 0-100 scale
        s += (eni / 100).clip(0, 1) * 30
    elif eni.max() > 0:        # 0-1 scale (just in case)
        s += eni.clip(0, 1) * 30

    # Title I amount if available (30 pts, log-normalized; additive to ENI if present)
    t1 = col("title1_amount")
    if t1.max() > 0:
        # Replace ENI-based score with Title I-based score for schools that have it
        t1_score = 30 * (np.log1p(t1) / np.log1p(t1.max()))
        has_t1 = t1 > 0
        s = s.where(~has_t1, s - (eni / 100).clip(0, 1) * 30 + t1_score)

    # CEP (20 pts)
    s += col("cep").astype(bool).astype(float) * 20

    # ELL % (20 pts, saturates at 50%)
    ell_pct = col("ell_pct")
    s += (ell_pct / 50).clip(0, 1) * 20

    # Title III or high-ELL bonus (5 pts)
    t3 = col("title3_amount")
    high_ell = (ell_pct >= 20).astype(float)
    has_t3 = (t3 > 0).astype(float) if t3.max() > 0 else pd.Series(0.0, index=df.index)
    s += ((has_t3 + high_ell).clip(0, 1)) * 5

    # Low ELA proficiency (15 pts inverted — lower % = more need = higher score)
    ela = col("ela_proficiency", default=50)
    ela = ela.where(ela > 0, 50)   # treat 0 as missing → neutral
    s += ((100 - ela.clip(0, 100)) / 100) * 15

    # Enrollment (10 pts, normalized)
    enroll = col("total_enrollment")
    if enroll.max() > 0:
        s += (enroll / enroll.max()) * 10

    return s.round(1)


def assign_tiers(scores: pd.Series) -> pd.Series:
    p80 = scores.quantile(0.80)
    p20 = scores.quantile(0.20)
    return pd.cut(scores, bins=[-np.inf, p20, p80, np.inf],
                  labels=["Low", "Medium", "High"])


# ─── Demo seed ────────────────────────────────────────────────────────────────

def generate_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    boroughs = rng.choice(["Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"],
                          n, p=[0.18, 0.28, 0.25, 0.22, 0.07])
    grade_bands = rng.choice(["Elementary", "Middle School", "High School", "K-8", "K-12"],
                              n, p=[0.38, 0.20, 0.25, 0.12, 0.05])
    boro_coords = {
        "Manhattan":     (40.783, -73.971), "Brooklyn":      (40.650, -73.950),
        "Bronx":         (40.844, -73.864), "Queens":        (40.728, -73.794),
        "Staten Island": (40.579, -74.152),
    }
    lats = np.array([boro_coords[b][0] + rng.normal(0, 0.04) for b in boroughs])
    lons = np.array([boro_coords[b][1] + rng.normal(0, 0.04) for b in boroughs])

    df = pd.DataFrame({
        "dbn":                  [f"{rng.integers(1,32):02d}K{i:04d}" for i in range(n)],
        "school_name":          [f"P.S./M.S./H.S. {rng.integers(1,400)} ({b[:3]})"
                                 for b in boroughs],
        "borough":              boroughs,
        "address":              [f"{rng.integers(100,500)} Demo St, NY" for _ in range(n)],
        "grade_band":           grade_bands,
        "lat":                  lats.round(6), "lon": lons.round(6),
        "total_enrollment":     rng.integers(150, 2800, n),
        "ell_count":            rng.integers(0, 600, n),
        "ell_pct":              rng.uniform(0, 55, n).round(1),
        "poverty_pct":          rng.uniform(20, 95, n).round(1),
        "economic_need_index":  rng.uniform(30, 99, n).round(1),
        "ela_proficiency":      rng.uniform(10, 85, n).round(1),
        "cep":                  rng.choice([0, 1], n, p=[0.35, 0.65]),
        "title1_amount":        rng.choice([0, 0, 150000, 350000, 700000, 1200000], n),
        "title3_amount":        rng.choice([0, 0, 50000, 120000], n),
    })
    df["priority_score"] = calculate_priority_scores(df)
    df["tier"] = assign_tiers(df["priority_score"])
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


# ─── Main orchestrator ────────────────────────────────────────────────────────

def refresh_all_data() -> tuple:
    msgs = []
    def log(msg):
        msgs.append(msg)
        logger.info(msg)

    # 1. School directory (required)
    log("① Fetching school directory (p6h4-mpyy)…")
    schools = fetch_school_directory()
    if schools.empty or "dbn" not in schools.columns:
        return False, "\n".join(msgs) + "\n\n✗ School directory failed."
    log(f"   ✓ {len(schools):,} schools loaded")

    # 2. Demographics
    log("② Fetching demographics (vmmu-wj3w) — enrollment, ELL, poverty, ENI…")
    demo = fetch_demographics()
    if not demo.empty:
        schools = schools.merge(
            demo[["dbn", "total_enrollment", "ell_count", "ell_pct",
                  "poverty_pct", "economic_need_index"]],
            on="dbn", how="left")
        log(f"   ✓ Matched {demo['dbn'].nunique():,} schools")
    else:
        for c, v in [("total_enrollment", 0), ("ell_count", 0), ("ell_pct", 0.0),
                     ("poverty_pct", 0.0), ("economic_need_index", 0.0)]:
            schools[c] = v
        log("   ⚠ Unavailable — all demographic fields set to 0")

    # 3. ELA scores
    log("③ Fetching ELA proficiency (iebs-5yhr, 2023)…")
    ela = fetch_ela_scores()
    if not ela.empty:
        schools = schools.merge(ela[["dbn", "ela_proficiency"]], on="dbn", how="left")
        log(f"   ✓ ELA matched for {ela['dbn'].nunique():,} schools")
    else:
        schools["ela_proficiency"] = np.nan
        log("   ⚠ ELA unavailable")

    # 4. CEP status
    log("④ Fetching CEP status (NYSED cn.nysed.gov → poverty ≥ 40% proxy fallback)…")
    demo_for_cep = schools[["dbn", "poverty_pct"]].copy() if "poverty_pct" in schools.columns else None
    cep_df = fetch_cep_status(poverty_df=demo_for_cep)
    if not cep_df.empty:
        if "dbn" in cep_df.columns:
            cep_set = set(cep_df["dbn"])
            schools["cep"] = schools["dbn"].isin(cep_set).astype(int)
        elif "_name_clean" in cep_df.columns:
            schools["_nl"] = schools["school_name"].astype(str).str.lower().str.strip()
            cep_names = set(cep_df["_name_clean"])
            schools["cep"] = schools["_nl"].isin(cep_names).astype(int)
            schools.drop(columns=["_nl"], inplace=True)
        log(f"   ✓ {schools['cep'].sum():,} CEP schools flagged")
    else:
        schools["cep"] = 0
        log("   ⚠ CEP set to 0")

    # 5. Title I — distribute borough county allocations proportionally by poverty headcount
    log("⑤ Fetching Title I allocations (NYSED HTML → distribute by poverty count)…")
    title_df = fetch_title_funding(schools_df=schools[["dbn", "borough",
                                                        "total_enrollment", "poverty_pct"]])
    if not title_df.empty:
        schools = schools.merge(title_df[["dbn", "title1_amount"]], on="dbn", how="left")
        log(f"   ✓ Title I estimated for {(schools['title1_amount'] > 0).sum():,} schools")
    else:
        schools["title1_amount"] = 0.0
        log("   ⚠ Title I unavailable (ENI used as proxy in scoring)")
    schools["title3_amount"] = 0.0  # no public per-school Title III data

    # Fill NaNs
    defaults = {"total_enrollment": 0, "ell_count": 0, "ell_pct": 0.0,
                "poverty_pct": 0.0, "economic_need_index": 0.0,
                "title1_amount": 0.0, "title3_amount": 0.0, "cep": 0}
    for col, default in defaults.items():
        if col not in schools.columns:
            schools[col] = default
        schools[col] = pd.to_numeric(schools[col], errors="coerce").fillna(default)
    if "ela_proficiency" not in schools.columns:
        schools["ela_proficiency"] = np.nan

    # Score
    log("⑥ Calculating priority scores…")
    schools["priority_score"] = calculate_priority_scores(schools)
    schools["tier"] = assign_tiers(schools["priority_score"])
    schools = schools.sort_values("priority_score", ascending=False).reset_index(drop=True)

    final_cols = ["dbn", "school_name", "borough", "address", "grade_band",
                  "lat", "lon", "total_enrollment", "ell_count", "ell_pct",
                  "poverty_pct", "economic_need_index", "ela_proficiency",
                  "cep", "title1_amount", "title3_amount",
                  "priority_score", "tier"]
    for c in final_cols:
        if c not in schools.columns:
            schools[c] = None
    schools = schools[final_cols]

    save_schools(schools)
    log(f"\n✓ Done — {len(schools):,} schools scored and saved.")
    return True, "\n".join(msgs)
