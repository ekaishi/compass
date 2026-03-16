"""
app.py — Gale K12 NYC Sales Compass
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_fetcher import (
    load_schools, refresh_all_data, generate_demo_data, save_schools,
    load_subway_stations, fetch_subway_stations, save_subway_stations,
    map_schools_to_stations, _haversine_km,
)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gale NYC Sales Compass",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.badge-high   { background:#14532d; color:#86efac; padding:2px 8px;
                border-radius:4px; font-size:.82rem; font-weight:600; }
.badge-medium { background:#78350f; color:#fcd34d; padding:2px 8px;
                border-radius:4px; font-size:.82rem; font-weight:600; }
.badge-low    { background:#7f1d1d; color:#fca5a5; padding:2px 8px;
                border-radius:4px; font-size:.82rem; font-weight:600; }
[data-testid="metric-container"] label { font-size: .78rem !important; }
</style>
""", unsafe_allow_html=True)

TIER_COLORS = {"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"}

# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load():
    return load_schools()


def get_df():
    df, last_updated = cached_load()
    is_demo = False
    if df.empty:
        df = generate_demo_data()
        last_updated = None
        is_demo = True
    return df, last_updated, is_demo


df_all, last_updated, is_demo = get_df()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 Gale Compass")
    st.caption("NYC K12 School Targeting Tool")
    st.divider()

    refresh_clicked = st.button("🔄 Refresh Data from Sources",
                                use_container_width=True, type="primary",
                                help="Re-fetches NYC Open Data + NYSED + generates AI summaries (~2–5 min)")
    if is_demo:
        st.info("Showing **demo data** (200 synthetic schools). "
                "Click Refresh to load all 1,800+ real NYC schools.")
    elif last_updated:
        st.caption(f"Last refreshed: {last_updated[:16].replace('T', ' ')}")

    st.divider()
    st.subheader("Filters")

    boroughs = sorted(df_all["borough"].dropna().unique())
    sel_boroughs = st.multiselect("Borough", boroughs, default=boroughs)

    grade_bands = sorted(df_all["grade_band"].dropna().unique())
    sel_grades = st.multiselect("Grade Band", grade_bands, default=grade_bands)

    t1_map = {"All schools": None, "Has Title I funding": True, "No Title I": False}
    t1_choice = st.radio("Title I", list(t1_map.keys()), horizontal=False)

    t3_map = {"All schools": None, "Has Title III funding": True, "No Title III": False}
    t3_choice = st.radio("Title III", list(t3_map.keys()), horizontal=False)

    ell_max_val = float(df_all["ell_pct"].max()) if not df_all.empty else 100.0
    ell_range = st.slider("ELL % range", 0.0, ell_max_val,
                          (0.0, ell_max_val), step=0.5,
                          help="Filter by English Language Learner percentage")

    st.markdown("**iPlan Goals** (filter by AI-detected priorities)")
    lit_only  = st.checkbox("Has literacy goal")
    ell_only  = st.checkbox("Has ELL/multilingual goal")
    att_only  = st.checkbox("Has attendance goal")

    st.divider()
    enroll_max_val = int(df_all["total_enrollment"].max()) or 3000
    enroll_range = st.slider("Min enrollment", 0, enroll_max_val, 0)
    score_min    = st.slider("Min priority score", 0.0,
                             float(df_all["priority_score"].max() or 100), 0.0, 0.5)

# ─── Handle Refresh ──────────────────────────────────────────────────────────

if refresh_clicked:
    with st.spinner("Fetching data + generating AI summaries… this takes 2–5 minutes."):
        ok, log_text = refresh_all_data()
    if ok:
        st.cache_data.clear()
        st.success("Data refreshed successfully!")
        with st.expander("Fetch log"):
            st.text(log_text)
        st.rerun()
    else:
        st.error("Refresh failed — check your internet connection.")
        with st.expander("Error log"):
            st.text(log_text)

# ─── Apply filters ────────────────────────────────────────────────────────────

df = df_all.copy()

if sel_boroughs:
    df = df[df["borough"].isin(sel_boroughs)]
if sel_grades:
    df = df[df["grade_band"].isin(sel_grades)]
if t1_map[t1_choice] is not None:
    if t1_map[t1_choice]:
        df = df[df["title1_amount"] > 0]
    else:
        df = df[df["title1_amount"] == 0]
if t3_map[t3_choice] is not None:
    if t3_map[t3_choice]:
        df = df[df["title3_amount"] > 0]
    else:
        df = df[df["title3_amount"] == 0]
df = df[df["ell_pct"].between(ell_range[0], ell_range[1])]
if lit_only:
    df = df[df.get("has_literacy_goal", pd.Series(0, index=df.index)).astype(bool)]
if ell_only:
    df = df[df.get("has_ell_goal", pd.Series(0, index=df.index)).astype(bool)]
if att_only:
    df = df[df.get("has_attendance_goal", pd.Series(0, index=df.index)).astype(bool)]

df = df[df["total_enrollment"] >= enroll_range]
df = df[df["priority_score"] >= score_min]
df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pad_dbn(dbn: str) -> str:
    """Ensure DBN is 6-char zero-padded format, e.g. '1M15' → '01M015'."""
    s = str(dbn).strip().upper()
    if len(s) >= 6:
        return s[:6]
    for i, c in enumerate(s):
        if c.isalpha():
            return s[:i].zfill(2) + c + s[i + 1:].zfill(3)
    return s


_FOCUS_THEMES = {
    "literacy":    ["literacy", "reading", "ela", "fluency", "comprehension"],
    "ELL support": ["ell", "multilingual", "english language", "newcomer", "bilingual"],
    "attendance":  ["attendance", "absenteeism", "chronic absence"],
    "math":        ["math", "numeracy", "algebra", "stem"],
    "rigor":       ["rigor", "college", "career", "grade-level", "academic achievement"],
    "SEL":         ["social-emotional", "sel", "mental health", "wellness", "climate"],
    "library":     ["library", "research", "database", "digital", "resources"],
    "family":      ["family", "community", "parent"],
}


def extract_focus_goals(text: str, n: int = 3) -> str:
    """Return top N focus keywords from CEP summary text as 'a · b · c'."""
    if not text:
        return ""
    tl = text.lower()
    found = []
    for theme, keywords in _FOCUS_THEMES.items():
        if any(kw in tl for kw in keywords):
            found.append(theme)
        if len(found) >= n:
            break
    return " · ".join(found[:n])


# ─── Helper: radar chart ──────────────────────────────────────────────────────

def _radar_values(school, df_ref):
    """Return normalized (0–1) values for each radar axis."""
    ela = school.get("ela_proficiency")
    ela_val = (1 - float(ela) / 100.0) if pd.notna(ela) and float(ela) > 0 else 0.5  # inverted: lower ELA = more need

    ell_pct = float(school.get("ell_pct") or 0)
    ell_val = min(ell_pct / 50.0, 1.0)

    t1 = float(school.get("title1_amount") or 0)
    t1_max = float(df_ref["title1_amount"].max() or 1)
    if t1 > 0 and t1_max > 0:
        t1_val = float(np.log1p(t1) / np.log1p(t1_max))
    else:
        pov = float(school.get("poverty_pct") or 0)
        t1_val = min(pov / 100.0, 1.0)

    enroll = float(school.get("total_enrollment") or 0)
    enroll_max = float(df_ref["total_enrollment"].max() or 1)
    enroll_val = enroll / enroll_max if enroll_max > 0 else 0.0

    lit_val = 1.0 if school.get("has_literacy_goal") else 0.0
    att_val = 1.0 if school.get("has_attendance_goal") else 0.0

    return {
        "ELA Need":        ela_val,
        "ELL Population":  ell_val,
        "Attendance Flag": att_val,
        "Title I Level":   t1_val,
        "Enrollment":      enroll_val,
        "Literacy Goal":   lit_val,
    }


def _make_radar(school, df_ref, name=None, color="#6366f1"):
    vals = _radar_values(school, df_ref)
    cats = list(vals.keys())
    r    = list(vals.values()) + [list(vals.values())[0]]  # close the polygon
    cats_closed = cats + [cats[0]]
    label = name or str(school.get("school_name", ""))[:40]
    trace = go.Scatterpolar(
        r=r, theta=cats_closed,
        fill="toself", fillcolor=color + "33",
        line=dict(color=color, width=2),
        name=label,
    )
    return trace, cats


def _pitch_angle(school):
    """One-sentence pitch angle from highest-signal axes."""
    ela  = school.get("ela_proficiency")
    ell  = float(school.get("ell_pct") or 0)
    has_lit = bool(school.get("has_literacy_goal"))
    has_ell = bool(school.get("has_ell_goal"))
    t1   = float(school.get("title1_amount") or 0)
    name = str(school.get("school_name", "This school"))[:35]

    if has_lit and has_ell:
        return (f"With both literacy and ELL goals confirmed in their iPlan, "
                f"{name} has documented the exact gaps that Gale In Context is built to close.")
    elif has_ell or ell >= 20:
        return (f"{ell:.0f}% ELL enrollment makes Gale In Context's 30-language interface "
                f"a natural fit — multilingual content, zero IT setup.")
    elif has_lit or (pd.notna(ela) and float(ela) < 35):
        ela_str = f"{ela:.0f}%" if pd.notna(ela) else "below average"
        return (f"ELA proficiency at {ela_str} signals a pressing literacy gap — "
                f"Gale eBooks and In Context give students grade-leveled, curriculum-aligned resources.")
    elif t1 > 500000:
        return (f"~${t1:,.0f} in estimated Title I funding means federal dollars are already "
                f"earmarked for exactly the kind of supplemental resources Gale provides.")
    else:
        return ("Strong need profile — reference depth, multilingual support, and "
                "standards-aligned content map directly to Gale's core value proposition.")


# ─── Header & KPIs ───────────────────────────────────────────────────────────

title_suffix = "  *(demo data — click Refresh)*" if is_demo else ""
st.markdown(f"## 🎯 Gale NYC K12 Sales Compass{title_suffix}")
st.caption(f"Showing **{len(df):,}** of **{len(df_all):,}** schools · ranked highest priority first")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("High Priority",    f"{(df['tier']=='High').sum():,}", help="Top 20% priority score")
_lit_count = int(df.get("has_literacy_goal", pd.Series(0, index=df.index)).astype(bool).sum())
k2.metric("Literacy Goal",    f"{_lit_count:,}",
           help="Schools with literacy flagged as a priority in their iPlan (AI-detected)")
_t1_count = int((df["title1_amount"] > 0).sum()) if "title1_amount" in df.columns else 0
k3.metric("Title I Schools",  f"{_t1_count:,}" if _t1_count else "—",
           help="Estimated Title I allocation > $0")
k4.metric("Avg ELL %",        f"{df['ell_pct'].mean():.1f}%",
           help="English Language Learners — key Gale In Context hook")
k5.metric("Total Students",   f"{df['total_enrollment'].sum():,.0f}")

st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_list, tab_map, tab_chart, tab_geo = st.tabs(
    ["📋 Ranked List", "🗺 Map", "📊 Charts", "🗺 Geography"]
)

# ════════════════════════════════════════════════════════
#  TAB 1 — RANKED LIST
# ════════════════════════════════════════════════════════
with tab_list:
    has_title1 = "title1_amount" in df.columns and (df["title1_amount"] > 0).any()

    table_df = df.reset_index(drop=True).copy()

    # Zero-pad DBN for URL construction
    table_df["_dbn_padded"] = table_df["dbn"].apply(_pad_dbn)

    # CEP link: only show if PDF was confirmed available (or not yet checked)
    # "unavailable"/"data"/"error" = PDF was inaccessible; don't show a broken link
    def _cep_url(row):
        src = str(row.get("summary_source", ""))
        if src in ("unavailable", "data", "error"):
            return None
        return ("https://www.nycenet.edu/documents/oaosi/cep/2024-25/CEP_"
                + row["_dbn_padded"] + ".pdf")

    table_df["cep_pdf_url"]  = table_df.apply(_cep_url, axis=1)
    table_df["snapshot_url"] = ("https://tools.nycps.org/reports/quality-snapshot/"
                                + table_df["_dbn_padded"])

    # Focus goals from CEP summary
    table_df["focus_goals"] = table_df.get("needs_summary",
                                            pd.Series("", index=table_df.index)).apply(
        lambda t: extract_focus_goals(str(t or ""))
    )

    # Boolean goal columns
    for gc in ["has_literacy_goal", "has_ell_goal"]:
        if gc in table_df.columns:
            table_df[gc] = table_df[gc].astype(bool)

    score_help = (
        "Composite score (0–100): "
        "Title I funding / poverty proxy (30 pts) · "
        "ELL % (20 pts) · "
        "ELL/Title III bonus (5 pts) · "
        "Low ELA proficiency (15 pts) · "
        "Enrollment (10 pts) · "
        "CEP literacy goal (12 pts) · "
        "CEP ELL goal (8 pts)"
    )

    col_cfg = {
        "school_name":       st.column_config.TextColumn("School Name", width="large"),
        "borough":           st.column_config.TextColumn("Borough", width="small"),
        "grade_band":        st.column_config.TextColumn("Grade Band"),
        "priority_score":    st.column_config.NumberColumn("Score ℹ️",
                                 format="%.1f", help=score_help),
        "tier":              st.column_config.TextColumn("Tier", width="small"),
        "title1_amount":     st.column_config.NumberColumn("Title I $", format="$%,.0f",
                                 help="Estimated Title I Part A allocation (NYSED, distributed by poverty count)"),
        "title3_amount":     st.column_config.NumberColumn("Title III $", format="$%,.0f",
                                 help="Estimated Title III Part A allocation (NYSED, distributed by ELL count)"),
        "ell_pct":           st.column_config.NumberColumn("ELL %", format="%.1f%%"),
        "ell_count":         st.column_config.NumberColumn("ELL Count", format="%d"),
        "ela_proficiency":   st.column_config.NumberColumn("ELA Prof %", format="%.1f%%"),
        "total_enrollment":  st.column_config.NumberColumn("Enrollment", format="%d"),
        "has_literacy_goal": st.column_config.CheckboxColumn("Literacy Goal",
                                 help="CEP identifies literacy/ELA as a school priority"),
        "has_ell_goal":      st.column_config.CheckboxColumn("ELL Goal",
                                 help="CEP identifies ELL/multilingual as a school priority"),
        "focus_goals":       st.column_config.TextColumn("Focus Goals",
                                 help="Top themes extracted from school's CEP summary"),
        "cep_pdf_url":       st.column_config.LinkColumn("CEP",
                                 display_text="📄 PDF", help="Comprehensive Education Plan PDF (opens new tab)"),
        "snapshot_url":      st.column_config.LinkColumn("Snapshot",
                                 display_text="📊 View", help="NYC Quality Snapshot (opens new tab)"),
        # hide raw/internal columns
        "dbn": None, "_dbn_padded": None, "address": None, "lat": None, "lon": None,
        "poverty_pct": None, "has_attendance_goal": None, "has_library_mention": None,
        "needs_summary": None, "summary_source": None,
        "nearest_station": None, "nearest_line": None, "station_walk_min": None,
    }

    # Column order per spec
    show_cols = [
        "school_name", "borough", "grade_band", "priority_score", "tier",
        "title1_amount", "title3_amount",
        "ell_pct", "ell_count", "ela_proficiency", "total_enrollment",
        "has_literacy_goal", "has_ell_goal", "focus_goals",
        "cep_pdf_url", "snapshot_url",
    ]
    # Filter to columns that exist in this dataframe
    show_cols = [c for c in show_cols if c in table_df.columns]

    selection = st.dataframe(
        table_df[show_cols],
        column_config=col_cfg,
        use_container_width=True,
        height=480,
        on_select="rerun",
        selection_mode="multi-row",
        hide_index=True,
    )

    # Tier legend + download row
    leg1, leg2, leg3, _gap, dl_col = st.columns([1, 1, 1, 3, 2])
    leg1.markdown('<span class="badge-high">● High (top 20%)</span>', unsafe_allow_html=True)
    leg2.markdown('<span class="badge-medium">● Medium</span>',       unsafe_allow_html=True)
    leg3.markdown('<span class="badge-low">● Low (bottom 20%)</span>', unsafe_allow_html=True)
    dl_col.download_button(
        label="⬇ Download CSV",
        data=df.to_csv(index=False),
        file_name="gale_nyc_targets.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Detail / Comparison panel ─────────────────────────────────────────────
    selected_rows = selection.selection.get("rows", [])

    if not selected_rows:
        st.caption("← Click any row to see the school detail panel. "
                   "Select up to 4 rows to compare.")

    elif len(selected_rows) > 4:
        st.warning("Select at most 4 schools to compare radar charts.")

    elif len(selected_rows) >= 2:
        # ── Multi-school comparison ───────────────────────────────────────────
        selected_schools = [df.iloc[i] for i in selected_rows]
        st.divider()
        st.markdown(f"### Comparing {len(selected_schools)} schools")

        COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444"]
        fig_cmp = go.Figure()
        for i, sch in enumerate(selected_schools):
            trace, cats = _make_radar(sch, df_all, name=str(sch["school_name"])[:30],
                                      color=COLORS[i])
            fig_cmp.add_trace(trace)
        fig_cmp.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Comparison table
        cmp_rows = []
        for sch in selected_schools:
            ela = sch.get("ela_proficiency")
            cmp_rows.append({
                "School": str(sch["school_name"])[:45],
                "Score":  f"{sch['priority_score']:.1f}",
                "Tier":   str(sch["tier"]),
                "ELL %":  f"{sch['ell_pct']:.1f}%",
                "ELA %":  f"{ela:.1f}%" if pd.notna(ela) else "—",
                "Title I": f"${sch['title1_amount']:,.0f}" if sch.get("title1_amount", 0) > 0 else "—",
                "Literacy": "✅" if sch.get("has_literacy_goal") else "—",
                "ELL Goal": "✅" if sch.get("has_ell_goal") else "—",
            })
        st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, use_container_width=True)

    else:
        # ── Single school detail panel ────────────────────────────────────────
        idx    = selected_rows[0]
        school = df.iloc[idx]
        tier   = str(school.get("tier", "Low"))
        dbn     = str(school["dbn"])
        dbn_pad = _pad_dbn(dbn)

        tier_badge = {
            "High":   '<span class="badge-high">High Priority</span>',
            "Medium": '<span class="badge-medium">Medium Priority</span>',
            "Low":    '<span class="badge-low">Low Priority</span>',
        }.get(tier, "")

        st.divider()
        st.markdown(f"### {school['school_name']}  {tier_badge}", unsafe_allow_html=True)

        cep_src      = str(school.get("summary_source", ""))
        cep_url      = f"https://www.nycenet.edu/documents/oaosi/cep/2024-25/CEP_{dbn_pad}.pdf"
        snapshot_url = f"https://tools.nycps.org/reports/quality-snapshot/{dbn_pad}"
        cep_link     = (f"[📄 CEP PDF]({cep_url}){{target=_blank}}"
                        if cep_src != "unavailable"
                        else "📄 CEP unavailable")
        st.caption(
            f"DBN {dbn_pad}  ·  {school.get('address', '—')}  ·  "
            f"{school['borough']}  ·  {school['grade_band']}  ·  "
            f"[📄 CEP PDF]({cep_url})  ·  [📊 Quality Snapshot]({snapshot_url})"
        )

        # Metric row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Priority Score",  f"{school['priority_score']:.1f} / 100")
        m2.metric("Enrollment",      f"{int(school['total_enrollment'] or 0):,}")
        m3.metric("ELL Students",
                  f"{int(school['ell_count'] or 0):,}  ({school['ell_pct']:.1f}%)")
        m4.metric("Poverty %",       f"{school['poverty_pct']:.1f}%" if school['poverty_pct'] else "—")
        ela = school.get("ela_proficiency")
        m5.metric("ELA Proficiency",
                  f"{ela:.1f}%" if pd.notna(ela) else "—",
                  help="% students at/above proficiency (Level 3+4), 2023 state test")
        m6.metric("Title I Est.",
                  f"${school['title1_amount']:,.0f}" if school.get("title1_amount", 0) > 0 else "—")

        # Main detail columns
        left, mid, right = st.columns([3, 3, 4])

        # ── Needs summary (AI) ────────────────────────────────────────────────
        with left:
            st.markdown("**School Needs Summary**")
            summary = str(school.get("needs_summary") or "").strip()
            src     = str(school.get("summary_source") or "")
            if summary:
                src_label = {"cep_pdf": "from CEP PDF",
                             "data":    "from quantitative data",
                             "error":   "unavailable"}.get(src, "")
                if src_label:
                    st.caption(f"AI summary ({src_label})")
                for line in summary.split("\n"):
                    if line.strip():
                        st.markdown(line)
            else:
                st.caption("No AI summary yet for this school.")
                if st.button("✨ Generate Summary", key=f"gen_{dbn}"):
                    with st.spinner("Generating with Claude…"):
                        try:
                            from ai_summarizer import summarize_school
                            result = summarize_school(
                                dbn, str(school["school_name"]),
                                school_row=school.to_dict(), force=True,
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Summary failed: {e}")

            st.markdown("---")
            st.markdown("**Gale pitch angle**")
            st.info(_pitch_angle(school))

        # ── Selling angles + indicators ───────────────────────────────────────
        with mid:
            st.markdown("**Need indicators**")
            st.markdown(f"""
| Signal | Value |
|---|---|
| Poverty % | {school['poverty_pct']:.1f}% |
| ELL % | {school['ell_pct']:.1f}% |
| ELA proficiency | {"—" if pd.isna(ela) else f"{ela:.1f}%"} |
| Title I est. | {"$" + f"{school['title1_amount']:,.0f}" if school.get('title1_amount', 0) > 0 else "—"} |
| Literacy goal (CEP) | {"✅ Yes" if school.get("has_literacy_goal") else "❌ No"} |
| ELL goal (CEP) | {"✅ Yes" if school.get("has_ell_goal") else "❌ No"} |
| Attendance flag | {"✅ Yes" if school.get("has_attendance_goal") else "—"} |
| Library/digital mention | {"✅ Yes" if school.get("has_library_mention") else "—"} |
""")

            st.markdown("**Gale selling angles**")
            angles = []
            if school["ell_pct"] >= 20:
                angles.append(f"🌐 **{school['ell_pct']:.0f}% ELL** — multilingual Gale In Context interface is a direct fit")
            if school.get("title1_amount", 0) > 0:
                angles.append(f"💰 **~${school['title1_amount']:,.0f} Title I** — earmarked for supplemental resources")
            if pd.notna(ela) and float(ela) < 40:
                angles.append(f"📖 **{ela:.0f}% ELA proficiency** — strong literacy intervention argument")
            if school.get("has_literacy_goal"):
                angles.append("📌 **iPlan literacy goal** — school leadership has already prioritised this")
            if school.get("has_ell_goal"):
                angles.append("🌍 **iPlan ELL goal** — multilingual resources align with school's own plan")
            if school["total_enrollment"] >= 800:
                angles.append(f"🏫 **{int(school['total_enrollment']):,} students** — site license scales well")
            if not angles:
                angles.append("Review full profile for selling angles.")
            for a in angles:
                st.markdown(f"- {a}")

        # ── Radar chart ───────────────────────────────────────────────────────
        with right:
            st.markdown("**School profile radar**")
            trace, cats = _make_radar(school, df_all, color="#6366f1")
            fig_radar = go.Figure(data=[trace])
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1],
                                    tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=10)),
                ),
                showlegend=False,
                height=320,
                margin=dict(t=20, b=20, l=40, r=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Mini map
            lat = pd.to_numeric(school.get("lat"), errors="coerce")
            lon = pd.to_numeric(school.get("lon"), errors="coerce")
            if pd.notna(lat) and pd.notna(lon):
                mini = pd.DataFrame({
                    "lat": [lat], "lon": [lon],
                    "name": [school["school_name"]],
                    "score": [school["priority_score"]], "tier": [tier],
                })
                fig_mini = px.scatter_mapbox(
                    mini, lat="lat", lon="lon",
                    hover_name="name",
                    hover_data={"score": ":.1f", "tier": True, "lat": False, "lon": False},
                    color="tier", color_discrete_map=TIER_COLORS,
                    size="score", size_max=22, zoom=14,
                    center={"lat": float(lat), "lon": float(lon)},
                    mapbox_style="open-street-map",
                )
                fig_mini.update_layout(height=200, margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                        showlegend=False)
                st.plotly_chart(fig_mini, use_container_width=True)

        # Score breakdown
        with st.expander("Score breakdown"):
            t1_val = float(school.get("title1_amount") or 0)
            t1_max = float(df_all["title1_amount"].max() or 1)
            pov_pct = float(school.get("poverty_pct") or 0)
            if t1_val > 0 and t1_max > 0:
                poverty_t1_pts = round(30 * float(np.log1p(t1_val) / np.log1p(t1_max)), 1)
            else:
                poverty_t1_pts = round((pov_pct / 100) * 30, 1)
            ela_pts = float(((100 - float(ela if pd.notna(ela) else 50)) / 100) * 15)
            components = {
                "Title I / Poverty proxy":  min(30, poverty_t1_pts),
                "ELL %":                    round(min(20, (school["ell_pct"] / 50) * 20), 1),
                "High-ELL bonus":           5.0 if school["ell_pct"] >= 20 else 0.0,
                "Low ELA (inverted)":       round(ela_pts, 1),
                "Enrollment":               round(min(10, (school["total_enrollment"] /
                                                (df_all["total_enrollment"].max() or 1)) * 10), 1),
                "CEP literacy goal":        12.0 if school.get("has_literacy_goal") else 0.0,
                "CEP ELL goal":             8.0  if school.get("has_ell_goal") else 0.0,
            }
            comp_df = pd.DataFrame({
                "Component": list(components.keys()),
                "Points":    list(components.values()),
                "Max":       [30, 20, 5, 15, 10, 12, 8],
            })
            fig_comp = px.bar(comp_df, x="Points", y="Component", orientation="h",
                              color="Points", color_continuous_scale="RdYlGn",
                              range_color=[0, 30], text="Points",
                              labels={"Points": "Points earned"})
            fig_comp.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_comp.update_layout(height=300, margin={"t": 10, "b": 10},
                                    coloraxis_showscale=False,
                                    yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_comp, use_container_width=True)


# ════════════════════════════════════════════════════════
#  TAB 2 — MAP
# ════════════════════════════════════════════════════════
with tab_map:
    map_df = df.copy()
    map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
    map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
    map_df = map_df.dropna(subset=["lat", "lon"])
    map_df = map_df[(map_df["lat"].between(40.4, 41.0)) &
                    (map_df["lon"].between(-74.4, -73.6))]

    if map_df.empty:
        st.info("No geo-coded schools in the current filter selection.")
    else:
        st.caption(f"Mapping {len(map_df):,} schools · dot size = priority score · color = tier")
        fig = px.scatter_mapbox(
            map_df,
            lat="lat", lon="lon",
            color="tier",
            color_discrete_map=TIER_COLORS,
            size="priority_score", size_max=18,
            hover_name="school_name",
            hover_data={
                "priority_score": ":.1f", "borough": True, "grade_band": True,
                "ell_count": True, "ela_proficiency": ":.1f",
                "tier": False, "lat": False, "lon": False,
            },
            zoom=10,
            center={"lat": 40.710, "lon": -73.960},
            mapbox_style="open-street-map",
            category_orders={"tier": ["High", "Medium", "Low"]},
        )
        fig.update_layout(height=640, margin={"r": 0, "t": 0, "l": 0, "b": 0},
                           legend_title_text="Priority Tier")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
#  TAB 3 — CHARTS
# ════════════════════════════════════════════════════════
with tab_chart:
    col_a, col_b = st.columns(2)

    with col_a:
        top_n = min(25, len(df))
        top25 = df.nlargest(top_n, "priority_score").copy()
        top25["label"] = top25["school_name"].str[:45]
        fig_top = px.bar(top25, x="priority_score", y="label", orientation="h",
                         color="tier", color_discrete_map=TIER_COLORS,
                         title=f"Top {top_n} Schools by Priority Score",
                         labels={"priority_score": "Score", "label": ""},
                         category_orders={"tier": ["High", "Medium", "Low"]})
        fig_top.update_layout(height=540, yaxis={"categoryorder": "total ascending"},
                               showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with col_b:
        boro_agg = (df.groupby("borough", as_index=False)
                    .agg(schools=("dbn", "count"),
                         high_priority=("tier", lambda x: (x == "High").sum()),
                         avg_score=("priority_score", "mean"))
                    .sort_values("high_priority", ascending=False))
        fig_boro = px.bar(boro_agg, x="borough", y="high_priority",
                          color="avg_score", color_continuous_scale="RdYlGn",
                          text="high_priority",
                          title="High-Priority Schools by Borough",
                          labels={"high_priority": "# High-Priority", "avg_score": "Avg Score"})
        fig_boro.update_traces(textposition="outside")
        fig_boro.update_layout(height=360, coloraxis_showscale=True)
        st.plotly_chart(fig_boro, use_container_width=True)

        grade_agg = df.groupby(["grade_band", "tier"]).size().reset_index(name="count")
        fig_grade = px.bar(grade_agg, x="grade_band", y="count", color="tier",
                           color_discrete_map=TIER_COLORS,
                           title="Schools by Grade Band & Tier", barmode="stack",
                           labels={"count": "Schools", "grade_band": ""},
                           category_orders={"tier": ["High", "Medium", "Low"]})
        fig_grade.update_layout(height=300, legend_title_text="Tier")
        st.plotly_chart(fig_grade, use_container_width=True)

    scatter_df = df[df["ell_count"] > 0].copy()
    if not scatter_df.empty:
        enroll_size = scatter_df["total_enrollment"].replace(0, 300).clip(lower=100)
        fig_scatter = px.scatter(scatter_df, x="ell_pct", y="priority_score", color="tier",
                                 size=enroll_size, size_max=28,
                                 hover_name="school_name",
                                 hover_data={"borough": True, "grade_band": True,
                                              "title1_amount": ":,.0f"},
                                 color_discrete_map=TIER_COLORS,
                                 title="ELL % vs Priority Score  (bubble size = enrollment)",
                                 labels={"ell_pct": "ELL %", "priority_score": "Priority Score"},
                                 category_orders={"tier": ["High", "Medium", "Low"]})
        fig_scatter.update_layout(height=420, legend_title_text="Tier")
        st.plotly_chart(fig_scatter, use_container_width=True)

    t1_df = df[df["title1_amount"] > 0].copy()
    if not t1_df.empty:
        fig_t1 = px.histogram(t1_df, x="title1_amount", color="tier",
                               color_discrete_map=TIER_COLORS, nbins=30,
                               title="Title I Funding Distribution",
                               labels={"title1_amount": "Title I Allocation ($)"},
                               barmode="overlay", opacity=0.75,
                               category_orders={"tier": ["High", "Medium", "Low"]})
        fig_t1.update_layout(height=320, legend_title_text="Tier")
        fig_t1.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_t1, use_container_width=True)


# ════════════════════════════════════════════════════════
#  TAB 4 — GEOGRAPHY (Interactive station + school map)
# ════════════════════════════════════════════════════════
with tab_geo:
    st.markdown("### 🗺 Geography — Schools & Subway Territory")
    st.caption("All filtered schools plotted on the map. Select a subway station to see "
               "nearby schools and their aggregate profile.")

    stations_df = load_subway_stations()

    if stations_df.empty:
        st.info("Subway station data not yet loaded. "
                "Click **Refresh Data** in the sidebar, or load stations only:")
        if st.button("🚇 Load Subway Station Data", type="primary", key="load_geo_stations"):
            with st.spinner("Fetching MTA subway station data…"):
                new_stations = fetch_subway_stations()
                if not new_stations.empty:
                    save_subway_stations(new_stations)
                    st.success(f"Loaded {len(new_stations):,} subway stations.")
                    st.rerun()
                else:
                    st.error("Could not fetch station data — check internet connection.")
    else:
        # ── Controls ────────────────────────────────────────────────────────
        def _split_lines(raw):
            return [p.strip() for p in str(raw).replace(",", " ").split() if p.strip()]

        all_lines = sorted({line for raw in stations_df["line"].dropna()
                            for line in _split_lines(raw)})

        ctrl1, ctrl2 = st.columns([2, 3])
        with ctrl1:
            line_filter = st.multiselect(
                "Filter subway lines (empty = all)",
                all_lines, default=[],
                help="Show only stations serving these lines",
                key="geo_lines",
            )
        with ctrl2:
            station_list = ["(none — show all schools)"]
            if line_filter:
                vis_st = stations_df[
                    stations_df["line"].apply(
                        lambda l: bool(set(_split_lines(l)) & set(line_filter))
                    )
                ]
            else:
                vis_st = stations_df
            station_list += sorted(vis_st["station_name"].unique().tolist())
            sel_station = st.selectbox(
                "Focus station → nearby schools panel",
                station_list, index=0,
                key="geo_station",
            )

        # ── Map data ────────────────────────────────────────────────────────
        geo_schools = df.copy()
        geo_schools["lat"] = pd.to_numeric(geo_schools["lat"], errors="coerce")
        geo_schools["lon"] = pd.to_numeric(geo_schools["lon"], errors="coerce")
        geo_schools = geo_schools.dropna(subset=["lat", "lon"])
        geo_schools = geo_schools[
            geo_schools["lat"].between(40.4, 41.0) &
            geo_schools["lon"].between(-74.4, -73.6)
        ]

        # Determine map center
        if sel_station != "(none — show all schools)":
            st_row = vis_st[vis_st["station_name"] == sel_station]
            map_center = ({"lat": float(st_row.iloc[0]["lat"]),
                           "lon": float(st_row.iloc[0]["lon"])}
                          if not st_row.empty
                          else {"lat": 40.710, "lon": -73.960})
            map_zoom = 13
        else:
            map_center = {"lat": 40.710, "lon": -73.960}
            map_zoom = 10

        # Build map
        if geo_schools.empty:
            st.info("No geo-coded schools in current filter.")
        else:
            fig_geo = px.scatter_mapbox(
                geo_schools,
                lat="lat", lon="lon",
                color="tier", color_discrete_map=TIER_COLORS,
                size="priority_score", size_max=16,
                hover_name="school_name",
                hover_data={
                    "priority_score": ":.1f", "borough": True,
                    "grade_band": True, "ell_pct": ":.1f",
                    "tier": False, "lat": False, "lon": False,
                },
                zoom=map_zoom,
                center=map_center,
                mapbox_style="open-street-map",
                category_orders={"tier": ["High", "Medium", "Low"]},
            )

            # Add subway station layer
            vis_st_plot = vis_st.dropna(subset=["lat", "lon"])
            if not vis_st_plot.empty:
                fig_geo.add_trace(go.Scattermapbox(
                    lat=vis_st_plot["lat"],
                    lon=vis_st_plot["lon"],
                    mode="markers",
                    marker=dict(size=7, color="#64748b"),
                    text=vis_st_plot["station_name"] + " (" + vis_st_plot["line"] + ")",
                    hoverinfo="text",
                    name="Subway Stations",
                    showlegend=True,
                ))

            # Highlight selected station
            if sel_station != "(none — show all schools)" and not st_row.empty:
                fig_geo.add_trace(go.Scattermapbox(
                    lat=st_row["lat"], lon=st_row["lon"],
                    mode="markers",
                    marker=dict(size=18, color="#f59e0b"),
                    text=[sel_station],
                    hoverinfo="text",
                    name=f"⭐ {sel_station}",
                    showlegend=True,
                ))

            fig_geo.update_layout(
                height=560, margin={"r": 0, "t": 0, "l": 0, "b": 0},
                legend_title_text="",
            )
            st.plotly_chart(fig_geo, use_container_width=True)

        # ── Nearby schools panel ─────────────────────────────────────────
        if sel_station != "(none — show all schools)" and not geo_schools.empty:
            st_row = vis_st[vis_st["station_name"] == sel_station]
            if not st_row.empty:
                st_lat = float(st_row.iloc[0]["lat"])
                st_lon = float(st_row.iloc[0]["lon"])
                st_line = str(st_row.iloc[0]["line"])

                dists   = _haversine_km(st_lat, st_lon,
                                        geo_schools["lat"].values,
                                        geo_schools["lon"].values)
                nearby  = geo_schools[dists <= 1.5].copy()
                nearby["walk_min"] = (dists[dists <= 1.5] / 5.0 * 60).round(1)
                nearby  = nearby.sort_values("priority_score", ascending=False)

                st.divider()
                st.markdown(
                    f"#### ⭐ {sel_station}  "
                    f"<span style='color:#94a3b8;font-size:.9em'>"
                    f"Lines {st_line} · {len(nearby)} schools within 1.5 km</span>",
                    unsafe_allow_html=True,
                )

                if nearby.empty:
                    st.info("No schools found within 1.5 km of this station.")
                else:
                    pan_l, pan_r = st.columns([3, 2])

                    with pan_l:
                        nearby_disp = nearby[[
                            "school_name", "tier", "priority_score",
                            "ell_pct", "grade_band", "walk_min",
                        ]].copy()
                        st.dataframe(
                            nearby_disp,
                            column_config={
                                "school_name":    st.column_config.TextColumn("School", width="large"),
                                "tier":           st.column_config.TextColumn("Tier", width="small"),
                                "priority_score": st.column_config.NumberColumn("Score", format="%.1f"),
                                "ell_pct":        st.column_config.NumberColumn("ELL %", format="%.1f%%"),
                                "grade_band":     st.column_config.TextColumn("Grades"),
                                "walk_min":       st.column_config.NumberColumn("Walk", format="%.1f min"),
                            },
                            hide_index=True, use_container_width=True, height=360,
                        )
                        st.download_button(
                            "⬇ Export nearby schools CSV",
                            data=nearby.to_csv(index=False),
                            file_name=f"gale_near_{sel_station.replace(' ', '_')}.csv",
                            mime="text/csv",
                        )

                    with pan_r:
                        st.markdown("**Aggregate profile — avg of nearby schools**")
                        # Build average school dict for radar
                        avg = nearby.mean(numeric_only=True).to_dict()
                        avg["school_name"] = f"Avg ({len(nearby)} schools)"
                        avg["has_literacy_goal"]  = int(
                            nearby["has_literacy_goal"].astype(float).mean() >= 0.5)
                        avg["has_attendance_goal"] = int(
                            nearby["has_attendance_goal"].astype(float).mean() >= 0.5)
                        avg["has_ell_goal"] = int(
                            nearby["has_ell_goal"].astype(float).mean() >= 0.5)

                        trace_avg, _ = _make_radar(avg, df_all,
                                                   name=f"Avg ({len(nearby)} schools)",
                                                   color="#6366f1")
                        fig_nearby = go.Figure(data=[trace_avg])
                        fig_nearby.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                                       tickfont=dict(size=9))),
                            showlegend=False, height=320,
                            margin=dict(t=20, b=20, l=40, r=40),
                        )
                        st.plotly_chart(fig_nearby, use_container_width=True)

                        # Summary stats
                        st.caption(
                            f"Avg score: **{nearby['priority_score'].mean():.1f}**  ·  "
                            f"High priority: **{(nearby['tier']=='High').sum()}** schools  ·  "
                            f"Avg ELL: **{nearby['ell_pct'].mean():.1f}%**"
                        )
