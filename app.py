"""
app.py — Gale K12 NYC Sales Compass
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_fetcher import load_schools, refresh_all_data, generate_demo_data, save_schools

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gale NYC Sales Compass",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Tighten default Streamlit padding */
.block-container { padding-top: 1.2rem; }

/* Tier badge colors */
.badge-high   { background:#14532d; color:#86efac; padding:2px 8px;
                border-radius:4px; font-size:.82rem; font-weight:600; }
.badge-medium { background:#78350f; color:#fcd34d; padding:2px 8px;
                border-radius:4px; font-size:.82rem; font-weight:600; }
.badge-low    { background:#7f1d1d; color:#fca5a5; padding:2px 8px;
                border-radius:4px; font-size:.82rem; font-weight:600; }

/* Metric label size */
[data-testid="metric-container"] label { font-size: .78rem !important; }
</style>
""", unsafe_allow_html=True)

TIER_COLORS = {"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"}
TIER_BG     = {"High": "#14532d22", "Medium": "#78350f22", "Low": "#7f1d1d22"}

# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load():
    return load_schools()


def get_df():
    """Load from cache; fall through to demo if empty."""
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

    # Refresh / demo buttons
    refresh_clicked = st.button("🔄 Refresh Data from Sources",
                                use_container_width=True, type="primary",
                                help="Re-fetches NYC Open Data + NYSED (~60 sec)")
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

    cep_map = {"All schools": None, "CEP schools only": True, "Non-CEP only": False}
    cep_choice = st.radio("CEP Status", list(cep_map.keys()), horizontal=False)

    t1_map = {"All schools": None, "Has Title I funding": True, "No Title I": False}
    t1_choice = st.radio("Title I", list(t1_map.keys()), horizontal=False)

    t3_only = st.checkbox("Title III schools only")

    st.divider()
    enroll_max_val = int(df_all["total_enrollment"].max()) or 3000
    enroll_range = st.slider("Min enrollment", 0, enroll_max_val, 0)

    score_min = st.slider("Min priority score", 0.0,
                          float(df_all["priority_score"].max() or 100), 0.0, 0.5)

# ─── Handle Refresh ──────────────────────────────────────────────────────────

if refresh_clicked:
    with st.spinner("Fetching data… this takes about 60 seconds."):
        ok, log = refresh_all_data()
    if ok:
        st.cache_data.clear()
        st.success("Data refreshed successfully!")
        with st.expander("Fetch log"):
            st.text(log)
        st.rerun()
    else:
        st.error("Refresh failed — check your internet connection.")
        with st.expander("Error log"):
            st.text(log)

# ─── Apply filters ────────────────────────────────────────────────────────────

df = df_all.copy()

if sel_boroughs:
    df = df[df["borough"].isin(sel_boroughs)]
if sel_grades:
    df = df[df["grade_band"].isin(sel_grades)]
if cep_map[cep_choice] is not None:
    df = df[df["cep"].astype(bool) == cep_map[cep_choice]]
if t1_map[t1_choice] is not None:
    if t1_map[t1_choice]:
        df = df[df["title1_amount"] > 0]
    else:
        df = df[df["title1_amount"] == 0]
if t3_only:
    df = df[df["title3_amount"] > 0]

df = df[df["total_enrollment"] >= enroll_range]
df = df[df["priority_score"] >= score_min]
df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

# ─── Header & KPIs ───────────────────────────────────────────────────────────

title_suffix = "  *(demo data — click Refresh to load real schools)*" if is_demo else ""
st.markdown(f"## 🎯 Gale NYC K12 Sales Compass{title_suffix}")
st.caption(f"Showing **{len(df):,}** of **{len(df_all):,}** schools · "
           f"ranked highest priority first")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("High Priority",    f"{(df['tier']=='High').sum():,}",
           help="Top 20% priority score")
k2.metric("CEP Schools",      f"{df['cep'].astype(bool).sum():,}",
           help="Community Eligibility Provision / poverty ≥ 40%")
_t1_count = (df["title1_amount"] > 0).sum() if "title1_amount" in df.columns else 0
k3.metric("Title I Schools",  f"{_t1_count:,}" if _t1_count else "—",
           help="Actual Title I $ when NYSED data available")
k4.metric("Avg ELL %",        f"{df['ell_pct'].mean():.1f}%",
           help="English Language Learners — key Gale In Context hook")
k5.metric("Total Students",   f"{df['total_enrollment'].sum():,.0f}")

st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_list, tab_map, tab_chart = st.tabs(["📋 Ranked List", "🗺 Map", "📊 Charts"])

# ════════════════════════════════════════════════════════
#  TAB 1 — RANKED LIST
# ════════════════════════════════════════════════════════
with tab_list:
    has_title1 = "title1_amount" in df.columns and (df["title1_amount"] > 0).any()

    # ── Selectable table using st.dataframe column_config ──────────────────────
    table_df = df.reset_index(drop=True).copy()

    col_cfg = {
        "school_name":         st.column_config.TextColumn("School Name", width="large"),
        "borough":             st.column_config.TextColumn("Borough"),
        "grade_band":          st.column_config.TextColumn("Grade Band"),
        "priority_score":      st.column_config.NumberColumn("Score", format="%.1f"),
        "tier":                st.column_config.TextColumn("Tier", width="small"),
        "cep":                 st.column_config.CheckboxColumn("CEP"),
        "economic_need_index": st.column_config.NumberColumn("Econ Need", format="%.0f"),
        "ell_pct":             st.column_config.NumberColumn("ELL %", format="%.1f%%"),
        "ell_count":           st.column_config.NumberColumn("ELL Count", format="%d"),
        "ela_proficiency":     st.column_config.NumberColumn("ELA Prof %", format="%.1f%%"),
        "total_enrollment":    st.column_config.NumberColumn("Enrollment", format="%d"),
        "title1_amount":       st.column_config.NumberColumn("Title I $", format="$%,.0f"),
        "poverty_pct":         st.column_config.NumberColumn("Poverty %", format="%.1f%%"),
        # hide raw columns not needed in table
        "dbn": None, "address": None, "lat": None, "lon": None,
        "title3_amount": None,
    }

    show_cols = ["school_name", "borough", "grade_band", "priority_score", "tier",
                 "cep", "economic_need_index", "ell_pct", "ell_count",
                 "ela_proficiency", "total_enrollment"]
    if has_title1:
        show_cols.insert(6, "title1_amount")

    # Convert cep to bool so CheckboxColumn renders correctly
    table_df["cep"] = table_df["cep"].astype(bool)

    selection = st.dataframe(
        table_df[show_cols],
        column_config=col_cfg,
        use_container_width=True,
        height=480,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
    )

    # Tier legend + download row
    leg1, leg2, leg3, _gap, dl_col = st.columns([1, 1, 1, 3, 2])
    leg1.markdown('<span class="badge-high">● High (top 20%)</span>', unsafe_allow_html=True)
    leg2.markdown('<span class="badge-medium">● Medium</span>', unsafe_allow_html=True)
    leg3.markdown('<span class="badge-low">● Low (bottom 20%)</span>', unsafe_allow_html=True)
    dl_col.download_button(
        label="⬇ Download CSV",
        data=df.to_csv(index=False),
        file_name="gale_nyc_targets.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Detail panel ───────────────────────────────────────────────────────────
    selected_rows = selection.selection.get("rows", [])
    if selected_rows:
        idx = selected_rows[0]
        school = df.iloc[idx]
        tier   = str(school.get("tier", "Low"))

        tier_badge = {
            "High":   '<span class="badge-high">High Priority</span>',
            "Medium": '<span class="badge-medium">Medium Priority</span>',
            "Low":    '<span class="badge-low">Low Priority</span>',
        }.get(tier, "")

        st.divider()
        st.markdown(
            f"### {school['school_name']}  {tier_badge}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"DBN {school['dbn']}  ·  {school.get('address', '—')}  ·  "
            f"{school['borough']}  ·  {school['grade_band']}"
        )

        # ── Metric row ──────────────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Priority Score",  f"{school['priority_score']:.1f} / 100")
        m2.metric("Enrollment",      f"{int(school['total_enrollment'] or 0):,}")
        m3.metric("ELL Students",
                  f"{int(school['ell_count'] or 0):,}  ({school['ell_pct']:.1f}%)")
        m4.metric("Poverty %",       f"{school['poverty_pct']:.1f}%" if school['poverty_pct'] else "—")
        m5.metric("ELA Proficiency",
                  f"{school['ela_proficiency']:.1f}%" if pd.notna(school.get('ela_proficiency')) else "—",
                  help="% students at/above proficiency (Level 3+4), 2023 state test")
        m6.metric("Title I Est.",
                  f"${school['title1_amount']:,.0f}" if school.get('title1_amount', 0) > 0 else "—")

        # ── Detail columns ──────────────────────────────────────────────────
        left, mid, right = st.columns([2, 2, 3])

        with left:
            st.markdown("**Need indicators**")
            cep_str = "✅ Yes" if school.get("cep") else "❌ No"
            eni     = school.get("economic_need_index", 0) or 0
            st.markdown(f"""
| Signal | Value |
|---|---|
| CEP status | {cep_str} |
| Economic Need Index | {eni:.0f} / 100 |
| Poverty % | {school['poverty_pct']:.1f}% |
| ELL % | {school['ell_pct']:.1f}% |
| ELA proficiency | {"—" if pd.isna(school.get('ela_proficiency', float('nan'))) else f"{school['ela_proficiency']:.1f}%"} |
""")

        with mid:
            st.markdown("**Gale selling angles**")
            angles = []
            if school['ell_pct'] >= 20:
                angles.append(f"🌐 **{school['ell_pct']:.0f}% ELL** — multilingual Gale In Context interface is a direct fit")
            if school.get("cep"):
                angles.append("📌 **CEP school** — federal funding already in place, budget exists")
            if school.get("title1_amount", 0) > 0:
                angles.append(f"💰 **~${school['title1_amount']:,.0f} Title I** — earmarked for supplemental resources")
            ela = school.get("ela_proficiency")
            if pd.notna(ela) and ela < 40:
                angles.append(f"📖 **{ela:.0f}% ELA proficiency** — strong literacy intervention argument")
            if school['total_enrollment'] >= 800:
                angles.append(f"🏫 **{int(school['total_enrollment']):,} students** — site license scales well")
            if not angles:
                angles.append("Review full profile for selling angles.")
            for a in angles:
                st.markdown(f"- {a}")

        with right:
            # Mini map centred on this school
            lat = pd.to_numeric(school.get("lat"), errors="coerce")
            lon = pd.to_numeric(school.get("lon"), errors="coerce")
            if pd.notna(lat) and pd.notna(lon):
                mini = pd.DataFrame({
                    "lat": [lat], "lon": [lon],
                    "name": [school["school_name"]],
                    "score": [school["priority_score"]],
                    "tier": [tier],
                })
                fig_mini = px.scatter_mapbox(
                    mini, lat="lat", lon="lon",
                    hover_name="name",
                    hover_data={"score": ":.1f", "tier": True,
                                "lat": False, "lon": False},
                    color="tier",
                    color_discrete_map=TIER_COLORS,
                    size="score", size_max=22,
                    zoom=14,
                    center={"lat": float(lat), "lon": float(lon)},
                    mapbox_style="open-street-map",
                )
                fig_mini.update_layout(
                    height=240, margin={"r": 0, "t": 0, "l": 0, "b": 0},
                    showlegend=False,
                )
                st.plotly_chart(fig_mini, use_container_width=True)
            else:
                st.caption("No coordinates available for this school.")

        # ── Score breakdown bar ─────────────────────────────────────────────
        with st.expander("Score breakdown"):
            components = {
                "Economic Need Index": min(30, (eni / 100) * 30),
                "CEP bonus":           20.0 if school.get("cep") else 0.0,
                "ELL %":               min(20, (school["ell_pct"] / 50) * 20),
                "High-ELL bonus":      5.0 if school["ell_pct"] >= 20 else 0.0,
                "Low ELA (inverted)":  ((100 - (ela if pd.notna(ela) else 50)) / 100) * 15,
                "Enrollment":          min(10, (school["total_enrollment"] /
                                              (df["total_enrollment"].max() or 1)) * 10),
            }
            comp_df = pd.DataFrame({
                "Component": list(components.keys()),
                "Points":    [round(v, 1) for v in components.values()],
                "Max":       [30, 20, 20, 5, 15, 10],
            })
            fig_comp = px.bar(
                comp_df, x="Points", y="Component", orientation="h",
                color="Points", color_continuous_scale="RdYlGn",
                range_color=[0, 30],
                text="Points",
                labels={"Points": "Points earned"},
            )
            fig_comp.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_comp.update_layout(height=280, margin={"t": 10, "b": 10},
                                   coloraxis_showscale=False,
                                   yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.caption("← Click any row to see the school detail panel.")

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
        st.info("No geo-coded schools in the current filter selection. "
                "Coordinates come from the school directory — refresh data to populate.")
    else:
        st.caption(f"Mapping {len(map_df):,} schools with coordinates. "
                   "Dot size = priority score · color = tier.")

        fig = px.scatter_mapbox(
            map_df,
            lat="lat", lon="lon",
            color="tier",
            color_discrete_map=TIER_COLORS,
            size="priority_score",
            size_max=18,
            hover_name="school_name",
            hover_data={
                "priority_score": ":.1f",
                "borough": True,
                "grade_band": True,
                "cep": True,
                "ell_count": True,
                "ela_proficiency": ":.1f",
                "tier": False, "lat": False, "lon": False,
            },
            zoom=10,
            center={"lat": 40.710, "lon": -73.960},
            mapbox_style="open-street-map",
            category_orders={"tier": ["High", "Medium", "Low"]},
        )
        fig.update_layout(
            height=640,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend_title_text="Priority Tier",
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
#  TAB 3 — CHARTS
# ════════════════════════════════════════════════════════
with tab_chart:
    col_a, col_b = st.columns(2)

    # Top 25 schools horizontal bar
    with col_a:
        top_n = min(25, len(df))
        top25 = df.nlargest(top_n, "priority_score").copy()
        top25["label"] = top25["school_name"].str[:40]
        fig_top = px.bar(
            top25,
            x="priority_score", y="label",
            orientation="h",
            color="tier",
            color_discrete_map=TIER_COLORS,
            title=f"Top {top_n} Schools by Priority Score",
            labels={"priority_score": "Score", "label": ""},
            category_orders={"tier": ["High", "Medium", "Low"]},
        )
        fig_top.update_layout(height=520, yaxis={"categoryorder": "total ascending"},
                               showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    # Borough breakdown
    with col_b:
        boro_agg = (df.groupby("borough", as_index=False)
                    .agg(schools=("dbn", "count"),
                         high_priority=("tier", lambda x: (x == "High").sum()),
                         avg_score=("priority_score", "mean"))
                    .sort_values("high_priority", ascending=False))

        fig_boro = px.bar(
            boro_agg,
            x="borough", y="high_priority",
            color="avg_score",
            color_continuous_scale="RdYlGn",
            text="high_priority",
            title="High-Priority Schools by Borough",
            labels={"high_priority": "# High-Priority Schools",
                    "avg_score": "Avg Score"},
        )
        fig_boro.update_traces(textposition="outside")
        fig_boro.update_layout(height=360, coloraxis_showscale=True)
        st.plotly_chart(fig_boro, use_container_width=True)

        # Grade band tier stack
        grade_agg = (df.groupby(["grade_band", "tier"])
                     .size().reset_index(name="count"))
        fig_grade = px.bar(
            grade_agg,
            x="grade_band", y="count", color="tier",
            color_discrete_map=TIER_COLORS,
            title="Schools by Grade Band & Tier",
            barmode="stack",
            labels={"count": "Schools", "grade_band": ""},
            category_orders={"tier": ["High", "Medium", "Low"]},
        )
        fig_grade.update_layout(height=300, legend_title_text="Tier")
        st.plotly_chart(fig_grade, use_container_width=True)

    # ELL % vs Priority Score scatter (full width)
    scatter_df = df[df["ell_count"] > 0].copy()
    if not scatter_df.empty:
        enroll_size = scatter_df["total_enrollment"].replace(0, 300).clip(lower=100)
        fig_scatter = px.scatter(
            scatter_df,
            x="ell_pct", y="priority_score",
            color="tier",
            size=enroll_size,
            size_max=28,
            hover_name="school_name",
            hover_data={"borough": True, "grade_band": True,
                        "cep": True, "title1_amount": ":,.0f"},
            color_discrete_map=TIER_COLORS,
            title="ELL % vs Priority Score  (bubble size = enrollment)",
            labels={"ell_pct": "ELL %", "priority_score": "Priority Score"},
            category_orders={"tier": ["High", "Medium", "Low"]},
        )
        fig_scatter.update_layout(height=420, legend_title_text="Tier")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Title I funding distribution
    t1_df = df[df["title1_amount"] > 0].copy()
    if not t1_df.empty:
        fig_t1 = px.histogram(
            t1_df,
            x="title1_amount",
            color="tier",
            color_discrete_map=TIER_COLORS,
            nbins=30,
            title="Title I Funding Distribution (schools with any Title I $)",
            labels={"title1_amount": "Title I Allocation ($)"},
            barmode="overlay",
            opacity=0.75,
            category_orders={"tier": ["High", "Medium", "Low"]},
        )
        fig_t1.update_layout(height=320, legend_title_text="Tier")
        fig_t1.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_t1, use_container_width=True)
