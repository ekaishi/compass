# Gale NYC K12 Sales Compass

A local Streamlit app that ranks all 1,800+ NYC public schools by sales priority
for Gale In Context and Gale eBooks products.

---

## Quick start (two commands)

```bash
# 1. Install dependencies  (one-time; ~30 seconds)
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## First run

The app starts immediately showing **demo data** (200 synthetic schools) so you
can explore the interface right away.
Click **🔄 Refresh Data from Sources** in the sidebar to download and score all
real NYC schools (~60 seconds, requires internet).
Data is cached locally in `schools.db` — subsequent launches are instant.

---

## What the Priority Score measures

Each school receives a 0–100 composite score:

| Signal | Max pts | Why it matters |
|---|---|---|
| Title I funding amount | 30 | Direct budget signal — high $ = high need = more likely to spend |
| CEP status (Community Eligibility Provision) | 20 | Binary flag: school qualifies when ≥40% students qualify for free lunch — strong Title I proxy |
| ELL % (English Language Learners) | 20 | Core hook for Gale In Context multilingual interface |
| Title III indicator | 5 | Has dedicated ELL funding — budget already exists |
| Low ELA proficiency | 15 | Unmet literacy need — clear argument for Gale products |
| Enrollment | 10 | Deal size proxy |

**Tiers:**
- 🟢 **High** — top 20% by score
- 🟡 **Medium** — middle 60%
- 🔴 **Low** — bottom 20%

---

## Data sources

| Data | Source | Notes |
|---|---|---|
| School directory (DBN, name, borough, grade span, coordinates) | [NYC Open Data](https://data.cityofnewyork.us) — School Point Locations | Dataset IDs: `jfju-ynrr`, `s52a-8aq6` |
| Enrollment & ELL count/% | NYC Open Data — DOE Demographic Snapshot | Multiple dataset IDs tried; newest year auto-selected |
| ELA proficiency | NYC Open Data — DOE School Quality Report | Multiple dataset IDs tried |
| CEP school list | [NYSED](https://www.nysed.gov/community-eligibility) — annual Excel download | Matched on school name |
| Title I & III allocations | [NYSED](https://www.nysed.gov/title-i-part) — annual Excel download | Matched on DBN; set to $0 if file unavailable |

All sources are public and require no API key.

### Updating dataset IDs

NYC Open Data publishes a new school year snapshot each fall under a **new
dataset ID**.  If the demographic or quality data looks stale (or returns 0
schools), open `data_fetcher.py` and update the list near the top:

```python
DATASET_IDS = {
    "directory":    ["jfju-ynrr", "s52a-8aq6"],
    "demographics": ["nie6-jtmn", ...],   # ← add new-year ID at front
    "quality":      ["bnrc-gbhg", ...],   # ← same
}
```

Find current dataset IDs by searching "demographic snapshot" or
"school quality" on [data.cityofnewyork.us](https://data.cityofnewyork.us).

---

## App layout

```
Sidebar
  ├─ Refresh Data button
  ├─ Borough filter
  ├─ Grade Band filter  (Elementary / Middle / High / K-8 / K-12 …)
  ├─ CEP filter
  ├─ Title I filter
  ├─ Title III filter
  ├─ Min enrollment slider
  └─ Min priority score slider

Main area
  ├─ 📋 Ranked List  — sortable table, color-coded rows, CSV download
  ├─ 🗺  Map         — Plotly scatter map (dot size = score, color = tier)
  └─ 📊 Charts
        ├─ Top 25 schools bar chart
        ├─ High-priority schools by borough
        ├─ Grade band × tier stacked bar
        ├─ ELL % vs Priority Score scatter
        └─ Title I funding distribution histogram
```

---

## Requirements

- Python 3.10 or newer
- Internet connection for the first Refresh (after that, fully offline)
- No API keys, no accounts, no CRM

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Demo data" message after Refresh | A data source timed out — check internet, retry |
| ELA / CEP columns all blank | NYC Open Data changed the dataset ID — update `DATASET_IDS` in `data_fetcher.py` |
| Map shows no dots | Coordinates come from the school directory; make sure `jfju-ynrr` fetch succeeded |
| App is slow to start | Only on first launch; after `schools.db` is populated it's instant |
