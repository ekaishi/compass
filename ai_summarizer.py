"""
ai_summarizer.py — AI-powered school needs summaries using Claude.

For each school, fetches the NYC CEP (Comprehensive Education Plan) PDF at:
  https://www.nycenet.edu/documents/oaosi/cep/2025-26/cep_{SHORT_DBN}.pdf
  (short DBN = strip district prefix: 01M015 → M015)
  Falls back to 2024-25 if the 2025-26 PDF returns 404.

Claude reads the PDF and extracts:
  - ELA / literacy goals and current gap
  - ELL / Multilingual Learner priorities
  - Attendance / chronic absenteeism goals
  - Any mentions of library, research, or digital learning tools

If the CEP PDF is unavailable (requires login, 404, too large, etc.), Claude
generates a data-driven summary from the school's quantitative signals instead.

Model: claude-haiku-4-5 (fast, cost-efficient for batch processing).
Caching: SQLite school_summaries table. Only re-generated on Refresh.
"""

import os
import time
import base64
import sqlite3
import logging
import requests
import anthropic
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schools.db")

# Try 2025-26 first, fall back to 2024-25. Use lowercase filename, short DBN (M015 not 01M015).
CEP_PDF_URLS = [
    "https://www.nycenet.edu/documents/oaosi/cep/2025-26/cep_{short_dbn}.pdf",
    "https://www.nycenet.edu/documents/oaosi/cep/2024-25/cep_{short_dbn}.pdf",
]


def _dbn_to_short(dbn: str) -> str:
    """Strip district prefix: '01M015' → 'M015'."""
    s = str(dbn).strip().upper()
    for i, c in enumerate(s):
        if c.isalpha():
            return s[i:]
    return s

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; gale-sales-tool/1.0)"}

# Max PDF size to pass inline to Claude (5 MB)
MAX_PDF_BYTES = 5 * 1024 * 1024


# ─── Pydantic schema for structured output ────────────────────────────────────

class NeedsSummary(BaseModel):
    bullets: list[str]              # 2–4 bullet points
    has_literacy_goal: bool         # CEP prioritises literacy / ELA
    has_ell_goal: bool              # CEP prioritises ELL / multilingual learners
    has_attendance_goal: bool       # Attendance / chronic absenteeism flagged
    has_library_mention: bool       # Mentions library, research tools, or digital learning


# ─── SQLite helpers ───────────────────────────────────────────────────────────

def _conn():
    return sqlite3.connect(DB_PATH)


def init_summary_table():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS school_summaries (
                dbn                 TEXT PRIMARY KEY,
                needs_summary       TEXT,
                has_literacy_goal   INTEGER DEFAULT 0,
                has_ell_goal        INTEGER DEFAULT 0,
                has_attendance_goal INTEGER DEFAULT 0,
                has_library_mention INTEGER DEFAULT 0,
                summary_source      TEXT,
                generated_at        TEXT
            )
        """)
        # Migrate: add has_library_mention if missing
        cols = [r[1] for r in c.execute("PRAGMA table_info(school_summaries)").fetchall()]
        if "has_library_mention" not in cols:
            c.execute("ALTER TABLE school_summaries ADD COLUMN has_library_mention INTEGER DEFAULT 0")


def get_cached_summary(dbn: str) -> Optional[dict]:
    init_summary_table()
    with _conn() as c:
        row = c.execute(
            "SELECT needs_summary, has_literacy_goal, has_ell_goal, "
            "has_attendance_goal, has_library_mention, summary_source, generated_at "
            "FROM school_summaries WHERE dbn = ?", (dbn,)
        ).fetchone()
    if row:
        return {
            "needs_summary":      row[0],
            "has_literacy_goal":  bool(row[1]),
            "has_ell_goal":       bool(row[2]),
            "has_attendance_goal": bool(row[3]),
            "has_library_mention": bool(row[4]),
            "summary_source":     row[5],
            "generated_at":       row[6],
        }
    return None


def save_summary(dbn: str, data: dict):
    init_summary_table()
    with _conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO school_summaries
            (dbn, needs_summary, has_literacy_goal, has_ell_goal,
             has_attendance_goal, has_library_mention, summary_source, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dbn,
            data.get("needs_summary", ""),
            int(data.get("has_literacy_goal", False)),
            int(data.get("has_ell_goal", False)),
            int(data.get("has_attendance_goal", False)),
            int(data.get("has_library_mention", False)),
            data.get("summary_source", "unknown"),
            datetime.now().isoformat(),
        ))


def load_all_summaries():
    """Return all cached summaries as a DataFrame."""
    import pandas as pd
    init_summary_table()
    with _conn() as c:
        return pd.read_sql(
            "SELECT dbn, needs_summary, has_literacy_goal, has_ell_goal, "
            "has_attendance_goal, has_library_mention, summary_source, generated_at "
            "FROM school_summaries", c
        )


def clear_all_summaries():
    """Delete all cached summaries (called at the start of a full Refresh)."""
    init_summary_table()
    with _conn() as c:
        c.execute("DELETE FROM school_summaries")


# ─── CEP PDF fetching ─────────────────────────────────────────────────────────

def _fetch_cep_pdf(dbn: str) -> tuple:
    """
    Try to download the school's CEP PDF.
    Tries 2025-26 first, then 2024-25. Uses short DBN (M015 not 01M015).
    Returns (pdf_bytes, "cep_pdf") on success, or (b"", "unavailable") on failure.
    """
    short = _dbn_to_short(dbn)
    for url_tmpl in CEP_PDF_URLS:
        url = url_tmpl.format(short_dbn=short)
        try:
            r = requests.get(url, timeout=30, headers=HEADERS, stream=True)
            if r.status_code == 404:
                logger.debug(f"CEP PDF {dbn}: 404 at {url}")
                continue
            if r.status_code != 200:
                logger.debug(f"CEP PDF {dbn}: HTTP {r.status_code} at {url}")
                continue

            content_type = r.headers.get("content-type", "").lower()
            if "pdf" not in content_type and "octet-stream" not in content_type:
                logger.debug(f"CEP PDF {dbn}: unexpected content-type {content_type} at {url}")
                continue

            chunks = []
            total = 0
            for chunk in r.iter_content(chunk_size=65536):
                chunks.append(chunk)
                total += len(chunk)
                if total >= MAX_PDF_BYTES:
                    logger.debug(f"CEP PDF {dbn}: truncated at {MAX_PDF_BYTES} bytes")
                    break
            pdf_bytes = b"".join(chunks)
            logger.debug(f"CEP PDF {dbn}: {len(pdf_bytes):,} bytes from {url}")
            return pdf_bytes, "cep_pdf"

        except Exception as e:
            logger.debug(f"CEP PDF fetch {dbn} at {url}: {e}")
            continue

    return b"", "unavailable"


# ─── Claude summarization ─────────────────────────────────────────────────────

_CEP_PROMPT = """\
You are analyzing a NYC school's Comprehensive Education Plan (CEP) for a \
library database salesperson selling Gale In Context (multilingual reference \
database) and Gale eBooks to K-12 schools.

From the CEP document, extract a structured summary focused on:
1. ELA/literacy goals — include specific targets or proficiency gaps if mentioned
2. ELL/Multilingual Learner goals or supports
3. Attendance/chronic absenteeism — any goals or current rates mentioned
4. Any mention of library resources, research tools, digital learning, or databases

School: {school_name} (DBN {dbn})

Write 2–4 concise bullet points a salesperson can use to open a conversation.
"""

_DATA_PROMPT = """\
You are analyzing NYC school data for a library database salesperson selling \
Gale In Context (multilingual reference database) and Gale eBooks to K-12 schools.

The school's CEP PDF was not accessible. Based on the quantitative profile below, \
write a needs summary that a salesperson could use to open a conversation.

School: {school_name} (DBN {dbn})
Borough: {borough} | Grade band: {grade_band}
Total enrollment: {enrollment}
ELL students: {ell_count} ({ell_pct:.1f}% of enrollment)
Poverty rate: {poverty_pct:.1f}%
ELA proficiency (2023): {ela}
Estimated Title I allocation: {title1}
"""


def _call_claude_with_pdf(pdf_bytes: bytes, prompt: str) -> Optional[NeedsSummary]:
    """Call Claude with a PDF document and structured output prompt."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    try:
        response = client.messages.parse(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            output_format=NeedsSummary,
        )
        return response.parsed_output
    except Exception as e:
        logger.warning(f"Claude PDF API call failed: {e}")
        return None


def _call_claude_text(prompt: str) -> Optional[NeedsSummary]:
    """Call Claude with a text-only prompt and structured output."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping AI summary")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.parse(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            output_format=NeedsSummary,
        )
        return response.parsed_output
    except Exception as e:
        logger.warning(f"Claude text API call failed: {e}")
        return None


# ─── Public interface ─────────────────────────────────────────────────────────

def summarize_school(
    dbn: str,
    school_name: str,
    school_row: Optional[dict] = None,
    force: bool = False,
) -> dict:
    """
    Generate (or return cached) needs summary for one school.

    school_row: dict with keys enrollment, ell_count, ell_pct, poverty_pct,
                ela_proficiency, title1_amount, borough, grade_band.
    force: ignore cache and re-generate.
    """
    if not force:
        cached = get_cached_summary(dbn)
        if cached:
            return cached

    # 1. Try to fetch and read the CEP PDF
    pdf_bytes, source = _fetch_cep_pdf(dbn)

    if pdf_bytes:
        prompt = _CEP_PROMPT.format(school_name=school_name, dbn=dbn)
        result = _call_claude_with_pdf(pdf_bytes, prompt)
    else:
        # Build data-driven prompt from quantitative signals
        r = school_row or {}
        ela_str = (f"{r.get('ela_proficiency', 0):.1f}%"
                   if r.get("ela_proficiency") else "unavailable")
        t1_str  = (f"${r.get('title1_amount', 0):,.0f}"
                   if r.get("title1_amount", 0) > 0 else "unknown")
        prompt = _DATA_PROMPT.format(
            school_name=school_name, dbn=dbn,
            borough=r.get("borough", "—"),
            grade_band=r.get("grade_band", "—"),
            enrollment=int(r.get("total_enrollment", 0) or 0),
            ell_count=int(r.get("ell_count", 0) or 0),
            ell_pct=float(r.get("ell_pct", 0) or 0),
            poverty_pct=float(r.get("poverty_pct", 0) or 0),
            ela=ela_str,
            title1=t1_str,
        )
        result = _call_claude_text(prompt)

    if result is None:
        data = {
            "needs_summary":      "AI summary unavailable — set ANTHROPIC_API_KEY.",
            "has_literacy_goal":  False,
            "has_ell_goal":       False,
            "has_attendance_goal": False,
            "has_library_mention": False,
            "summary_source":     "error",
        }
    else:
        bullets_text = "\n".join(f"• {b}" for b in result.bullets)
        data = {
            "needs_summary":      bullets_text,
            "has_literacy_goal":  result.has_literacy_goal,
            "has_ell_goal":       result.has_ell_goal,
            "has_attendance_goal": result.has_attendance_goal,
            "has_library_mention": result.has_library_mention,
            "summary_source":     source,
        }

    save_summary(dbn, data)
    return data


def batch_summarize(
    schools: list,
    progress_callback=None,
    delay: float = 0.5,
) -> int:
    """
    Generate summaries for a list of school dicts.
    Each dict must have: dbn, school_name, plus optional data fields.
    progress_callback(i, total, dbn) is called after each school.
    Returns count of summaries generated.
    """
    init_summary_table()
    total = len(schools)
    count = 0

    for i, school in enumerate(schools):
        dbn  = school["dbn"]
        name = school.get("school_name", dbn)
        try:
            summarize_school(dbn, name, school_row=school)
            count += 1
        except Exception as e:
            logger.warning(f"  Summary failed for {dbn}: {e}")
        if progress_callback:
            progress_callback(i + 1, total, name)
        time.sleep(delay)

    return count
