"""Generate a self-contained HTML dashboard visualizing themes through time.

Produces a single HTML file with interactive charts using embedded Chart.js.
No server required — open the file in any browser.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def generate_dashboard(
    conn: duckdb.DuckDBPyConnection,
    output_path: str | Path = "data/artifacts/theme_dashboard.html",
    top_n_themes: int = 25,
    start_date: date | None = None,
    end_date: date | None = None,
) -> Path:
    """Generate an interactive HTML dashboard showing themes through time.

    Args:
        conn: DuckDB connection
        output_path: Where to write the HTML file
        top_n_themes: Max number of themes to display
        start_date: Optional filter
        end_date: Optional filter

    Returns:
        Path to the generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Gather data
    themes_data = _get_themes_summary(conn, top_n_themes)
    timeseries_data = _get_timeseries_data(conn, top_n_themes, start_date, end_date)
    burst_data = _get_recent_bursts(conn)
    stats = _get_pipeline_stats(conn)
    company_data = _get_top_company_scores(conn)

    html = _render_html(themes_data, timeseries_data, burst_data, stats, company_data)
    output_path.write_text(html, encoding="utf-8")
    log.info("dashboard_generated", path=str(output_path))
    return output_path


def _get_themes_summary(conn: duckdb.DuckDBPyConnection, top_n: int) -> list[dict]:
    rows = conn.execute(
        """SELECT t.theme_id, t.current_label, t.description, t.first_seen, t.last_seen,
                  t.active_flag, t.llm_labeled_at,
                  COUNT(DISTINCT ctl.cluster_id) as cluster_count,
                  COALESCE(SUM(ts_agg.total_articles), 0) as total_articles
           FROM themes t
           LEFT JOIN cluster_theme_link ctl ON t.theme_id = ctl.theme_id
           LEFT JOIN (
               SELECT theme_id, SUM(article_count) as total_articles
               FROM theme_timeseries GROUP BY theme_id
           ) ts_agg ON t.theme_id = ts_agg.theme_id
           WHERE t.active_flag = true
           GROUP BY t.theme_id, t.current_label, t.description, t.first_seen,
                    t.last_seen, t.active_flag, t.llm_labeled_at
           ORDER BY total_articles DESC
           LIMIT ?""",
        [top_n],
    ).fetchall()

    return [
        {
            "theme_id": r[0],
            "label": r[1] or "Unlabeled",
            "description": r[2] or "",
            "first_seen": str(r[3]) if r[3] else "",
            "last_seen": str(r[4]) if r[4] else "",
            "active": r[5],
            "llm_labeled": r[6] is not None,
            "cluster_count": r[7],
            "total_articles": r[8],
        }
        for r in rows
    ]


def _get_timeseries_data(
    conn: duckdb.DuckDBPyConnection,
    top_n: int,
    start_date: date | None,
    end_date: date | None,
) -> dict:
    """Get timeseries data organized for Chart.js."""
    # Get top theme IDs
    theme_ids = conn.execute(
        """SELECT theme_id FROM (
               SELECT theme_id, SUM(article_count) as total
               FROM theme_timeseries GROUP BY theme_id ORDER BY total DESC LIMIT ?
           )""",
        [top_n],
    ).fetchall()
    theme_ids = [r[0] for r in theme_ids]

    if not theme_ids:
        return {"dates": [], "series": []}

    conditions = []
    params: list = []
    if start_date:
        conditions.append("date >= ?")
        params.append(str(start_date))
    if end_date:
        conditions.append("date <= ?")
        params.append(str(end_date))

    where = " AND ".join(conditions) if conditions else "1=1"

    # Get all dates
    dates = conn.execute(
        f"SELECT DISTINCT date FROM theme_timeseries WHERE {where} ORDER BY date",
        params,
    ).fetchall()
    date_labels = [str(r[0]) for r in dates]

    # Get series per theme
    series = []
    for tid in theme_ids:
        label_row = conn.execute(
            "SELECT current_label FROM themes WHERE theme_id = ?", [tid]
        ).fetchone()
        label = label_row[0] if label_row else tid[:15]

        ts_params = list(params) + [tid]
        rows = conn.execute(
            f"""SELECT date, intensity, article_count, burst_zscore
                FROM theme_timeseries
                WHERE {where} AND theme_id = ?
                ORDER BY date""",
            ts_params,
        ).fetchall()

        date_map = {str(r[0]): {"intensity": r[1], "count": r[2], "zscore": r[3]} for r in rows}

        intensities = [date_map.get(d, {}).get("intensity", 0) for d in date_labels]
        counts = [date_map.get(d, {}).get("count", 0) for d in date_labels]

        series.append({
            "theme_id": tid,
            "label": label[:40] if label else tid[:15],
            "intensities": intensities,
            "counts": counts,
        })

    return {"dates": date_labels, "series": series}


def _get_recent_bursts(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        """SELECT ts.theme_id, t.current_label, ts.date, ts.intensity, ts.burst_zscore
           FROM theme_timeseries ts
           JOIN themes t ON ts.theme_id = t.theme_id
           WHERE ts.burst_zscore >= 1.5
           ORDER BY ts.date DESC, ts.burst_zscore DESC
           LIMIT 20"""
    ).fetchall()

    return [
        {
            "theme": r[1] or r[0][:15],
            "date": str(r[2]),
            "intensity": round(r[3], 4),
            "zscore": round(r[4], 2),
        }
        for r in rows
    ]


def _get_pipeline_stats(conn: duckdb.DuckDBPyConnection) -> dict:
    def _count(table: str) -> int:
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            return 0

    return {
        "articles": _count("articles"),
        "embedded": _count("article_embeddings"),
        "clusters": _count("clusters_raw"),
        "themes": _count("themes"),
        "active_themes": conn.execute(
            "SELECT COUNT(*) FROM themes WHERE active_flag = true"
        ).fetchone()[0],
        "timeseries_rows": _count("theme_timeseries"),
        "companies": _count("companies"),
        "company_scores": _count("company_theme_scores"),
    }


def _get_top_company_scores(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    try:
        rows = conn.execute(
            """SELECT c.ticker, c.company_name, SUM(cts.score) as total_score,
                      COUNT(DISTINCT cts.theme_id) as theme_count
               FROM company_theme_scores cts
               JOIN companies c ON cts.company_id = c.company_id
               GROUP BY c.ticker, c.company_name
               ORDER BY ABS(SUM(cts.score)) DESC
               LIMIT 20"""
        ).fetchall()
        return [
            {"ticker": r[0], "name": r[1], "score": round(r[2], 4), "themes": r[3]}
            for r in rows
        ]
    except Exception:
        return []


def _render_html(
    themes: list[dict],
    timeseries: dict,
    bursts: list[dict],
    stats: dict,
    companies: list[dict],
) -> str:
    """Render the full HTML dashboard."""

    # Generate color palette
    colors = [
        "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea",
        "#0891b2", "#e11d48", "#65a30d", "#c026d3", "#ea580c",
        "#4f46e5", "#059669", "#d97706", "#7c3aed", "#0d9488",
        "#be123c", "#15803d", "#a16207", "#7e22ce", "#b91c1c",
        "#1d4ed8", "#047857", "#92400e", "#6d28d9", "#0e7490",
    ]

    # Build Chart.js datasets
    datasets_js = []
    for i, s in enumerate(timeseries.get("series", [])):
        color = colors[i % len(colors)]
        datasets_js.append({
            "label": s["label"],
            "data": s["intensities"],
            "borderColor": color,
            "backgroundColor": color + "20",
            "borderWidth": 1.5,
            "pointRadius": 0,
            "fill": False,
            "tension": 0.3,
        })

    # Build count datasets
    count_datasets_js = []
    for i, s in enumerate(timeseries.get("series", [])):
        color = colors[i % len(colors)]
        count_datasets_js.append({
            "label": s["label"],
            "data": s["counts"],
            "backgroundColor": color + "80",
            "borderColor": color,
            "borderWidth": 1,
        })

    dates_json = json.dumps(timeseries.get("dates", []))
    datasets_json = json.dumps(datasets_js)
    count_datasets_json = json.dumps(count_datasets_js)
    themes_json = json.dumps(themes)
    bursts_json = json.dumps(bursts)
    stats_json = json.dumps(stats)
    companies_json = json.dumps(companies)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NYT Theme Discovery Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 20px;
  }}
  .header {{
    text-align: center; margin-bottom: 30px; padding: 20px;
    background: linear-gradient(135deg, #1e293b, #334155); border-radius: 12px;
  }}
  .header h1 {{ font-size: 28px; color: #f8fafc; margin-bottom: 8px; }}
  .header p {{ color: #94a3b8; font-size: 14px; }}
  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }}
  .stat-card {{
    background: #1e293b; border-radius: 8px; padding: 16px; text-align: center;
    border: 1px solid #334155;
  }}
  .stat-card .value {{ font-size: 24px; font-weight: 700; color: #60a5fa; }}
  .stat-card .label {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
  .section {{
    background: #1e293b; border-radius: 12px; padding: 20px;
    margin-bottom: 24px; border: 1px solid #334155;
  }}
  .section h2 {{
    font-size: 18px; color: #f1f5f9; margin-bottom: 16px;
    padding-bottom: 8px; border-bottom: 1px solid #334155;
  }}
  .chart-container {{ position: relative; height: 400px; }}
  .chart-container-sm {{ position: relative; height: 300px; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
  }}
  th, td {{
    padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155;
  }}
  th {{ color: #94a3b8; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
  td {{ color: #e2e8f0; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600;
  }}
  .badge-active {{ background: #065f46; color: #6ee7b7; }}
  .badge-burst {{ background: #7c2d12; color: #fdba74; }}
  .badge-labeled {{ background: #1e3a5f; color: #93c5fd; }}
  .tabs {{
    display: flex; gap: 8px; margin-bottom: 16px;
  }}
  .tab {{
    padding: 8px 16px; border-radius: 6px; cursor: pointer;
    background: #334155; color: #94a3b8; border: none; font-size: 13px;
  }}
  .tab.active {{ background: #2563eb; color: white; }}
  .hidden {{ display: none; }}
  .zscore-high {{ color: #f87171; font-weight: 700; }}
  .zscore-med {{ color: #fbbf24; }}
  .score-pos {{ color: #4ade80; }}
  .score-neg {{ color: #f87171; }}
  .footer {{
    text-align: center; color: #475569; font-size: 12px; margin-top: 24px;
    padding: 12px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>NYT Theme Discovery Dashboard</h1>
  <p>Themes discovered from New York Times article data via embeddings + clustering</p>
</div>

<div class="stats-grid" id="statsGrid"></div>

<div class="section">
  <h2>Theme Intensity Over Time</h2>
  <div class="tabs">
    <button class="tab active" onclick="showChart('intensity')">Intensity</button>
    <button class="tab" onclick="showChart('counts')">Article Counts</button>
  </div>
  <div class="chart-container" id="intensityChartContainer">
    <canvas id="intensityChart"></canvas>
  </div>
  <div class="chart-container hidden" id="countsChartContainer">
    <canvas id="countsChart"></canvas>
  </div>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
  <div class="section">
    <h2>Active Themes</h2>
    <div style="max-height: 500px; overflow-y: auto;">
      <table id="themesTable"><thead><tr>
        <th>Theme</th><th>Articles</th><th>Clusters</th><th>Period</th><th>Status</th>
      </tr></thead><tbody></tbody></table>
    </div>
  </div>

  <div class="section">
    <h2>Recent Bursts</h2>
    <div style="max-height: 500px; overflow-y: auto;">
      <table id="burstsTable"><thead><tr>
        <th>Theme</th><th>Date</th><th>Intensity</th><th>Z-Score</th>
      </tr></thead><tbody></tbody></table>
    </div>
  </div>
</div>

<div class="section" id="companiesSection" style="display:none;">
  <h2>Top Company Theme Scores</h2>
  <table id="companiesTable"><thead><tr>
    <th>Ticker</th><th>Company</th><th>Total Score</th><th>Themes</th>
  </tr></thead><tbody></tbody></table>
</div>

<div class="footer">
  Generated by NYT Factor Pipeline &mdash; Themes discovered without LLM, labeled sparsely with ChatGPT API
</div>

<script>
const dates = {dates_json};
const datasets = {datasets_json};
const countDatasets = {count_datasets_json};
const themes = {themes_json};
const bursts = {bursts_json};
const stats = {stats_json};
const companies = {companies_json};

// Stats grid
const statsGrid = document.getElementById('statsGrid');
const statItems = [
  ['Articles', stats.articles], ['Embedded', stats.embedded],
  ['Clusters', stats.clusters], ['Active Themes', stats.active_themes],
  ['Timeseries Rows', stats.timeseries_rows], ['Companies', stats.companies],
];
statItems.forEach(([label, value]) => {{
  statsGrid.innerHTML += `<div class="stat-card"><div class="value">${{(value || 0).toLocaleString()}}</div><div class="label">${{label}}</div></div>`;
}});

// Intensity chart
const intensityCtx = document.getElementById('intensityChart').getContext('2d');
const intensityChart = new Chart(intensityCtx, {{
  type: 'line',
  data: {{ labels: dates, datasets: datasets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 11 }}, boxWidth: 12 }} }},
      tooltip: {{ backgroundColor: '#1e293b', borderColor: '#475569', borderWidth: 1 }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b', maxTicksLimit: 20 }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }}, title: {{ display: true, text: 'Intensity', color: '#94a3b8' }} }},
    }},
  }}
}});

// Counts chart
const countsCtx = document.getElementById('countsChart').getContext('2d');
const countsChart = new Chart(countsCtx, {{
  type: 'bar',
  data: {{ labels: dates, datasets: countDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 11 }}, boxWidth: 12 }} }},
    }},
    scales: {{
      x: {{ stacked: true, ticks: {{ color: '#64748b', maxTicksLimit: 20 }}, grid: {{ color: '#1e293b' }} }},
      y: {{ stacked: true, ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }}, title: {{ display: true, text: 'Article Count', color: '#94a3b8' }} }},
    }},
  }}
}});

// Tab switching
function showChart(type) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('intensityChartContainer').classList.toggle('hidden', type !== 'intensity');
  document.getElementById('countsChartContainer').classList.toggle('hidden', type !== 'counts');
}}

// Themes table
const themesBody = document.querySelector('#themesTable tbody');
themes.forEach(t => {{
  const status = t.llm_labeled
    ? '<span class="badge badge-labeled">LLM Labeled</span>'
    : '<span class="badge badge-active">Auto</span>';
  themesBody.innerHTML += `<tr>
    <td><strong>${{t.label}}</strong><br><small style="color:#64748b">${{t.description.substring(0, 80)}}</small></td>
    <td>${{t.total_articles}}</td><td>${{t.cluster_count}}</td>
    <td><small>${{t.first_seen}} to ${{t.last_seen}}</small></td><td>${{status}}</td>
  </tr>`;
}});

// Bursts table
const burstsBody = document.querySelector('#burstsTable tbody');
bursts.forEach(b => {{
  const zclass = b.zscore >= 3 ? 'zscore-high' : b.zscore >= 2 ? 'zscore-med' : '';
  burstsBody.innerHTML += `<tr>
    <td>${{b.theme}}</td><td>${{b.date}}</td><td>${{b.intensity.toFixed(4)}}</td>
    <td class="${{zclass}}">${{b.zscore.toFixed(2)}}</td>
  </tr>`;
}});

// Companies table
if (companies.length > 0) {{
  document.getElementById('companiesSection').style.display = 'block';
  const compBody = document.querySelector('#companiesTable tbody');
  companies.forEach(c => {{
    const scoreClass = c.score >= 0 ? 'score-pos' : 'score-neg';
    compBody.innerHTML += `<tr>
      <td><strong>${{c.ticker}}</strong></td><td>${{c.name}}</td>
      <td class="${{scoreClass}}">${{c.score.toFixed(4)}}</td><td>${{c.themes}}</td>
    </tr>`;
  }});
}}
</script>
</body>
</html>"""
