# NYT Factor Pipeline

Theme discovery from New York Times article data for factor investing research.

Discovers evolving news themes from NYT metadata using embeddings + clustering,
then maps those themes to RBICS-style company exposures for quantitative research.

## Architecture

### Why embeddings + clustering instead of LLM-per-article?

Sending every NYT article to an LLM is expensive (~$50–200/day at scale), slow,
and non-reproducible. Instead, this pipeline:

1. **Embeds** each article's metadata (headline, abstract, keywords) using a cheap
   local embedding model (~0 cost, milliseconds per article)
2. **Clusters** the embeddings weekly using HDBSCAN to discover themes automatically
3. **Tracks** themes over time by comparing cluster centroids to known theme centroids
4. Uses the **ChatGPT API only sparingly** to name themes, suggest merges, and infer
   RBICS industry mappings — about 1 LLM call per new theme, not per article

This means the pipeline costs ~$0.01–0.10/day for LLM calls instead of $50+.

### Why both Archive API and Article Search API?

| API | Use Case | Characteristics |
|-----|----------|----------------|
| **Archive API** | Historical backfill (closed months) | 1 request = 1 month of articles, ~4,000 articles/month |
| **Article Search API** | Recent tail (current month) | Paginated, 10 results/request, flexible filtering |

The Archive API is efficient for bulk historical data. The Article Search API fills
the gap between the latest archive month and today.

### Rate limiting, budgeting, and checkpointing

- **Rate limiting**: Configurable requests-per-minute (default: 3 RPM)
- **Daily budget**: Separate budgets for Archive (1500/day) and Article Search (500/day)
- **Checkpointing**: Every completed month/window/page is checkpointed in DuckDB
- **Resume**: On restart, the pipeline skips completed work automatically
- **Backoff**: Exponential backoff with jitter on 429/5xx errors
- **Request log**: Every API call is logged with timing and status

### Normalized article schema

Both APIs produce different JSON structures. The normalization layer extracts:
- `headline_main`, `abstract`, `snippet`, `lead_paragraph`
- `section_name`, `news_desk`, `type_of_material`
- `print_section`, `print_page`, `word_count`
- `keywords` (flattened), `byline`
- A deduplicated `normalized_text` field for embedding

Articles are deduplicated by a stable hash of `uri` or `web_url`.

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone and install
git clone <repo-url> && cd nyt-factor-pipeline
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"

# Or with pip
pip install -e ".[all]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required secrets:**
- `NYT_API_KEY` — Get one at https://developer.nytimes.com/

**Optional secrets:**
- `OPENAI_API_KEY` — Only needed for theme labeling, merge review, and RBICS mapping.
  The entire non-LLM pipeline works without this.

### Initialize database

```bash
nyt-pipeline init-db
```

## Usage

### Historical backfill

```bash
# Estimate cost first
nyt-pipeline estimate-backfill-cost --start 2020-01 --end 2025-12

# Run archive ingestion (resumes automatically)
nyt-pipeline ingest-archive --start 2020-01 --end 2025-12

# Ingest recent tail (after latest archive month)
nyt-pipeline ingest-recent --start-date 2026-01-01 --end-date 2026-03-17

# Score and filter
nyt-pipeline score-articles

# Embed (uses local model by default)
nyt-pipeline embed-articles

# Cluster
nyt-pipeline cluster-window --window weekly --start-date 2020-01-01 --end-date 2026-03-17

# Track themes
nyt-pipeline track-themes

# Build time series
nyt-pipeline build-theme-timeseries
```

### Full backfill (single command)

```bash
nyt-pipeline run-backfill --start 2020-01 --end 2025-12
```

### Incremental daily update

```bash
nyt-pipeline run-incremental-update
```

### Visualize themes

```bash
# Generate an interactive HTML dashboard
nyt-pipeline generate-dashboard

# Open in browser
open data/artifacts/theme_dashboard.html
```

The dashboard shows:
- Theme intensity over time (interactive line chart)
- Stacked article count chart
- Active themes table with labels, article counts, and date ranges
- Recent burst detection highlights
- Top company scores (if company data is loaded)

The dashboard is also auto-generated at the end of `run-backfill` and `run-incremental-update`.

### LLM-powered features (requires OPENAI_API_KEY)

```bash
# Name newly discovered themes
nyt-pipeline label-new-themes

# Review potential theme merges
nyt-pipeline propose-theme-merges

# Map themes to RBICS industries
nyt-pipeline map-themes-rbics
```

### Company scoring

```bash
# Ingest company-RBICS mapping
nyt-pipeline ingest-companies --csv data/sample_companies.csv

# Score companies
nyt-pipeline score-companies --date 2026-03-17
# Or for a range:
nyt-pipeline score-companies --start-date 2026-01-01 --end-date 2026-03-17
```

### Check rate limit state

```bash
nyt-pipeline show-rate-limit-state
```

## Adding company RBICS data

Provide a CSV with these columns:

```csv
company_id,ticker,company_name,rbics_code,rbics_name
AAPL,AAPL,Apple Inc,tech_consumer_electronics,Consumer Electronics
```

A sample file is at `data/sample_companies.csv`. Replace with your actual RBICS mappings.

## OpenAI configuration

OpenAI is used **only** for:
1. Naming a newly emerged theme (~1 call per new theme)
2. Reviewing whether two similar themes should merge (~1 call per candidate pair)
3. Suggesting RBICS industry mappings for a stable theme (~1 call per theme)

All LLM responses are cached by prompt hash. Re-runs do not re-call the API.

The entire embedding, clustering, theme tracking, and time series pipeline
runs without any OpenAI key.

## Output for factor investing

The pipeline produces:

| Table | Description |
|-------|-------------|
| `theme_timeseries` | Daily intensity of each theme (article count, weighted count, z-score) |
| `company_theme_scores` | Daily company-level scores: `intensity * exposure_strength * direction` |
| `theme_rbics_exposure` | Theme-to-industry mappings with direction and strength |
| `company_theme_exposure` | Company-to-theme mappings via RBICS codes |

**Company score formula:**
```
score(company, date) = Σ theme_intensity(date) × exposure_strength(theme, company) × direction_sign
```

Where `direction_sign` is +1 (positive), -1 (negative), or 0 (ambiguous).

You can query total, positive-only, and negative-only scores for portfolio construction.

## Project structure

```
src/nyt_factor_pipeline/
├── cli.py                    # CLI entrypoint (all commands)
├── config.py                 # Typed settings from .env
├── db.py                     # DuckDB schema and connections
├── models.py                 # Pydantic models
├── logging_utils.py          # Structured logging
├── utils/                    # Date, hashing, retry, text utilities
├── ingest/                   # NYT API clients, normalization, checkpoints
│   ├── nyt_client.py         # HTTP client with rate limiting
│   ├── nyt_archive.py        # Archive API ingestion
│   ├── nyt_article_search.py # Article Search API ingestion
│   ├── normalize.py          # Unified article schema
│   ├── checkpoints.py        # Resumable checkpoints
│   ├── request_budget.py     # Rate limiting and daily budgets
│   └── incremental.py        # Resume plan computation
├── scoring/                  # Article importance and filtering
├── embeddings/               # Embedding abstraction (local + OpenAI)
├── clustering/               # HDBSCAN clustering and theme tracking
│   ├── cluster_weekly.py     # Rolling window clustering
│   ├── topic_tracking.py     # Theme creation and linking
│   ├── merge_split.py        # Merge candidate detection
│   ├── keywords.py           # TF-IDF keyword extraction
│   └── representatives.py    # Representative article selection
├── themes/                   # Theme store, timeseries, burst detection
├── llm/                      # Sparse ChatGPT integration
│   ├── openai_client.py      # OpenAI wrapper
│   ├── prompts.py            # All prompt templates
│   ├── cache.py              # Prompt-hash response cache
│   ├── theme_labeling.py     # Name themes via LLM
│   ├── theme_merge_review.py # Merge review via LLM
│   └── rbics_mapping.py      # Industry mapping via LLM
├── viz/                      # HTML dashboard visualization
│   └── dashboard.py          # Self-contained interactive HTML generator
└── exposures/                # RBICS, company ingestion, scoring
    ├── rbics_schema.py       # RBICS schema definition
    ├── company_ingest.py     # CSV ingestion
    ├── theme_to_rbics.py     # Theme-company exposure matching
    └── company_ranking.py    # Daily company-theme scoring
```

## Running tests

```bash
pytest tests/ -v
```

## Limitations and future improvements

- **No full-article text**: Uses only metadata (headline, abstract, snippet, keywords).
  Full article bodies would improve embedding quality but require scraping.
- **Local embeddings**: Default model (all-MiniLM-L6-v2) is fast but small.
  Larger models or OpenAI embeddings may improve clustering quality.
- **HDBSCAN sensitivity**: Clustering results depend on `min_cluster_size` and
  `min_samples` parameters. Tune these for your data volume.
- **Theme tracking EMA**: The centroid update uses a fixed alpha=0.3. Adaptive
  update rates could better handle fast-moving vs. stable themes.
- **RBICS matching**: LLM-suggested industries use free-text names, matched to
  user-provided RBICS via exact code or fuzzy name match. A proper taxonomy
  mapper would improve accuracy.
- **No sentiment**: The pipeline scores theme intensity but not sentiment direction.
  Adding a sentiment layer could distinguish positive from negative coverage.

## License

MIT
