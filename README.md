# MultiHop Corrective RAG (CRAG) — Web App

A **Retrieval-Augmented Generation** web application built around a **Corrective RAG (CRAG) + Multi-Hop + Reranking** pipeline. Upload your own documents (PDF, text, or URL) and chat with them — every answer is retrieved with dense + sparse (BM25) search, self-graded, corrected when the context is weak, reranked, and evaluated for faithfulness and answer relevancy.

Built as a research project on the **MultiHop-RAG** benchmark; ships with a full web UI.

---

## Table of Contents

- [How it works](#how-it-works)
- [Quick Start (Docker)](#quick-start-docker--recommended)
- [What happens on first startup](#what-happens-on-first-startup)
- [Using the app](#using-the-app)
- [Configuration reference](#configuration-reference)
- [Indexing the base dataset (optional)](#indexing-the-base-dataset-optional)
- [Testing](#testing)
- [Local development (without Docker)](#local-development-without-docker)
- [Deploying to a server](#deploying-to-a-server)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Research background](#research-background)

---

## How it works

```text
User Query
    │
    ▼
MultiHop Orchestrator          — decomposes multi-step questions (max 3 hops)
    │
    ▼
CRAG Wrapper                   — self-grades retrieval, refines ambiguous
    │                            chunks, falls back to external search
    ▼
Core Retriever
 ┌───────────────┬───────────────┐
 │ Dense         │ Sparse BM25   │
 │ (embeddings)  │ (keyword)     │
 └───────────────┴───────────────┘
            │
            ▼
      Merge + Dedup
            │
            ▼
   Reranker (jina-reranker-v3, local)
            │
            ▼
        Generator LLM (via OpenRouter)
            │
            ▼
   Answer + Faithfulness / Relevancy scores
```

**Models used by default**

| Role | Model | Runs |
|------|-------|------|
| Embeddings | `google/gemini-embedding-2-preview` | via OpenRouter API |
| Generator LLM | `qwen/qwen3-30b-a3b-instruct-2507` | via OpenRouter API |
| Reranker | `jinaai/jina-reranker-v3` | **locally** (downloaded, CPU/GPU) |
| CRAG evaluator | `intfloat/multilingual-e5-small` | **locally** (downloaded, CPU/GPU) |

---

## Quick Start (Docker — Recommended)

All services (backend, Celery worker, frontend, PostgreSQL, Redis, ChromaDB) start with one command.

### Prerequisites

- **Docker** and **Docker Compose v2**
- An **OpenRouter API key** (<https://openrouter.ai/keys>) — required for embeddings and answer generation
- ~5 GB free disk (images + ~2 GB of locally cached models)

### One-command deploy

```bash
git clone <repository-url> && cd <repository-name>
./deploy.sh
```

`deploy.sh` checks prerequisites, creates the `.env` files from their examples if missing (edit `backend/.env` to add your `OPENROUTER_API_KEY`), builds the images, **runs the test suite**, starts the stack, and waits until the backend reports healthy. Use `./deploy.sh --skip-tests` to skip the test gate.

Prefer to do it step by step? Follow below.

### Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create the backend environment file**

   ```bash
   cp backend/.env.example backend/.env
   ```

   Edit `backend/.env` and set at least:

   ```env
   OPENROUTER_API_KEY=sk-or-...
   SECRET_KEY=any-long-random-string
   ```

3. **Create the frontend environment file**

   ```bash
   cp frontend/.env.example frontend/.env
   ```

   The defaults work out of the box for Docker.

4. **(Optional but recommended on slow/unreliable connections) Pre-download the models**

   The reranker + evaluator models (~2 GB) are otherwise downloaded on first startup. To download them ahead of time from your host machine:

   ```bash
   pip install huggingface-hub PyYAML
   cd backend && python dl_reranker_model.py && cd ..
   ```

   The script is **resumable and safe to re-run** — interrupted downloads continue where they left off, and finished models are skipped. Models land in `backend/models/`, which is mounted into the containers.

5. **Start everything**

   ```bash
   docker compose up -d --build
   ```

   The first build takes a few minutes. Watch startup progress with:

   ```bash
   docker compose logs -f backend
   ```

6. **Open the app** at **<http://localhost:5151>**.

### Useful commands

| Command | Description |
|---------|-------------|
| `docker compose up -d --build` | Build and start all services |
| `docker compose logs -f backend` | Follow backend logs (startup, downloads) |
| `docker compose logs -f worker` | Follow Celery worker logs (document indexing) |
| `docker compose ps` | Show service status + health |
| `docker compose restart backend worker` | Restart app services |
| `docker compose down` | Stop all services (data is kept) |
| `docker compose down -v` | Stop and **delete all data** (DB, ChromaDB) |

### Services and ports

| Service | Container | Host port | Purpose |
|---------|-----------|-----------|---------|
| Frontend (Nginx) | `rag_frontend` | `5151` | Web UI, proxies `/api` and `/ws` to backend |
| Backend (Daphne) | `rag_backend` | `8051` | Django + Channels API |
| Celery worker | `rag_worker` | — | Document indexing jobs |
| ChromaDB | `rag_chromadb` | `8002` | Vector store |
| PostgreSQL | `rag_postgres` | `5432` | App database |
| Redis | `rag_redis` | `6351` | Queue, cache, websocket layer |

---

## What happens on first startup

The **backend container owns all initialization** (the worker waits until the backend is healthy, so nothing runs twice):

1. Waits for PostgreSQL and ChromaDB to accept connections.
2. Collects static files and applies database migrations.
3. Ensures the base ChromaDB collection exists and the MultiHop-RAG corpus file is cached (`backend/corpus/corpus.json` — already bundled in the repo, so normally no download happens).
4. With `PRELOAD_MODELS=true` (the default), downloads the local models selected in `backend/config.yml` into `backend/models/` **unless already present**. Defaults:
   - `jinaai/jina-reranker-v3` (~1.2 GB)
   - `intfloat/multilingual-e5-small` (~0.5 GB)

   Downloads retry automatically with backoff and **resume after interruption**. A model is only treated as complete when its `.download_complete` marker exists, so a half-finished download is never mistaken for a working model.
5. Starts the web server. Only then does the healthcheck pass and the worker start.

NLTK data (`punkt`, `punkt_tab`, `stopwords`) is baked into the Docker image at build time — no runtime download needed.

If a download fails permanently, the app still starts (with reduced functionality) and you can retry any time:

```bash
docker compose exec backend python dl_reranker_model.py
docker compose exec backend python insert_base_dataset.py
```

---

## Using the app

1. Open **<http://localhost:5151>**. `/` is the public landing page; **Get Started** takes you to `/login` (guest login — just pick a username and email), which lands you in the chat at `/chat`.
2. **Add a document**: upload a PDF, UTF-8 TXT or Markdown file (up to 20 MB), paste text, or submit a URL. PDFs must contain selectable text; scanned PDFs need OCR first.
3. Wait for **Ready**. The chat shows upload/indexing progress and the document list refreshes automatically. If indexing fails, it shows the reason; submitting the same document again retries it.
4. **Chat**: ask questions about your document. The pipeline streams status updates (retrieval → grading → reranking → generation → evaluation) over a websocket, then shows the answer with its supporting chunks and faithfulness / answer-relevancy scores.

### Configuring the pipeline per query

By default, questions search **My docs**, with **External search off**. Uploading a document selects My docs. Enable external search explicitly to allow Wikipedia/news, or choose Base corpus to search the benchmark. A pending upload pauses document questions; an unavailable corpus never silently switches to another source.

The **Pipeline** panel in the chat sidebar controls how the next query is answered. The choice is saved in the browser and sent with every question as a `CONFIG` object. It is grouped into collapsible sections; the main stages are enabled by default while answers stay grounded in your documents.

**Stages** — each one is a genuine composition change, not a flag the pipeline ignores:

| Control | Options | Effect |
|---------|---------|--------|
| **Multi-hop** | on/off + max hops 1–3 | Decompose the question into follow-up lookups, and the hop ceiling. 3 is the ceiling as well as the usual default: each hop is an LLM decision call plus a retrieval plus, with corrective on, a grading pass. |
| **Corrective (CRAG)** | on/off | Self-grade retrieved context and correct it when weak. |
| **Reranker** | on/off | Reorder merged candidates with the cross-encoder. On/off only — it runs from a model downloaded to the server, not through OpenRouter. |
| **Answer scoring** | on/off | Rate the answer with the RAGAS judge. Off saves two LLM-judge passes and an embedding call per question. |

**Models**:

| Control | Options | Effect |
|---------|---------|--------|
| **Chat model** | any OpenRouter chat model | Answer generation, multi-hop decisions and query expansion. Left as *Server default*, each stage keeps the model configured in `backend/config.yml`. |
| **Embedding model** | any OpenRouter embedding model | Embeds **newly indexed documents** — see the caveat below. |
| **Temperature** | 0.0–2.0 | How freely the *answer* is worded. The pipeline's own decision calls (which documents to fetch) stay deterministic either way — sampling those would change retrieval, not phrasing. |

**Retrieval**:

| Control | Options | Effect |
|---------|---------|--------|
| **Corpus** | Auto / My docs / Base corpus | Which ChromaDB collection to search. `My docs` is the default. `Auto` searches your documents when present, otherwise the shared corpus. `My docs` refuses to silently fall back — if you have no documents it says so. |
| **Retrievers** | Both / Dense / BM25 | Run both base retrievers, or only embedding similarity, or only keyword search. |
| **Chunks retrieved** | 1–20 | How many chunks each retriever fetches before merging. |
| **Chunks kept** | 1–20 | How many survive the merge/rerank — what the model actually reads. |
| **Web passages kept** | 1–20 | How many passages the corrective stage keeps from Wikipedia / news. Separate from the two above: these are fetched live and scored in memory, not read from the index. |
| **Drop stop words** | on/off | Strip English stop words before BM25 tokenisation. Changing it rebuilds the BM25 index, because the query is tokenised the same way the corpus was. |

**Corrective detail** (ignored when the Corrective stage is off):

| Control | Options | Effect |
|---------|---------|--------|
| **Strictness** | Default / Lenient / Balanced / Strict | How readily a chunk is accepted rather than escalated to the web. A preset rather than three raw thresholds — see below. |
| **External search** | on/off | Whether weak local context may escalate to the web at all. |
| **Wikipedia** / **News** | on/off each | The two external sources, individually. News needs a NewsAPI key. |
| **Query expansion** | on/off + 1–5 phrasings | LLM-rewritten keywords and reformulations. Off also removes the ambiguous-resolution retries, which are built on them. |
| **Knowledge refinement** | on/off | CRAG's strip-level re-scoring of a borderline chunk. Off, an ambiguous chunk stands exactly as retrieved — neither promoted nor discarded. |

**Indexing** (applies to documents uploaded from now on; existing chunks are already split):

| Control | Options | Effect |
|---------|---------|--------|
| **Chunking** | Default / Recursive / Paragraph / Fixed / Semantic | How uploaded documents are split. `Semantic` splits where the topic shifts and costs an embedding call per upload; with no OpenRouter key available it fails the upload rather than quietly falling back to another splitter, because a stored document gives no sign of how it was split. |
| **Chunk size** | 100–4000 chars | Target characters per chunk. |
| **Chunk overlap** | 0–1000 chars | Characters repeated between neighbouring chunks. Corrected server-side if it would not leave room to overlap. |

**Scoring detail** (ignored when Answer scoring is off):

| Control | Options | Effect |
|---------|---------|--------|
| **Answer relevancy** / **Faithfulness** | on/off each | The two RAGAS metrics. Each is its own judge pass, so switching one off is a real saving. |
| **Judge model** / **Judge embedding model** | any OpenRouter model | Left on the default, the judge is a *different* model from the one answering — a model scoring its own output flatters itself, which is why the judge is configured separately. |

Two sentinels run through all of this: `null` for a number and `""` for a model id or named strategy both mean **defer to whatever the deployment configured**, deliberately not "use the value written in the client". Giving them concrete defaults would override every deployment's own tuning in `backend/config.yml` — a panel nobody touched would silently retune the server.

Every one of these is resolved per request from a `contextvars.ContextVar`, never written onto the pipeline. The pipeline is a single process-wide instance shared by every concurrent query, so a setting stored on it is a setting two users share: whoever wrote last wins for both.

#### Why strictness is a preset, not three thresholds

The corrective grader normalises cosine similarity from `[-100, 100]` into `[0, 1]`, so real relevance scores cluster in roughly 0.5–0.95 and the useful band between "accept this chunk" and "escalate to the web" is only a few hundredths wide. Its decision logic also requires `upper > lower`: set them the other way round and every chunk grades "incorrect" and every query hits external search. Three sliders in a 0.04-wide band can express that pipeline; a preset cannot. `Balanced` reproduces the thresholds the app has always shipped.

#### What is deliberately not configurable

Two models run from snapshots downloaded to the server rather than through OpenRouter, so they are on/off only and cannot be swapped per query: the **reranker** (`jinaai/jina-reranker-v3`) and the **corrective grader** (`intfloat/multilingual-e5-small`). Switching either per request would mean a multi-gigabyte download mid-query. The grader's *strictness* is adjustable; the grader itself is not.

The model lists come from OpenRouter's two public catalogs, proxied and cached by `GET /api/v1/models/`. They are separate endpoints for a reason: embedding models are **not** in OpenRouter's main `/models` catalog at all, they live behind `/embeddings/models`. Both pickers are searchable and also accept a model id typed by hand, so a model released after the cached list still works.

With corrective off, for instance, multi-hop wraps the base retriever directly rather than the graded one. Server-side, `backend/common/runtime/config.py` validates and clamps whatever arrives, so **an absent or malformed `CONFIG` uses the document-only defaults**.

#### Why the embedding model is not a per-query choice

A ChromaDB collection holds one vector space. Every vector in it was produced by one embedding model, and embedding a query with a different model produces a vector of the wrong dimension — ChromaDB rejects it outright, or (same width, different space) it answers with confident nonsense.

So the embedding model is chosen **when a document is indexed**, recorded on the collection, and **enforced at query time**:

- The first document you upload fixes the model for your whole collection. Later uploads reuse it.
- Dense retrieval always embeds your question with the model the searched corpus was built with. If that differs from your pick, the answer carries a notice saying which model was actually used and why — the setting is overridden out loud, never silently.
- To switch, delete all your documents (which releases the collection) and re-upload them. The settings panel shows the picker as locked, with that explanation, whenever there is something indexed.
- The shared base corpus is always locked: it is not yours to re-index.

`GET /api/v1/corpus/<username>/` reports this state so the panel can explain a locked picker up front rather than after the query has run.

### Using your own API keys

The **API keys** panel in the chat sidebar accepts your own credentials:

| Key | What it does | Without it |
|-----|--------------|------------|
| **OpenRouter** | Every model call — generation, hop decisions, embeddings, indexing and the RAGAS judge | The server's `OPENROUTER_API_KEY` is used |
| **NewsAPI** | Corrective retrieval's recent-news lookup | That one lookup is skipped; Wikipedia still runs |

Keys are stored in your browser only, sent with each request as a top-level `KEYS` field, used for that request, and **never written to the database**. They travel separately from `CONFIG` deliberately: `CONFIG` is logged verbatim by the server, while `KEYS` is redacted before any log line and scrubbed out of any error message returned to the browser (`backend/common/runtime/api_keys.py`).

A key you supply is used for indexing too, so bringing your own means you pay for your own embeddings rather than the deployment's. Because indexing runs in a Celery worker, the credentials are handed over through a short-lived Redis entry keyed by a random token rather than as task arguments — task arguments are persisted in the broker and echoed into Celery's logs on failure (`backend/common/runtime/handoff.py`).

If the settings handoff is unavailable or expires, indexing fails with a retry message instead of silently changing the model, chunking settings, or billing key.

This also means the server does not strictly need its own `OPENROUTER_API_KEY`: with none configured, the app still starts and every user brings their own.

### When a stage fails

Each stage degrades on its own rather than failing the query. If dense retrieval dies the answer still comes from BM25; if the reranker is unavailable the merged chunks are used in retrieval order; if answer generation fails the retrieved **sources are still returned**. Every response carries a `degraded` list naming the stages that fell back (empty on a healthy run), and when nothing at all could be retrieved the app says so instead of letting the model guess.

If you have no uploaded documents, the default query asks you to upload one. Choose Auto or Base corpus explicitly to use the shared collection (see [Indexing the base dataset](#indexing-the-base-dataset-optional)).

### Frontend routes

| Route | Page | Indexed |
|-------|------|---------|
| `/` | Landing page — what the pipeline does, models, self-hosting, FAQ | yes |
| `/docs` | Step-by-step walkthrough with screenshots | yes |
| `/about` | Research background, evaluation metrics, scope of the deployment | yes |
| `/chat` | The app (redirects to `/login` without a session) | no |
| `/login` | Guest sign-in | no |
| anything else | 404 page | no |

Public-page metadata lives in `frontend/index.html` (static tags for social crawlers, which don't run JS) and `frontend/src/lib/seo.ts` (per-route title/description/canonical). `frontend/public/robots.txt` and `frontend/public/sitemap.xml` reference `https://crag.nevatal.tech` — change the domain there if you deploy elsewhere.

---

## Configuration reference

Connection settings and API keys live in `backend/.env` (see `backend/.env.example`). Pipeline defaults are loaded from **`backend/config.yml`**: generator and judge models, retrieval budgets, hop limit, corrective thresholds, local grader/reranker models, and chunking. Docker mounts this file into both backend and worker; restart both after editing. `RAG_CONFIG_FILE` can select a different YAML file. The chat sidebar overrides supported settings per request.

Local development uses the Vite proxy: set `DEV_BACKEND_URL=http://127.0.0.1:8000` in `frontend/.env`. `VITE_API_URL` sets the API base and `VITE_WS_URL` optionally sets a separate WebSocket origin. Rebuild the frontend after changing `VITE_` values in Docker.

Environment reference:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | Embeddings + LLM generation via OpenRouter. The deployment-wide default; users can supply their own in the chat sidebar instead, so this is only required if you want the app usable without one |
| `SECRET_KEY` | — | Django secret key — set to a long random string |
| `DEBUG` | `False` | Django debug mode |
| `DEVELOPMENT_MODE` | `False` | `true` = use SQLite instead of `DATABASE_URL` |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1,localhost,backend` | Comma-separated allowed hosts — add your domain when deploying |
| `DATABASE_URL` | `postgresql://myuser:changeme@postgres:5432/crag_db` | PostgreSQL connection string |
| `REDIS_URL` / `REDIS_HOST` / `REDIS_PORT` | `redis` service | Redis connection |
| `CHROMA_HOST` / `CHROMA_PORT` | `chromadb` / `8000` | ChromaDB connection |
| `INSERT_BASE_DATASET` | `false` | Embed + index the MultiHop-RAG corpus on startup (costs API credits) |
| `OPENAI_API_KEY` | — | Optional, only for OpenAI-direct calls |
| `LANGSMITH_*` | disabled | Optional LangSmith tracing |
| `NEWS_API_KEY` | — | Optional, for corrective retrieval's news lookup. Users can supply their own in the chat sidebar |

Extra knobs (rarely needed): `MODEL_CACHE_DIR` (model download dir, default `/app/models`), `MODEL_DOWNLOAD_MAX_ATTEMPTS` (default 5), `PRELOAD_MODELS` (download configured local models before startup, default `true`), `CHROMA_CONNECT_ATTEMPTS` (default 30). With `PRELOAD_MODELS=false`, uncached local models load when their stages are first used; disable those stages in the sidebar if they are not needed.

`CELERY_WORKER_CONCURRENCY` defaults to `2` to limit memory used by indexing workers. Increase it when the server has capacity for more simultaneous uploads.

Frontend settings in `frontend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `/api` | API base path (proxied by Nginx) |
| `VITE_API_VERSION` | `v1` | API version segment |

---

## Indexing the base dataset (optional)

The app ships with the **MultiHop-RAG** news corpus (`backend/corpus/corpus.json`, ~600 articles). By default it is **not** embedded into ChromaDB, because embedding it consumes OpenRouter API credits. Users who only chat with their own uploaded documents don't need it.

To enable it, set in `backend/.env`:

```env
INSERT_BASE_DATASET=true
OPENROUTER_API_KEY=sk-or-...
```

then restart (`docker compose restart backend`) or run it manually:

```bash
docker compose exec backend sh -c "INSERT_BASE_DATASET=true python insert_base_dataset.py"
```

The script is idempotent: it skips indexing when the collection is already populated, validates the corpus file, and refuses to insert if any embedding batch failed (so the index can never end up silently misaligned).

---

## Testing

Frontend connection regression checks run with `cd frontend && npm test`; `npm run build` verifies TypeScript and the production bundle.

Tests live **next to the code they cover**, and every one of them runs with no network, no API key and no PostgreSQL/Redis/ChromaDB server — all external calls are mocked.

| File | Covers |
|------|--------|
| `test_import_smoke.py` | Imports **every** backend module, resolves the URL conf, checks each websocket route points at a real consumer, and runs `manage.py check`. Deliberately has no skip guards: if the backend is broken, this fails loudly. |
| `pipeline/tests.py` | `AppRAGPipeline` — chunk normalisation, collection/corpus resolution, chain selection per config, and the full **degradation matrix** (each stage failing in turn). |
| `multi_hop/tests.py` | Hop loop, bridge decisions, dedup keys, early stopping, final re-rank fallbacks. |
| `corrective/tests.py` | correct / ambiguous / incorrect decision branches, external-search isolation, local-vs-external scoring. |
| `dense_rag/tests.py` | Embedding batching and partial failures, `where_filter` fallback, `n_results` clamping. |
| `sparse_rag/tests.py` | Tokenisation and stop words, BM25 ranking, lazy index loading. |
| `hybrid_rag/tests.py`, `hybrid_rag/test_merge.py` | Reranker fallback; merge/dedup/truncate and the rerank-disabled path. |
| `router/tests.py` | Every REST endpoint through the Django test client, including file-hash dedup on re-upload. |
| `router/test_document_flow.py` | Real file storage, PDF/text parsing, database and in-memory Chroma; 48 stage/retriever combinations, upload retries, corpus isolation, chunking/model pinning, deletion and concurrent request state. Remote model calls are fixtures. |
| `router/test_tasks.py` | `build_index_task` status transitions, duplicate guards, retry on failure. |
| `router/test_consumers.py` | The query websocket: the sidebar's `CONFIG` arriving at the pipeline intact. |
| `common/tests.py`, `common/test_pipeline_config.py`, `common/test_chunker.py` | NLTK bootstrap, local model paths, config normalisation/clamping, chunking strategies. |
| `ai_handler/tests.py`, `rag/tests.py`, `test_scripts.py` | LLM retry behaviour, engine-registry recovery, the resumable model downloader and corpus validation. |

**Run in Docker (no other services required):**

```bash
docker compose build backend
docker compose run --rm --no-deps --entrypoint "" \
  -e DEVELOPMENT_MODE=true -e OPENROUTER_API_KEY= \
  backend python manage.py test --verbosity 2
```

**Run inside an already-running stack:**

```bash
docker compose exec backend python manage.py test
```

**Run locally** (with the backend virtualenv from [Local development](#local-development-without-docker)):

```bash
cd backend
DEVELOPMENT_MODE=true python manage.py test
```

`./deploy.sh` runs this suite automatically before starting the stack and aborts the deployment if anything fails.

> **Note:** every backend package needs an `__init__.py` for its tests to be collected. Python 3.11 dropped namespace-package support from `unittest` discovery, so a test file in a directory without one is silently never run.

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

| Job | Steps |
|-----|-------|
| **Backend tests** | Python 3.14, CPU-only torch + `requirements.txt` (same two-step install as the Dockerfile), NLTK data, `manage.py check`, `makemigrations --check` (fails if a model change has no migration), then the full test suite. |
| **Frontend** | Node 20, `npm ci`, `npm run build` (which is `tsc -b && vite build`, so it typechecks too). Lint runs but is non-blocking while pre-existing lint errors are cleaned up. |

The backend job deliberately points `CHROMA_HOST`/`REDIS_HOST` at non-existent hosts, so a test that forgets to mock a service fails fast instead of hanging.

---

## Local development (without Docker)

### Prerequisites

- Python **3.14+**, Node.js + npm
- Running **Redis** server
- Running **ChromaDB** server (`pip install chromadb && chroma run --port 8000`)
- PostgreSQL — *or* set `DEVELOPMENT_MODE=true` to use SQLite

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` for local use:

```env
DEVELOPMENT_MODE=true          # SQLite, no PostgreSQL needed
CHROMA_HOST=localhost
CHROMA_PORT=8000
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379/0
OPENROUTER_API_KEY=sk-or-...
```

Then:

```bash
python dl_reranker_model.py     # download local models (resumable)
python manage.py migrate
python manage.py runserver
```

> NLTK data is downloaded automatically on first use if missing — no manual step needed.

### Celery worker (second terminal)

```bash
cd backend
source venv/bin/activate
celery -A ragreader worker --loglevel=info
```

### Frontend (third terminal)

```bash
cd frontend
npm install
npm run dev
```

Open **<http://localhost:5173>**.

---

## Deploying to a server

The Docker Compose setup runs as-is on any VPS with Docker installed — `./deploy.sh` performs the whole sequence (build → test → start → health wait):

1. Clone the repo on the server and create `backend/.env` / `frontend/.env` as in the Quick Start (or let `./deploy.sh` create them, then edit and re-run).
2. In `backend/.env`, set production values:
   - a strong `SECRET_KEY` and `POSTGRES_PASSWORD`/`DATABASE_URL` password (also change it in `docker-compose.yml`)
   - `DJANGO_ALLOWED_HOSTS=yourdomain.com,backend`
   - `DEBUG=False`
3. `docker compose up -d --build`
4. Point a reverse proxy (Caddy, Nginx, Traefik) at the frontend, e.g. with Caddy:

   ```
   yourdomain.com {
       reverse_proxy localhost:5151
   }
   ```

   The frontend's internal Nginx already proxies `/api/` and `/ws/` (websockets included) to the backend, so exposing port `5151` is enough. You can remove the other port mappings from `docker-compose.yml` if you don't need direct access to PostgreSQL/Redis/ChromaDB from the host.
5. Persistence: PostgreSQL and ChromaDB data live in named Docker volumes; downloaded models live in `./backend/models` on the host. Back these up if the data matters.

**Resource guidance:** the reranker runs on CPU inside the container — 4 GB RAM is a workable minimum, 8 GB recommended. A GPU is used automatically if available to the container but is not required.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| First start is slow, frontend can't reach backend yet | Models (~2 GB) are downloading. Watch `docker compose logs -f backend`. The backend only reports healthy after init finishes. |
| `Model download failed` in logs | Network issue. Downloads auto-retry with backoff; if they still fail, re-run `docker compose exec backend python dl_reranker_model.py` — it resumes where it stopped. Or pre-download on the host (Quick Start step 4). |
| Worker has not started yet | Compose waits for the backend health check before starting the worker. Check backend initialization with `docker compose logs -f backend`; the worker does not wait separately for optional model files. |
| `Could not reach ChromaDB` | ChromaDB is still starting; the backend retries for ~60 s. If it persists: `docker compose logs chromadb` and check port `8002` isn't already taken on the host. |
| Answers say `OpenRouter Error (...)` | Missing/invalid `OPENROUTER_API_KEY`, or no credits. Calls retry 3× before giving up. |
| `DATABASE_URL is required when DEVELOPMENT_MODE is not 'true'` | Set `DATABASE_URL` in `backend/.env`, or use `DEVELOPMENT_MODE=true` (SQLite) for local dev. |
| Port already in use | Change the **host** side of the port mapping in `docker-compose.yml` (e.g. `"5152:5176"`). |
| Websocket disconnects / no streaming updates | Make sure you access the app through the frontend port (`5151`) — its Nginx config proxies `/ws/`. If behind your own proxy, it must forward websocket upgrade headers. |
| Want a completely fresh start | `docker compose down -v && docker compose up -d --build` (deletes DB + vector data; downloaded models in `backend/models/` are kept). |
| Corrupted model suspicion | Delete the model's folder in `backend/models/` (e.g. `backend/models/jinaai--jina-reranker-v3/`) and re-run the download script. |

---

## Project structure

```text
.
├── docker-compose.yml          # Full stack: backend, worker, frontend, postgres, redis, chromadb
├── deploy.sh                   # Build → test → start → wait-for-healthy
├── backend/
│   ├── Dockerfile              # Multi-stage build; NLTK data baked in
│   ├── entrypoint.sh           # Waits for deps, migrates, prepares dataset + models
│   ├── dl_reranker_model.py    # Resumable model downloader (safe to re-run)
│   ├── insert_base_dataset.py  # Base corpus prep / optional indexing (safe to re-run)
│   ├── test_scripts.py         # Tests for the two startup scripts above
│   ├── corpus/corpus.json      # Bundled MultiHop-RAG news corpus
│   ├── ragreader/              # Django project (settings, ASGI, Celery)
│   ├── router/                 # Main app: API views, websocket consumers, Celery tasks, models
│   ├── pipeline/               # Pipeline orchestration (AppRAGPipeline)
│   ├── rag/                    # Engine registry (tests.py inside)
│   ├── dense_rag/              # Dense retrieval (embeddings via OpenRouter)
│   ├── sparse_rag/             # BM25 retrieval (NLTK tokenization)
│   ├── hybrid_rag/             # Merge + dedup + local reranker (tests.py inside)
│   ├── corrective/             # CRAG evaluator, query expansion, external search
│   ├── multi_hop/              # Multi-hop retrieval orchestration
│   ├── evaluation/             # RAGAs-based generation evaluation
│   ├── chroma/                 # ChromaDB helpers
│   ├── ai_handler/             # LLM clients (OpenRouter/OpenAI) with retry, shared client
│   │                           #   factory, and the OpenRouter model catalog proxy
│   └── common/                 # Chunker, prompts, NLTK setup, schema, helpers
│       └── runtime/            # Everything one request chose, and the keys it brought.
│                               #   The pipeline is a single shared instance, so none of
│                               #   this may be stored on it:
│                               #     config      — the CONFIG the browser sends, validated
│                               #     context     — the ContextVar the stages resolve against
│                               #     api_keys    — BYOK normalisation, redaction, scrubbing
│                               #     handoff     — getting both to the Celery worker without
│                               #                   putting credentials in the broker
│                               #     log_filters — keys out of every log record
│                               #     errors      — a failed request vs. a failed run
└── frontend/                   # React + Vite + Tailwind UI, served by Nginx
    └── src/components/settings/  # The pipeline panel, model pickers and key inputs
```

---

## Research background

This repository implements and evaluates an integrated RAG pipeline that combines three layers:

1. **Core Retriever** — dense (embedding) retrieval, sparse (BM25) retrieval, and hybrid retrieval with reranking.
2. **Corrective RAG (CRAG)** — self-grades retrieved chunks (correct / ambiguous / incorrect), refines ambiguous chunks via decompose-then-recompose, and escalates to fallback/external retrieval when context is judged insufficient.
3. **Multi-Hop Orchestrator** — decomposes complex questions requiring reasoning across multiple documents into sequential retrieval hops.

The pipeline is evaluated on the **MultiHop-RAG** benchmark (Tang et al.) with:

- **Retrieval metrics:** Hit Rate, Recall@k, Precision@k, MRR, MAP
- **Generation metrics:** BERTScore, RAGAs (Faithfulness, Answer Relevancy, Answer Correctness)

The web app surfaces two of these metrics (Faithfulness, Answer Relevancy) live for every answer.

**References**

- Yan et al. — *Corrective Retrieval-Augmented Generation*
- Tang et al. — *MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries*
- RAGAs evaluation framework; BERTScore

---

## License

Developed for academic and research purposes. See [LICENSE](LICENSE).
