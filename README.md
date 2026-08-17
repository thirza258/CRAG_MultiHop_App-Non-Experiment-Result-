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
   pip install huggingface-hub
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
4. Downloads the two local models into `backend/models/` **unless already present**:
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
2. **Add a document**: upload a PDF, paste text, or submit a URL.
3. Wait for indexing to finish (the worker chunks the document, embeds it, and stores it in ChromaDB).
4. **Chat**: ask questions about your document. The pipeline streams status updates (retrieval → grading → reranking → generation → evaluation) over a websocket, then shows the answer with its supporting chunks and faithfulness / answer-relevancy scores.

If you haven't uploaded any documents, queries fall back to the shared base collection (see [Indexing the base dataset](#indexing-the-base-dataset-optional)).

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

All backend settings live in `backend/.env` (see `backend/.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | **Required.** Embeddings + LLM generation via OpenRouter |
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
| `NEWS_API_KEY` | — | Optional, for external search fallback |

Extra knobs (rarely needed): `MODEL_CACHE_DIR` (model download dir, default `/app/models`), `MODEL_DOWNLOAD_MAX_ATTEMPTS` (default 5), `MODEL_WAIT_TIMEOUT` (worker wait for models, default 900 s), `CHROMA_CONNECT_ATTEMPTS` (default 30).

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

Tests live **next to the code they cover**: `common/tests.py`, `ai_handler/tests.py`, `hybrid_rag/tests.py`, `rag/tests.py`, and `test_scripts.py` (for the root-level startup scripts). They cover the resumable model downloader, corpus download/validation, NLTK bootstrap, local model path resolution, LLM retry behavior, reranker fallback, and engine registry recovery — all with mocked network/models, so no API keys or downloads are needed.

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

---

## Local development (without Docker)

### Prerequisites

- Python **3.11+**, Node.js + npm
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
| Worker logs `waiting for models prepared by the backend` | Normal — the worker waits (up to `MODEL_WAIT_TIMEOUT`, 900 s) for the backend to finish downloading into the shared `backend/models/` folder. |
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
│   ├── rag/                    # Engine registry + base retriever interfaces (tests.py inside)
│   ├── dense_rag/              # Dense retrieval (embeddings via OpenRouter)
│   ├── sparse_rag/             # BM25 retrieval (NLTK tokenization)
│   ├── hybrid_rag/             # Merge + dedup + local reranker (tests.py inside)
│   ├── corrective/             # CRAG evaluator, query expansion, external search
│   ├── multi_hop/              # Multi-hop retrieval orchestration
│   ├── evaluation/             # RAGAs-based generation evaluation
│   ├── chroma/                 # ChromaDB helpers
│   ├── ai_handler/             # LLM clients (OpenRouter/OpenAI) with retry (tests.py inside)
│   └── common/                 # Chunker, prompts, NLTK setup, helpers (tests.py inside)
└── frontend/                   # React + Vite + Tailwind UI, served by Nginx
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
