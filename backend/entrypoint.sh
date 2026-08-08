#!/bin/sh
set -e

cd /app

MODEL_DIR="${MODEL_CACHE_DIR:-/app/models}"
JINA_MARKER="$MODEL_DIR/jinaai--jina-reranker-v3/.download_complete"
E5_MARKER="$MODEL_DIR/intfloat--multilingual-e5-small/.download_complete"

# ── Wait for PostgreSQL ──────────────────────────────────────────────
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for PostgreSQL..."
    python -c "
import os, time, sys
from urllib.parse import urlparse

db_url = os.environ.get('DATABASE_URL')
if db_url:
    import psycopg2
    url = urlparse(db_url)
    for i in range(30, 0, -1):
        try:
            conn = psycopg2.connect(
                host=url.hostname,
                port=url.port or 5432,
                user=url.username,
                password=url.password,
                dbname=url.path.lstrip('/'),
                connect_timeout=2
            )
            conn.close()
            print('PostgreSQL is ready.')
            break
        except Exception as e:
            print(f'Waiting for PostgreSQL... ({i} retries left)')
            time.sleep(2)
    else:
        print('ERROR: PostgreSQL did not become ready in time')
        sys.exit(1)
"
fi

# ── Wait for ChromaDB (plain TCP check — works with any chroma image) ─
echo "Waiting for ChromaDB..."
python -c "
import os, socket, time
host = os.environ.get('CHROMA_HOST', 'chromadb')
port = int(os.environ.get('CHROMA_PORT', '8000'))
for i in range(30, 0, -1):
    try:
        socket.create_connection((host, port), timeout=2).close()
        print('ChromaDB is ready.')
        break
    except OSError:
        print(f'Waiting for ChromaDB at {host}:{port}... ({i} retries left)')
        time.sleep(2)
else:
    print('WARNING: ChromaDB not reachable — continuing, dependent steps will retry')
"

if [ "${SKIP_INIT:-0}" = "1" ]; then
    # ── Worker mode: the backend container owns migrations, dataset ──
    # insert and model download. Just wait until the models it downloads
    # into the shared volume are ready, then start.
    timeout="${MODEL_WAIT_TIMEOUT:-900}"
    echo "SKIP_INIT=1 — waiting up to ${timeout}s for models prepared by the backend..."
    waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if [ -f "$JINA_MARKER" ] && [ -f "$E5_MARKER" ]; then
            echo "Models are ready."
            break
        fi
        sleep 5
        waited=$((waited + 5))
    done
    if [ ! -f "$JINA_MARKER" ] || [ ! -f "$E5_MARKER" ]; then
        echo "WARNING: models not ready after ${timeout}s — starting anyway (will fall back to HuggingFace Hub)"
    fi
else
    echo "Collecting static files..."
    python manage.py collectstatic --noinput

    echo "Applying migrations..."
    python manage.py migrate --noinput

    echo "Preparing base dataset..."
    python insert_base_dataset.py || echo "WARNING: insert_base_dataset.py failed — continuing anyway (re-run: docker compose exec backend python insert_base_dataset.py)"

    mkdir -p "$MODEL_DIR"
    echo "Checking / downloading models..."
    python dl_reranker_model.py || echo "WARNING: model download incomplete — continuing anyway (re-run: docker compose exec backend python dl_reranker_model.py)"
fi

exec "$@"
