#!/bin/sh
set -e

cd /app

# ── Wait for PostgreSQL (using Python + psycopg2, already installed) ──
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

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Applying migrations..."
python manage.py migrate --noinput

echo "Inserting Base Data..."
python insert_base_dataset.py || echo "WARNING: insert_base_dataset.py failed — continuing anyway"

MODEL_DIR="${MODEL_CACHE_DIR:-/app/models}"

mkdir -p "$MODEL_DIR"

if [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Downloading and preparing models..."
    python dl_reranker_model.py || echo "WARNING: Model download failed — continuing anyway"
else
    echo "Models already present, skipping download..."
fi

exec "$@"