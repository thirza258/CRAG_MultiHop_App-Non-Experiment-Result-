#!/bin/sh
set -e

cd /app  

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Applying migrations..."
python manage.py migrate --noinput

echo "Inserting Base Data..."
python insert_base_dataset.py

MODEL_DIR="${MODEL_CACHE_DIR:-/app/models}"

mkdir -p "$MODEL_DIR"

if [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Downloading and preparing models..."
    python dl_reranker_model.py
else
    echo "Models already present, skipping download..."
fi

exec "$@"