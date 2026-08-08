#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# deploy.sh — build, test, and start the full stack with Docker Compose.
#
#   ./deploy.sh                normal deploy (build → test → up → wait)
#   ./deploy.sh --skip-tests   deploy without running the test suite
#
# Environment:
#   HEALTH_TIMEOUT   seconds to wait for the backend to become healthy
#                    (default 1800 — first run downloads ~2 GB of models)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; NC='\033[0m'
info() { printf "${BLUE}[deploy]${NC} %s\n" "$1"; }
ok()   { printf "${GREEN}[ ok ]${NC} %s\n" "$1"; }
warn() { printf "${YELLOW}[warn]${NC} %s\n" "$1"; }
fail() { printf "${RED}[fail]${NC} %s\n" "$1"; exit 1; }

SKIP_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --skip-tests) SKIP_TESTS=1 ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) fail "Unknown option: $arg (try --help)" ;;
    esac
done

# ── 1. Prerequisites ─────────────────────────────────────────────────
info "Checking prerequisites..."
command -v docker >/dev/null 2>&1 || fail "Docker is not installed. See https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1      || fail "The Docker daemon is not running. Start Docker and re-run."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required ('docker compose', not 'docker-compose')."
ok "Docker and Compose are available."

# ── 2. Environment files ─────────────────────────────────────────────
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    warn "Created backend/.env from the example — edit it and set OPENROUTER_API_KEY and SECRET_KEY."
fi
if [ ! -f frontend/.env ]; then
    cp frontend/.env.example frontend/.env
    ok "Created frontend/.env from the example (defaults are fine for Docker)."
fi

OPENROUTER_KEY="$(grep -E '^OPENROUTER_API_KEY=' backend/.env | head -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
if [ -z "$OPENROUTER_KEY" ]; then
    warn "OPENROUTER_API_KEY is empty in backend/.env — the app will start, but embeddings and answers will fail until you set it."
fi

# ── 3. Build ─────────────────────────────────────────────────────────
info "Building images (backend image is shared by the worker)..."
docker compose build
ok "Images built."

# ── 4. Test gate ─────────────────────────────────────────────────────
if [ "$SKIP_TESTS" = "1" ]; then
    warn "Skipping tests (--skip-tests)."
else
    info "Running the backend test suite inside the built image..."
    # DEVELOPMENT_MODE=true  → SQLite, so no database container is needed.
    # OPENROUTER_API_KEY=''  → keeps the RAG engine from initializing for
    #                          real during import; tests use mocks.
    if docker compose run --rm --no-deps --entrypoint "" \
        -e DEVELOPMENT_MODE=true \
        -e OPENROUTER_API_KEY= \
        backend python manage.py test --verbosity 2; then
        ok "All tests passed."
    else
        fail "Tests failed — deployment aborted. Fix the failures (or use --skip-tests to bypass at your own risk)."
    fi
fi

# ── 5. Start the stack ───────────────────────────────────────────────
info "Starting all services..."
docker compose up -d
ok "Services started."

# ── 6. Wait for the backend to become healthy ────────────────────────
TIMEOUT="${HEALTH_TIMEOUT:-1800}"
info "Waiting for the backend to finish initializing (migrations, dataset, model download)..."
info "First run can take a while — it downloads ~2 GB of models. Follow along: docker compose logs -f backend"

elapsed=0
while true; do
    status="$(docker inspect -f '{{.State.Health.Status}}' rag_backend 2>/dev/null || echo starting)"
    if [ "$status" = "healthy" ]; then
        break
    fi
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        printf "\n"
        docker compose logs --tail 30 backend || true
        fail "Backend not healthy after ${TIMEOUT}s. See the logs above (downloads may still be in progress — check 'docker compose logs -f backend')."
    fi
    if [ $((elapsed % 60)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
        info "Still initializing... (${elapsed}s elapsed, status: ${status})"
    fi
    sleep 10
    elapsed=$((elapsed + 10))
done

ok "Backend is healthy."

# ── 7. Summary ───────────────────────────────────────────────────────
printf "\n"
ok "Deployment complete."
printf "  %-12s %s\n" "App:"      "http://localhost:5151"
printf "  %-12s %s\n" "API:"      "http://localhost:8051"
printf "  %-12s %s\n" "Logs:"     "docker compose logs -f backend worker"
printf "  %-12s %s\n" "Status:"   "docker compose ps"
printf "  %-12s %s\n" "Stop:"     "docker compose down"
