import logging

import redis
from celery.result import AsyncResult
from django.db import connection
from rest_framework.response import Response
from rest_framework.views import APIView

from ragreader.settings import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """Lightweight liveness / readiness probe.

    Returns the reachability of each downstream dependency so an
    orchestrator (Docker health-check, Kubernetes liveness probe) can
    decide whether this instance is fit for traffic.
    """
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        checks = {}

        # ── Database ──────────────────────────────────────────────────
        try:
            connection.ensure_connection()
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {exc}"

        # ── Redis (broker + result backend) ───────────────────────────
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0,
                            socket_connect_timeout=2, socket_timeout=2)
            r.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"

        healthy = all(v == "ok" for v in checks.values())
        return Response(
            {"status": "ok" if healthy else "degraded", "checks": checks},
            status=200 if healthy else 503,
        )


class TaskStatusView(APIView):
    """REST fallback for checking Celery task state.

    Clients that lose their WebSocket connection can poll this endpoint
    to see whether their indexing task finished.
    """
    authentication_classes = []
    permission_classes = []

    def get(self, request, task_id: str):
        result = AsyncResult(task_id)
        payload = {
            "task_id": task_id,
            "status": result.status,
        }
        if result.ready():
            if result.successful():
                payload["result"] = result.result
            else:
                payload["error"] = str(result.result)
        return Response(payload)
