"""
ASGI config for ragreader project.
"""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ragreader.settings")

from django.core.asgi import get_asgi_application

django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import router.urls

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            router.urls.websocket_urlpatterns
        )
    ),
})