from django.urls import path
from .views import (
    InsertDataView,
    QueryView,
    InsertTextView,
    InsertURLView,
    ConversationHistoryView,
    ConversationView,
    DeleteDocumentView,
    DocumentView
)
from django.urls import re_path
from .consumers import QueryStreamConsumer

urlpatterns = [
    path("insert-data/", InsertDataView.as_view(), name="insert-data"),
    path("insert-text/", InsertTextView.as_view(), name="insert-text"),
    path("insert-url/", InsertURLView.as_view(), name="insert-url"),
    path("document/<str:username>/", DocumentView.as_view(), name="document-detail"),
    # The sidebar's delete button has always called this; the view existed but was
    # never routed, so every delete returned 404.
    path(
        "document/<str:document_id>/<str:username>/",
        DeleteDocumentView.as_view(),
        name="document-delete",
    ),
    path("conversation-history/<str:username>/", ConversationHistoryView.as_view(), name="conversation-history"),
    path("conversation/<str:conversation_id>/", ConversationView.as_view(), name="conversation"),
    path("query/", QueryView.as_view(), name="query"),

]

websocket_urlpatterns = [
    re_path(r"ws/query/stream/$", QueryStreamConsumer.as_asgi()),
]