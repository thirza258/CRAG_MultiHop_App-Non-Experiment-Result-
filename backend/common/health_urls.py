from django.urls import path
from . import health_views

urlpatterns = [
    path("", health_views.HealthCheckView.as_view(), name="health-check"),
    path("task/<str:task_id>/", health_views.TaskStatusView.as_view(), name="task-status"),
]
