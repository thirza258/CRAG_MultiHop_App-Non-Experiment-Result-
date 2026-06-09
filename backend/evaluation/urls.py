from django.urls import path
from .views import (
    ChunkView, 

)

urlpatterns = [
    path('chunk/', ChunkView.as_view(), name='chunk-create'),
    
]
         