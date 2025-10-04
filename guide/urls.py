from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('query/', views.query_code, name='query_code'),
    path('clear/', views.clear_session, name='clear_session'),
    path('status/', views.file_status, name='file_status'),
]
