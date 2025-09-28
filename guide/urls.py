from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_and_analyze_code, name='upload'),
    path('search/', views.analyze_code, name='search'),
    
]
