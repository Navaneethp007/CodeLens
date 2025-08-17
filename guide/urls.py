from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_and_index_code, name="upload"),
    path("search/", views.search_code, name="search")
]