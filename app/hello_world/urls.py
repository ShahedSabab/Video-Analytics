from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('output', views.execute, name="triton"),
    path('model_prediction', views.external),
]
