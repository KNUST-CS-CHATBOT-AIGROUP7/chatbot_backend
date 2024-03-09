from django.urls import path
from . import views

urlpatterns = [
    path('', views.Chatbot.as_view(), name='chatbot'),
]