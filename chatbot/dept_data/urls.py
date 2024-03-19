from django.urls import path
from . import views

urlpatterns = [
    path('chatbot/', views.Chatbot.as_view(), name='chatbot'),
]