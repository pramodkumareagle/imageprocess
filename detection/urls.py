from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('chatbot/', views.handle_chatbot, name='chatbot'),
    path('video_upload/', views.handle_video_upload, name='upload_video'),
]
