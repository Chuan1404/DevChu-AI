from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views


# router = DefaultRouter()
# router.register("predict", views.ClassificationModelAPIView.as_view(), basename="classification")

urlpatterns = [
   path("predict", views.ClassificationModelAPIView.as_view()),
   path("color_detect", views.ColorDetectAPIView.as_view())
]