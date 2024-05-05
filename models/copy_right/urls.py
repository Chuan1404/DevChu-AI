from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views


# router = DefaultRouter()
# router.register("predict", views.ClassificationModelAPIView.as_view(), basename="classification")

urlpatterns = [
   path("hide_signature", views.CopyRightAPIView.as_view()),
   path("extract_signature", views.ExtractCopyRightAPIView.as_view())
]