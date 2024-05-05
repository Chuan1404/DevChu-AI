from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path('classification_model/', include("classification_model.urls")),
    path('copy_right/', include("copy_right.urls")),
    path('admin/', admin.site.urls),
]

