"""CapstoneBackend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_swagger.views import get_swagger_view
from django.contrib import admin
from django.urls import path
from ImageAI import views

schema_view = get_swagger_view(title='ImageAI')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('post/', views.Processing.as_view()),
    path('key/', views.GiveKey.as_view()),
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('profile/<int:profile_id>/detail/', views.ProfileDetails.as_view(), name='profile-detail'),
    path('profile/<int:profile_id>/update/', views.ProfileUpdate.as_view(), name='profile-update'),
    path('user/<int:user_id>/detail/', views.UserDetails.as_view(), name='user-detail'),
    path('documentation', schema_view, name="documentation"),

]
