from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_swagger.views import get_swagger_view
from django.contrib import admin
from django.urls import path
from ImageAI import views

schema_view = get_swagger_view(title='ImageAI')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('super-resolution/', views.SuperResolution.as_view(), name="super"),
    path('colorize/', views.Colorize.as_view(), name="colorize"),
    path('deep-art/', views.DeepArt.as_view()),
    path('deblur/', views.Deblur.as_view()),
    path('classify/', views.Classify.as_view()),
    path('key/', views.GiveKey.as_view()),
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('profile/<int:profile_id>/detail/', views.ProfileDetails.as_view(), name='profile-detail'),
    path('profile/<int:profile_id>/update/', views.ProfileUpdate.as_view(), name='profile-update'),
    path('user/<int:user_id>/detail/', views.UserDetails.as_view(), name='user-detail'),
    path('documentation', schema_view, name="documentation"),
]
