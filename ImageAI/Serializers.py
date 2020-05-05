from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_api_key.models import APIKey
from .models import ImageAI, Profile
from datetime import datetime, timedelta


class ProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Profile
        exclude = ['user']


class UserSerializer(serializers.ModelSerializer):
    profile = ProfileSerializer()
    class Meta:
        model = User
        fields = ["first_name", "last_name","username", "email", "profile"]


class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        token['user'] = UserSerializer(user).data
        return token


class UserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    class Meta:
        model = User
        fields = ['username', 'password', 'first_name', "last_name", "email"]

    def create(self, validated_data):
        username = validated_data['username']
        password = validated_data['password']
        new_user = User(username=username)
        new_user.set_password(password)
        new_user.first_name = validated_data['first_name']
        new_user.last_name = validated_data['last_name']
        new_user.email = validated_data['email']
        new_user.save()
        profile =Profile.objects.create(user = new_user)
        api_key, key = APIKey.objects.create_key(name=username, expiry_date = datetime.now() + timedelta(days=30))
        profile.limit += 60
        profile.key= key
        profile.save()
        return validated_data


class ColorizeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageAI
        fields = ['img', 'format']

    def create(self, img, format):
        obj = ImageAI.objects.create(img = img, format = format)
        obj.save()
        return obj
