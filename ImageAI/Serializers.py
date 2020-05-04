from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Profile
from datetime import datetime, timedelta
from rest_framework_api_key.models import APIKey
from .models import ImageAI


class ProcessingSerializer(serializers.ModelSerializer):
    method = serializers.SerializerMethodField()
    class Meta:
        model = ImageAI
        fields = ["method"]
    def get_method(self, obj):
        return (Method.object.get(id=self.request.data.get("method")))


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

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["first_name", "last_name","username", "email"]

class ProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Profile
        exclude = ['user']
