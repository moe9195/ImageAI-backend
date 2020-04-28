from rest_framework import serializers
from .models import ImageAI
from .models import Method

class ProcessingSerializer(serializers.ModelSerializer):
    method = serializers.SerializerMethodField()
    class Meta:
        model = ImageAI
        fields = ["method"]
    def get_method(self, obj):
        return (Method.object.get(id=self.request.data.get("method")))