from django import forms
from .models import ImageAI



class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageAI
        fields = ['img']
