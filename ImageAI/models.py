from django.db import models

class ImageAI(models.Model):
    img = models.ImageField(blank=True,  null=True)
    processed_img = models.ImageField(blank=True, null=True)
    method = models.ForeignKey('Method', on_delete=models.CASCADE, related_name="method")
    class Meta:
        verbose_name_plural = "imageAI"

class Method(models.Model):
    name = models.CharField(max_length=50)
    description = models.TextField(blank=True, null=True)
