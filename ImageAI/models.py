from django.db import models
from django.contrib.auth.models import User

class ImageAI(models.Model):
    img = models.ImageField(blank=True,  null=True)
    processed_img = models.ImageField(blank=True, null=True)
    method = models.ForeignKey('Method', on_delete=models.CASCADE, related_name="method")
    class Meta:
        verbose_name_plural = "imageAI"

class Method(models.Model):
    name = models.CharField(max_length=50)
    description = models.TextField(blank=True, null=True)


class Profile (models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE , primary_key = True )
    subscribed = models.BooleanField(default = False)
    limit = models.PositiveIntegerField(default = 0)
    key = models.CharField(max_length = 50 , null = True, blank = True)

    def str(self):
        return (f'{self.user.username}\'s profile')
