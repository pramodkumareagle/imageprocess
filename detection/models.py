from django.db import models
import uuid


# Create your models here.
class UploadImage(models.Model):
    media_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)  # Unique ID for each "bucket"
    image = models.ImageField(upload_to='images/')  # Original uploaded image
    processed_image = models.ImageField(upload_to='processed_images/', blank=True, null=True)  # Processed image
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Media ID: {self.media_id} - Image: {self.image.name}"

