from django.shortcuts import render
import torch
from .forms import ImageUploadForm
from .models import UploadImage
import os
import cv2
from django.conf import settings

# Create your views here.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            image_path = uploaded_image.image.path
            return detect_objects(request, image_path)
        
    else:
        form = ImageUploadForm()
    return render(request, 'detection/upload.html', {'form': form})

def detect_objects(request, image_path):
    results = model(image_path)
    results.save()

    output_dir = os.path.join(settings.MEDIA_ROOT, 'runs', 'detect')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exp_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if not exp_dirs:
        raise FileNotFoundError(f" No file {output_dir}")
    latest_exp_dir = max(exp_dirs, key=os.path.getmtime)  # Find the most recently modified directory
    output_image = os.path.join(latest_exp_dir, os.listdir(latest_exp_dir)[0])  # Access the first file in the directory

    return render(request, 'detection/result.html', {'output_image': output_image})

    
