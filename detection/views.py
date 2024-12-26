from django.shortcuts import render
import torch
from .forms import ImageUploadForm
from .models import UploadImage
import os
from django.conf import settings

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            uploaded_image = form.save()
            image_path = uploaded_image.image.path  # Absolute path of the uploaded image

            # Process the image and render the result immediately
            return detect_objects(request, image_path, uploaded_image.image.url)

    else:
        form = ImageUploadForm()
    return render(request, 'detection/upload.html', {'form': form})


def detect_objects(request, image_path, uploaded_image_url):
    # Run YOLOv5 detection
    results = model(image_path)

    # Save results to the same directory as the uploaded image
    output_dir = os.path.dirname(image_path)
    results.save(save_dir=output_dir)

    # Locate the first processed file in the output directory
    processed_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png')) and 'exp' not in f]
    print(f"Processed files: {processed_files}")

    if not processed_files:
        raise FileNotFoundError("No processed files found.")

    processed_image = os.path.join(output_dir, processed_files[0])  # Get the first processed image
    print(f"Processed image path: {processed_image}")

    # Generate URL for the processed image
    processed_image_url = os.path.join(settings.MEDIA_URL, os.path.relpath(processed_image, settings.MEDIA_ROOT))
    print(f"Processed image URL: {processed_image_url}")

    # Get YOLOv5 detection data
    detection_data = results.pandas().xyxy[0].to_dict(orient="records")
    print(f"Detection data: {detection_data}")

    # Render the result
    return render(request, 'detection/result.html', {
        'uploaded_image_url': uploaded_image_url,
        'processed_image_url': processed_image_url,
        'detection_data': detection_data,
    })
