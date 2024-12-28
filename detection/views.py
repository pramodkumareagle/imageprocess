from django.shortcuts import render
import torch
from django import forms
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load Hugging Face DETR model
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Form for uploading an image
class ImageUploadForm(forms.Form):
    image = forms.ImageField()

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Access the uploaded image
            uploaded_image = request.FILES['image']

            # Process the image with both models
            return detect_objects_with_both_models(request, uploaded_image)

    else:
        form = ImageUploadForm()
    return render(request, 'detection/upload.html', {'form': form})

def detect_objects_with_both_models(request, uploaded_image):
    # Convert the uploaded image to a NumPy array and PIL Image
    pil_image = Image.open(uploaded_image).convert("RGB")
    np_image = np.array(pil_image)

    # Run YOLOv5 detection
    yolo_results = yolo_model(np_image)
    yolo_results.render()
    yolo_detection_data = yolo_results.pandas().xyxy[0]  # Get YOLO detection data as DataFrame

    # Filter YOLO results by confidence threshold
    yolo_confidence_threshold = 0.2
    yolo_filtered_data = yolo_detection_data[yolo_detection_data['confidence'] >= yolo_confidence_threshold]
    yolo_filtered_data = yolo_filtered_data.to_dict(orient="records")

    # Convert YOLO rendered image to PIL format
    yolo_rendered_image = Image.fromarray(yolo_results.ims[0])

    # Run Hugging Face DETR detection
    detr_inputs = detr_processor(images=pil_image, return_tensors="pt")
    detr_outputs = detr_model(**detr_inputs)
    target_sizes = torch.tensor([pil_image.size[::-1]])  # Width, height
    detr_results = detr_processor.post_process_object_detection(
        detr_outputs, target_sizes=target_sizes, threshold=0.2
    )[0]

    # Draw bounding boxes for DETR
    detr_rendered_image = pil_image.copy()
    draw = ImageDraw.Draw(detr_rendered_image)
    detr_detection_data = []
    for score, label, box in zip(detr_results["scores"], detr_results["labels"], detr_results["boxes"]):
        x, y, x2, y2 = box.tolist()
        draw.rectangle([x, y, x2, y2], outline="blue", width=4)
        draw.text((x, y), f"{detr_model.config.id2label[label.item()]}: {score:.2f}", fill="blue")
        detr_detection_data.append({
            "name": detr_model.config.id2label[label.item()],
            "confidence": f"{score:.2f}",
            "box": [x, y, x2, y2],
        })

    # Encode images to Base64
    uploaded_image_base64 = _encode_image_base64(pil_image)
    yolo_image_base64 = _encode_image_base64(yolo_rendered_image)
    detr_image_base64 = _encode_image_base64(detr_rendered_image)

    # Render the results in HTML
    return render(request, 'detection/result.html', {
        'uploaded_image_base64': uploaded_image_base64,
        'yolo_image_base64': yolo_image_base64,
        'detr_image_base64': detr_image_base64,
        'yolo_detection_data': yolo_filtered_data,
        'detr_detection_data': detr_detection_data,
    })

def _encode_image_base64(image):
    """Helper function to encode a PIL image to Base64."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')
