from django.shortcuts import render
import torch
from django import forms
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import openai
from django.http import JsonResponse
import os
from collections import Counter


openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load Hugging Face DETR model
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# LangChain setup
llm = OpenAI(temperature=0)  # Replace with your OpenAI API key
prompt = PromptTemplate(
    input_variables=["detections"],
    template="Analyze the following detections: {detections}"
)


# Form for uploading an image
class ImageUploadForm(forms.Form):
    image = forms.ImageField()

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = request.FILES['image']
            action = request.POST.get('action')
            if action == 'upload':
                return detect_objects_with_both_models(request, uploaded_image)
            elif action == 'langchain':
                return detect_with_langchain(request, uploaded_image)
            

    else:
        form = ImageUploadForm()
    return render(request, 'detection/upload.html', {'form': form})


def detect_objects_with_both_models(request, uploaded_image):
    pil_image = Image.open(uploaded_image).convert("RGB")
    np_image = np.array(pil_image)

    # YOLOv5 detection
    yolo_results = yolo_model(np_image)
    yolo_results.render()
    yolo_detection_data = yolo_results.pandas().xyxy[0].to_dict(orient="records")

    # Convert YOLO rendered image to PIL format
    yolo_rendered_image = Image.fromarray(yolo_results.ims[0])

    # DETR detection
    detr_inputs = detr_processor(images=pil_image, return_tensors="pt")
    detr_outputs = detr_model(**detr_inputs)
    target_sizes = torch.tensor([pil_image.size[::-1]])  # Width, height
    detr_results = detr_processor.post_process_object_detection(
        detr_outputs, target_sizes=target_sizes, threshold=0.2
    )[0]

    detr_detection_data = []
    detr_rendered_image = pil_image.copy()
    draw = ImageDraw.Draw(detr_rendered_image)
    for score, label, box in zip(detr_results["scores"], detr_results["labels"], detr_results["boxes"]):
        x, y, x2, y2 = box.tolist()
        draw.rectangle([x, y, x2, y2], outline="blue", width=4)
        draw.text((x, y), f"{detr_model.config.id2label[label.item()]}: {score:.2f}", fill="blue")
        detr_detection_data.append({
            "name": detr_model.config.id2label[label.item()],
            "confidence": f"{score:.2f}",
            "box": [x, y, x2, y2],
        })

    uploaded_image_base64 = _encode_image_base64(pil_image)
    yolo_image_base64 = _encode_image_base64(yolo_rendered_image)
    detr_image_base64 = _encode_image_base64(detr_rendered_image)

    return render(request, 'detection/result.html', {
        'uploaded_image_base64': uploaded_image_base64,
        'yolo_image_base64': yolo_image_base64,
        'detr_image_base64': detr_image_base64,
        'yolo_detection_data': yolo_detection_data,
        'detr_detection_data': detr_detection_data,
    })


def detect_with_langchain(request, uploaded_image):
    pil_image = Image.open(uploaded_image).convert("RGB")
    np_image = np.array(pil_image)

    # YOLOv5 detection
    yolo_results = yolo_model(np_image)
    yolo_detection_data = yolo_results.pandas().xyxy[0].to_dict(orient="records")

    # DETR detection
    detr_inputs = detr_processor(images=pil_image, return_tensors="pt")
    detr_outputs = detr_model(**detr_inputs)
    target_sizes = torch.tensor([pil_image.size[::-1]])
    detr_results = detr_processor.post_process_object_detection(
        detr_outputs, target_sizes=target_sizes, threshold=0.2
    )[0]

    # Annotate image with YOLO and DETR detections
    annotated_image = pil_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # YOLO Annotations
    for detection in yolo_detection_data:
        box = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
        label = detection['name']
        confidence = detection['confidence']
        draw.rectangle(box, outline="red", width=4)
        draw.text(
            (box[0] + 5, box[1] - 20),
            f"{label} ({confidence:.2f})",
            fill="red"
        )

    # DETR Annotations
    for score, label, box in zip(detr_results["scores"], detr_results["labels"], detr_results["boxes"]):
        x, y, x2, y2 = box.tolist()
        draw.rectangle([x, y, x2, y2], outline="blue", width=4)
        draw.text(
            (x + 5, y - 20),
            f"{detr_model.config.id2label[label.item()]} ({score:.2f})",
            fill="blue"
        )

    detr_detection_data = [
        {
            "name": detr_model.config.id2label[label.item()],
            "confidence": f"{score:.2f}",
            "box": [round(box[0].item(), 2), round(box[1].item(), 2), round(box[2].item(), 2), round(box[3].item(), 2)],
        }
        for score, label, box in zip(detr_results["scores"], detr_results["labels"], detr_results["boxes"])
    ]

    # Combine YOLO and DETR summaries
    combined_summary = f"YOLO detected: {', '.join([d['name'] for d in yolo_detection_data])}. DETR detected: {', '.join([d['name'] for d in detr_detection_data])}."
    chain = LLMChain(llm=llm, prompt=prompt)
    langchain_analysis = chain.run({"detections": combined_summary})

    # Preprocess the analysis into a list
    analysis_list = langchain_analysis.split(". ")

    # Encode the annotated image
    annotated_image_base64 = _encode_image_base64(annotated_image)

    return render(request, 'detection/langchain_analysis.html', {
        'annotated_image_base64': annotated_image_base64,
        'analysis_list': analysis_list,
        'yolo_detection_data': yolo_detection_data,
        'detr_detection_data': detr_detection_data,
    })


temp_data = {
    'image': None,
    'detections': None,
}

def handle_chatbot(request):
    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.FILES:
            uploaded_image = request.FILES['image']
            pil_image = Image.open(uploaded_image).convert("RGB")

            detections = detect_objects(pil_image)

            # Store the image and detections in memory
            temp_data['image'] = pil_image
            temp_data['detections'] = detections

            return JsonResponse({'message': 'Image uploaded successfully!', 'detections': detections})
        
        # Handle user message
        user_message = request.POST.get('user_message', '').strip()
        if user_message:
            if not temp_data['detections']:
                return JsonResponse({'message': 'Please upload an image first.'})
            
            # Add detections to the message
            detections_summary = ', '.join([f"{count} {obj}(s)" for obj, count in temp_data['detections'].items()])
            user_message += f" The image contains: {detections_summary}."

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant analyzing image detections."},
                        {"role": "user", "content": user_message}
                    ]
                )
                chatbot_response = response['choices'][0]['message']['content']
                return JsonResponse({'message': chatbot_response})
            except Exception as e:
                return JsonResponse({'message': str(e)})
    
    return render(request, 'detection/chatbot.html')

def detect_objects(image):
    results = yolo_model(image)
    detections = [item['name'] for item in results.pandas().xyxy[0].to_dict(orient="records")]
    # Group detections by object type and count them
    detection_counts = dict(Counter(detections))
    return detection_counts

from django.core.files.storage import FileSystemStorage
import cv2

def handle_video_upload(request):
    if request.method == 'POST' and 'video' in request.FILES:
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        video_path = fs.save(video_file.name, video_file)
        video_full_path = fs.path(video_path)

        # Process the video and save the output
        output_video_path, detections_summary = process_video(video_full_path)

        return JsonResponse({
            'message': 'Video processed successfully!',
            'detections': detections_summary,
            'processed_video_url': fs.url(output_video_path),
        })

    return render(request, 'detection/video_upload.html')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # save the processed video
    output_path = video_path.replace('.mp4', '_processed.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert the frame to PIL format to detect
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Run object detection on the frame
        detections = detect_objects_videos(pil_frame)
        all_detections.extend([d['name'] for d in detections])
        # Annotate the frame
        annotated_frame = annotate_frame(frame, detections)
        # write the annotated frame to the output video
        out.write(annotated_frame)

    cap.release()
    out.release()

    detections_summary = dict(Counter(all_detections))
    return output_path, detections_summary


def detect_objects_videos(frame):
    results = yolo_model(frame)
    # detecting the labels
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return detections


def annotate_frame(frame, detections):
    for detection in detections:
        xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        name = detection['name']
        # Draw bounding box and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # add label
        cv2.putText(frame, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def _encode_image_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

