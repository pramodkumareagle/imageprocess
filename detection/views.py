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


def handle_chatbot(request):
    if request.method == 'POST':
        user_message = request.POST.get('user_message', '').strip()
        if user_message:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": user_message}
                    ]
                )
                chatbot_response = response['choices'][0]['message']['content']
                return JsonResponse({"response": chatbot_response})
            except Exception as e:
                return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
        else:
            return JsonResponse({"error": "No user message provided"}, status=400)
    
    return render(request, 'detection/chatbot.html')

def _encode_image_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')
