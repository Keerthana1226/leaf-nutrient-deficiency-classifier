import gradio as gr
import torch
import torch.nn as nn
import joblib
import pickle
from PIL import Image
from torchvision import models, transforms
import numpy as np
import json
import os
from collections import OrderedDict

DEVICE = 'cpu'
IMG_SIZE = 224

banana_model_loaded = False
try:
    weights_path = 'banana_model.pth'
    pipeline_path = 'banana_pipeline.pkl'
    print(f"Attempting to load Banana model from: {weights_path} and {pipeline_path}")
    if not os.path.exists(weights_path) or not os.path.exists(pipeline_path):
        raise FileNotFoundError("banana_model.pth or banana_pipeline.pkl not found.")

    pipeline_components = joblib.load(pipeline_path)
    banana_label_encoder = pipeline_components['label_encoder']
    banana_transforms = pipeline_components['val_transforms']

    from torchvision.models import efficientnet_b0
    banana_model = efficientnet_b0(weights=None)
    num_ftrs = banana_model.classifier[1].in_features
    banana_model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, len(banana_label_encoder.classes_))
    )
    banana_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    banana_model.to(DEVICE)
    banana_model.eval()
    print("Banana model loaded successfully.")
    banana_model_loaded = True
except Exception as e:
    print(f"Error loading banana model: {e}")


groundnut_model_loaded = False
try:
    weights_path = 'groundnut_mobilenet.pth'
    label_encoder_path = 'label_encoder_groundnut.pkl'
    print(f"Attempting to load Groundnut model from: {weights_path} and {label_encoder_path}")
    if not os.path.exists(weights_path) or not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"{weights_path} or {label_encoder_path} not found.")

    # Load the scikit-learn LabelEncoder
    with open(label_encoder_path, 'rb') as f:
        groundnut_label_encoder = pickle.load(f)
    groundnut_class_names = groundnut_label_encoder.classes_
    num_classes_groundnut = len(groundnut_class_names)

    # Re-create the exact Groundnut model architecture from your notebook
    groundnut_model = models.mobilenet_v2(weights=None)
    groundnut_model.classifier[1] = nn.Linear(groundnut_model.last_channel, num_classes_groundnut)
    
    # Load the weights with map_location='cpu'
    groundnut_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    groundnut_model.to(DEVICE)
    groundnut_model.eval()
    
    # Define the specific transforms for the Groundnut model
    groundnut_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Groundnut model loaded successfully.")
    groundnut_model_loaded = True
except Exception as e:
    print(f"Error loading groundnut model: {e}")


rice_model_loaded = False
try:
    weights_path = 'rice_mobilenet_from_scratch.pth'
    class_mapping_path = 'class_mapping.pkl'
    print(f"Attempting to load Rice model from: {weights_path} and {class_mapping_path}")
    if not os.path.exists(weights_path) or not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"{weights_path} or {class_mapping_path} not found.")

    with open(class_mapping_path, 'rb') as f:
        rice_class_mapping = pickle.load(f)
    rice_class_names = list(rice_class_mapping.values())
    num_classes_rice = len(rice_class_names)

    rice_model = models.mobilenet_v2(weights=None)
    num_features = rice_model.classifier[1].in_features
    rice_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(p=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes_rice)
    )
    
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('mobilenet_v2.', '')
        new_state_dict[name] = v
    
    rice_model.load_state_dict(new_state_dict)
    rice_model.to(DEVICE)
    rice_model.eval()
    
    rice_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Rice model loaded successfully.")
    rice_model_loaded = True
except Exception as e:
    print(f"Error loading rice model: {e}")


# --- PREDICTION FUNCTIONS ---
def predict_banana(image):
    if not banana_model_loaded: return {"Error: Banana model is not available": 1.0}
    image = image.convert("RGB")
    image_tensor = banana_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = banana_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidences = {banana_label_encoder.classes_[i]: float(probabilities[i]) for i in range(len(probabilities))}
    return {k.replace('Augmented ', '').replace('Banana ', ''): v for k, v in confidences.items()}

def predict_groundnut(image):
    if not groundnut_model_loaded: return {"Error: Groundnut model is not available": 1.0}
    image = image.convert("RGB")
    image_tensor = groundnut_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = groundnut_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidences = {groundnut_class_names[i]: float(probabilities[i]) for i in range(len(probabilities))}
    return confidences

def predict_rice(image):
    if not rice_model_loaded: return {"Error: Rice model is not available": 1.0}
    image = image.convert("RGB")
    image_tensor = rice_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = rice_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidences = {rice_class_names[i]: float(probabilities[i]) for i in range(len(probabilities))}
    return confidences

# --- GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multi-Plant Disease & Deficiency Classifier")
    gr.Markdown("Select a plant type, upload an image, and see the model's prediction.")
    with gr.Tabs():
        with gr.TabItem("Banana"):
            with gr.Row():
                banana_input = gr.Image(type="pil", label="Upload Banana Leaf Image")
                banana_output = gr.Label(num_top_classes=3, label="Prediction")
            banana_button = gr.Button("Classify", variant="primary")
        with gr.TabItem("Groundnut"):
            with gr.Row():
                groundnut_input = gr.Image(type="pil", label="Upload Groundnut Leaf Image")
                groundnut_output = gr.Label(num_top_classes=3, label="Prediction")
            groundnut_button = gr.Button("Classify", variant="primary")
        with gr.TabItem("Rice"):
            with gr.Row():
                rice_input = gr.Image(type="pil", label="Upload Rice Leaf Image")
                rice_output = gr.Label(num_top_classes=3, label="Prediction")
            rice_button = gr.Button("Classify", variant="primary")
            
    banana_button.click(predict_banana, inputs=banana_input, outputs=banana_output, api_name="predict_banana")
    groundnut_button.click(predict_groundnut, inputs=groundnut_input, outputs=groundnut_output, api_name="predict_groundnut")
    rice_button.click(predict_rice, inputs=rice_input, outputs=rice_output, api_name="predict_rice")

demo.launch()