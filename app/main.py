from fastapi import FastAPI, UploadFile, File
from PIL import Image
import time
import torch
import torch.nn as nn
from torchvision import models, transforms

# Create FastAPI app
app = FastAPI()

# Class names
classes = [
    "drivers_license",
    "other",
    "passport"
]

# Use CPU
device = torch.device("cpu")

# Load model architecture
model = models.mobilenet_v3_small(pretrained=False)

# Replace final layer
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    3
)

# Load trained weights
model.load_state_dict(
    torch.load("model/best_model.pth", map_location=device)
)

# Set evaluation mode
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction endpoint
@app.post("/v1/classify")
async def classify(file: UploadFile = File(...)):

    start_time = time.time()
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {
            "error": "Invalid file type. Please upload JPG or PNG image."
        }

    # Open image
    image = Image.open(file.file).convert("RGB")

    # Transform image
    image = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

    processing_time = int((time.time() - start_time) * 1000)

    # Return JSON
    return {
        "document_type": classes[pred.item()],
        "confidence": float(confidence.item()),
        "processing_time_ms": processing_time
    }