# app.py (Final Version with UI and enhanced drawing)

import torch
import cv2
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
import torchvision.transforms as transforms
from fastapi.middleware.cors import CORSMiddleware

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- App Setup ---
app = FastAPI(title="Elephant Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Model Loading ---
def create_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

MODEL_PATH = './outputs/model_epoch_5.pth'
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.5

print(f"Loading model onto device: {DEVICE}")
model = create_model(NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# Define the image transformation pipeline
# Note: Resize is handled by FastAPI for display consistency, model still expects original image size
transform = transforms.Compose([transforms.ToTensor()])


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return FileResponse('index.html')


@app.post("/predict/")
async def predict(image_file: UploadFile = File(...)):
    contents = await image_file.read()
    input_image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Store original size for drawing accurate boxes later
    original_width, original_height = input_image_pil.size

    # --- Preprocess for model ---
    input_tensor = transform(input_image_pil).unsqueeze(0).to(DEVICE)

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    
    # --- Draw boxes on a *copy* of the original image ---
    # Convert PIL image to OpenCV format for drawing
    image_to_draw = cv2.cvtColor(np.array(input_image_pil), cv2.COLOR_RGB2BGR)
    
    # Drawing parameters for better look
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0 # Increased for better visibility
    font_thickness = 2
    box_thickness = 3 # Increased for better visibility
    text_color = (0, 255, 0) # Green
    box_color = (0, 255, 0) # Green

    if len(outputs[0]['boxes']) > 0:
        for i, box in enumerate(outputs[0]['boxes']):
            score = outputs[0]['scores'][i].item()
            if score > CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax = map(int, box)
                
                # Draw the bounding box
                cv2.rectangle(image_to_draw, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
                
                # Prepare text for score
                text = f"Elephant: {score*100:.1f}%" # Display one decimal place for score
                
                # Calculate text size to place it smartly
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                
                # Ensure text is not drawn out of bounds at the top
                text_y = ymin - 10 if ymin - 10 > text_height + 5 else ymax + text_height + 5

                # Draw the text background for better readability
                cv2.rectangle(image_to_draw, (xmin, text_y - text_height - baseline), (xmin + text_width + 5, text_y + 5), (0,0,0), -1) # Black background
                # Draw the text
                cv2.putText(image_to_draw, text, (xmin + 5, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Encode the image with boxes back to a byte stream
    _, encoded_image = cv2.imencode(".jpg", image_to_draw)
    
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")