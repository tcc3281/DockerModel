from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn
import time
import torch
from torchvision import models
import torch.nn as nn
import os

# === Global Variables and Configurations ===
NUM_CLASSES = 51
model_yolo = None
model_resnet = None
class_names_resnet = []

yolo_model_path = "model/best.pt"
resnet_model_path = "model/resnet50_best_model_v05.pth"
class_mapping_path = "class_mapping.txt"

# === Utility Functions ===
def load_class_mapping(txt_path):
    """Load class mapping from a file."""
    idx_to_class = {}
    with open(txt_path, 'r') as f:
        for line in f:
            idx, class_name = line.strip().split(':')
            idx_to_class[int(idx)] = class_name.strip()
    return idx_to_class

class_names_resnet = load_class_mapping(class_mapping_path)


async def load_yolo_model():
    """Tải model YOLO"""
    global model_yolo
    model_yolo = YOLO(yolo_model_path)
    # Warm up model
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model_yolo.predict(dummy_input, verbose=False)


def load_resnet_model():
    """Load ResNet model."""
    model = models.resnet50(weights=None)
    for name, param in model.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    checkpoint = torch.load(resnet_model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
    num_classes_checkpoint = next((state_dict[key].shape[0] for key in ['fc.weight', 'classifier.weight', 'head.fc.weight'] if key in state_dict), None)
    if num_classes_checkpoint is None:
        num_classes_checkpoint = len(checkpoint.get('classes_order', []))
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes_checkpoint)
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng"""
    global model_yolo, model_resnet
    model_yolo = YOLO(yolo_model_path)
    model_resnet = load_resnet_model()

    # Warm up models
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    _ = model_yolo.predict(dummy_image, verbose=False)

    dummy_tensor = torch.randn(1, 3, 224, 224)
    _ = model_resnet(dummy_tensor)

    yield

    # Clean up (nếu cần)


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    title="Traffic Sign Detection API",
    version="1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


def process_image(contents: bytes) -> np.ndarray:
    """Xử lý ảnh đầu vào tối ưu"""
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không thể đọc ảnh từ dữ liệu đầu vào")

    # Resize và chuẩn hóa ảnh
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@app.post("/detect")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        # Xử lý với model_yolo
        results = model_yolo.predict(image)

        # Tính toán thời gian
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return {
            "predictions": [{
                "class": model_yolo.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            } for box in results[0].boxes],
            "image_size": results[0].orig_shape,
            "detected_objects": len(results[0].boxes),
            "speed": {
                "preprocess": results[0].speed["preprocess"],
                "inference": results[0].speed["inference"],
                "postprocess": results[0].speed["postprocess"]
            },
            "total_time": total_time
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))


def process_image_for_resnet(contents: bytes) -> torch.Tensor:
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image")

    # Resize và chuẩn hóa
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # Quan trọng: float32

    # Chuẩn hóa theo ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    # Chuyển đổi sang tensor và thêm batch dimension
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image.float()  # Đảm bảo kiểu float32


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()

        # Preprocess
        preprocess_start = time.time()
        tensor = process_image_for_resnet(contents)

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = model_resnet(tensor)

        # Postprocess
        postprocess_start = time.time()
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        class_name = class_names_resnet[top_class.item()]

        # Tính thời gian
        preprocess_time = (time.time() - preprocess_start) * 1000
        inference_time = (time.time() - inference_start) * 1000
        postprocess_time = (time.time() - postprocess_start) * 1000
        total_time = (time.time() - start_time) * 1000

        return {
            "classification": {
                "class": class_name,
                "confidence": top_prob.item()
            },
            "speed": {
                "preprocess": preprocess_time,
                "inference": inference_time,
                "postprocess": postprocess_time
            },
            "total_time": total_time
        }

    except Exception as e:
        print(e)
        raise HTTPException(500, detail=f"Classification error: {str(e)}")


@app.post("/detect_and_classify")
async def detect_and_classify(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        # Detection
        detect_start = time.time()
        detect_results = model_yolo.predict(image)
        detect_time = (time.time() - detect_start) * 1000

        # Classification for each detected object
        classify_results = []
        for box in detect_results[0].boxes:
            if float(box.conf) > 0.6:  # Filter detection confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped_image = image[y1:y2, x1:x2]
                cropped_tensor = process_image_for_resnet(cv2.imencode('.jpg', cropped_image)[1].tobytes())

                with torch.no_grad():
                    outputs = model_resnet(cropped_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    top_prob, top_class = torch.max(probs, dim=1)

                    if top_prob.item() > 0.6:  # Filter classification confidence
                        classify_results.append({
                            "detection": {
                                "class": model_yolo.names[int(box.cls)],
                                "confidence": float(box.conf),
                                "bbox": [x1, y1, x2, y2]
                            },
                            "classification": {
                                "class": class_names_resnet[top_class.item()],
                                "confidence": top_prob.item()
                            }
                        })

        total_time = (time.time() - start_time) * 1000

        return {
            "results": classify_results,
            "image_size": detect_results[0].orig_shape,
            "detected_objects": len(classify_results),
            "speed": {
                "detection": detect_time,
                "total": total_time
            }
        }

    except Exception as e:
        raise HTTPException(500, detail=f"Detection and classification error: {str(e)}")

@app.get("/test")
async def test():
    return "API is running!"

# Mount static files
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = current_dir  # Trỏ trực tiếp đến thư mục hiện tại
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
async def serve_index():
    """Serve the index.html file."""
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)

if __name__ == '__main__':
    # Set host to '0.0.0.0' to allow external access
    host = '0.0.0.0'
    port = 8000  # Ensure this matches the exposed port in Docker
    uvicorn.run(
        app,
        host=host,
        port=port
    )
# python -m http.server 3000 --bind 0.0.0.0
# http://localhost:3000/
