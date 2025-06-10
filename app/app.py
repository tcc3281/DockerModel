import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import models
from ultralytics import YOLO

# === Global Variables and Configurations ===
NUM_CLASSES = 51
model_yolo = None
model_resnet = None
class_names_resnet = []

yolo_model_path = "model/best.pt"
resnet_model_path = "model/resnet50_best_model_v19.pth"
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


def debug_model_weights():
    """Debug function để kiểm tra model weights có được load đúng không"""
    try:
        print("\n=== DEBUGGING MODEL WEIGHTS ===")

        # Kiểm tra ResNet model
        if model_resnet is not None:
            # Kiểm tra các layer cuối
            fc_layers = model_resnet.fc
            print(f"FC Architecture: {fc_layers}")

            # Kiểm tra weights của layer cuối
            last_layer = fc_layers[-1]  # Linear layer cuối
            print(f"Last layer shape: {last_layer.weight.shape}")
            print(f"Last layer bias shape: {last_layer.bias.shape}")

            # Kiểm tra distribution của weights
            weight_mean = last_layer.weight.mean().item()
            weight_std = last_layer.weight.std().item()
            bias_mean = last_layer.bias.mean().item()

            print(f"Last layer weight mean: {weight_mean:.6f}")
            print(f"Last layer weight std: {weight_std:.6f}")
            print(f"Last layer bias mean: {bias_mean:.6f}")

            # Test với random input
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model_resnet(test_input)
                probs = torch.nn.functional.softmax(output, dim=1)
                max_prob = probs.max().item()
                print(f"Test prediction max probability: {max_prob:.4f}")

                # Kiểm tra entropy (độ không chắc chắn)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).item()
                print(f"Prediction entropy: {entropy:.4f} (high entropy = uncertain)")

            # Nếu weights quá nhỏ hoặc entropy quá cao thì có vấn đề
            if weight_std < 0.001:
                print("❌ WARNING: Weights seem too small - model might not be loaded properly!")
            if entropy > 3.5:
                print("❌ WARNING: High entropy - model predictions are very uncertain!")
            if max_prob < 0.1:
                print("❌ WARNING: Very low confidence - model likely not working!")

        else:
            print("❌ Model is None!")

    except Exception as e:
        print(f"❌ Error during model debugging: {e}")


def test_model_with_known_input():
    """Test model với input đã biết để kiểm tra output"""
    try:
        print("\n=== TESTING MODEL WITH SYNTHETIC INPUT ===")

        # Tạo test input patterns khác nhau
        test_cases = [
            ("zeros", torch.zeros(1, 3, 224, 224)),
            ("ones", torch.ones(1, 3, 224, 224)),
            ("random_normal", torch.randn(1, 3, 224, 224)),
            ("random_uniform", torch.rand(1, 3, 224, 224))
        ]

        for test_name, test_input in test_cases:
            with torch.no_grad():
                output = model_resnet(test_input)
                probs = torch.nn.functional.softmax(output, dim=1)
                top_prob, top_class = torch.max(probs, dim=1)

                print(f"{test_name}: max_prob={top_prob.item():.4f}, class={top_class.item()}")

                # Kiểm tra distribution của output
                output_mean = output.mean().item()
                output_std = output.std().item()
                print(f"  Output mean: {output_mean:.4f}, std: {output_std:.4f}")

    except Exception as e:
        print(f"❌ Error during model testing: {e}")


def load_resnet_model():
    """Load ResNet model with extensive debugging and validation."""
    try:
        print(f"Loading ResNet model from {resnet_model_path}")

        # Kiểm tra file tồn tại
        if not os.path.exists(resnet_model_path):
            raise FileNotFoundError(f"Model file not found: {resnet_model_path}")

        # Load checkpoint trước
        checkpoint = torch.load(resnet_model_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        # Kiểm tra checkpoint structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Using 'model_state_dict' from checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Using 'state_dict' from checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")

        print(f"State dict keys sample: {list(state_dict.keys())[:10]}")

        # Khởi tạo model architecture
        model = models.resnet50(weights=None)

        # ⚠️ QUAN TRỌNG: Phải thay thế FC layer TRƯỚC khi load weights
        original_fc_in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(original_fc_in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.Linear(512, NUM_CLASSES)
        )
        print(f"FC layer replaced: {original_fc_in_features} -> 512 -> {NUM_CLASSES}")

        # Clean up key names
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace('module.', '').replace('backbone.', '')
            cleaned_state_dict[cleaned_key] = value

        # Load weights với strict checking
        try:
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=True)
            print("✅ Loaded weights with strict=True (perfect match)")
        except RuntimeError as e:
            print(f"❌ Strict loading failed: {e}")
            print("Trying with strict=False...")
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {unexpected_keys}")

        # Set evaluation mode
        model.eval()

        # Disable all dropout layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.eval()
                print(f"Disabled dropout in: {name}")

        print("✅ ResNet model loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Error loading ResNet model: {e}")
        import traceback
        traceback.print_exc()
        raise e


def verify_model_compatibility():
    """Verify that the loaded model is compatible with expected classes."""
    try:
        # Kiểm tra số lượng classes
        expected_classes = [
            "P.127-5", "P.127-15", "P.136", "P.123a", "P.137", "P.123b", "P.124a", "P.103", "P.128", "P.127-30",
            "R.301f", "R.301a", "R.301e", "R.301i", "R.301d", "R.302a", "R.303", "R.403a", "R.309", "P.127-40",
            "P.127-50", "P.127-60", "P.101", "P.130", "P.102", "P.127-70", "P.127-80", "P.138", "P.139", "P.126",
            "P.122", "P.106a", "R.301h", "R.302b", "P.127-100", "P.127-120", "P.125", "P.127-20", "P.124c", "P.131a",
            "I.101", "W.207", "W.205", "W.208", "W.201", "W.206", "W.245", "W.221", "W.203", "R.308", "I.423"
        ]

        print(f"Expected number of classes: {len(expected_classes)}")
        print(f"NUM_CLASSES constant: {NUM_CLASSES}")
        print(f"Loaded class names: {len(class_names_resnet)}")

        # Kiểm tra consistency
        if len(expected_classes) != NUM_CLASSES:
            print(
                f"WARNING: Mismatch between expected_classes ({len(expected_classes)}) and NUM_CLASSES ({NUM_CLASSES})")

        if len(class_names_resnet) != NUM_CLASSES:
            print(
                f"WARNING: Mismatch between loaded classes ({len(class_names_resnet)}) and NUM_CLASSES ({NUM_CLASSES})")

        # Kiểm tra class names
        for i, expected_class in enumerate(expected_classes):
            if i in class_names_resnet and class_names_resnet[i] != expected_class:
                print(
                    f"WARNING: Class mismatch at index {i}: expected '{expected_class}', got '{class_names_resnet[i]}'")

        print("Model compatibility check completed!")
        return True

    except Exception as e:
        print(f"Error during compatibility check: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng với comprehensive debugging"""
    global model_yolo, model_resnet

    try:
        print("=== STARTING MODEL INITIALIZATION ===")

        # Load YOLO model
        print("Loading YOLO model...")
        model_yolo = YOLO(yolo_model_path)
        print(f"✅ YOLO model loaded successfully with {len(model_yolo.names)} classes")

        # Load ResNet model
        print("Loading ResNet model...")
        model_resnet = load_resnet_model()

        # Verify compatibility
        print("Verifying model compatibility...")
        verify_model_compatibility()

        # 🔍 CRITICAL: Debug model weights
        debug_model_weights()
        test_model_with_known_input()

        # Warm up models với proper error handling
        print("\n=== WARMING UP MODELS ===")
        try:
            # YOLO warm up
            dummy_image_yolo = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model_yolo.predict(dummy_image_yolo, verbose=False)
            print("✅ YOLO model warmed up successfully")

            # ResNet warm up với real preprocessing
            print("Testing ResNet with preprocessed input...")
            dummy_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)

            # Test với ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            normalized_tensor = (dummy_tensor - mean) / std

            with torch.no_grad():
                output = model_resnet(normalized_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                max_prob = probs.max().item()
                print(f"ResNet warm-up max probability: {max_prob:.4f}")

                if max_prob < 0.05:
                    print("❌ CRITICAL: ResNet confidence extremely low during warm-up!")
                    print("This indicates model weights are not loaded correctly!")
                else:
                    print("✅ ResNet warm-up successful with reasonable confidence")

        except Exception as e:
            print(f"❌ Model warm-up failed: {e}")
            import traceback
            traceback.print_exc()

        print("=== MODEL INITIALIZATION COMPLETED ===")

    except Exception as e:
        print(f"❌ FATAL ERROR during model initialization: {e}")
        import traceback
        traceback.print_exc()
        raise e

    yield

    # Clean up
    print("=== CLEANING UP MODELS ===")
    model_yolo = None
    model_resnet = None


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


@app.post("/detect_and_classify_simple_or")
async def detect_and_classify_simple(
    file: UploadFile = File(...),
    detection_threshold: float = Query(0.6, ge=0.1, le=1.0, description="Detection confidence threshold"),
    classification_threshold: float = Query(0.6, ge=0.1, le=1.0, description="Classification confidence threshold"),
    max_detections: int = Query(10, ge=1, le=50, description="Maximum number of detections to process")
):
    """
    Optimized endpoint for mobile apps to detect and classify traffic signs.
    
    Returns:
    - Simple list of detected label strings
    - Optimized for compressed images from mobile devices
    - Configurable confidence thresholds
    - Limited number of detections for faster processing
    """
    try:
        # Read and validate file
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(413, detail="Image file too large (max 10MB)")
        
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(400, detail="Empty file received")
        
        # Decode image with error handling
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, detail="Invalid image format")
        
        # Resize image if too large (optimization for mobile)
        height, width = image.shape[:2]
        if max(height, width) > 1280:
            scale = 1280 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # YOLOv8 detection with optimized settings
        detect_results = model_yolo.predict(
            image, 
            conf=detection_threshold,
            max_det=max_detections,
            verbose=False  # Reduce logging
        )

        # Quick exit if no detections
        if not detect_results[0].boxes or len(detect_results[0].boxes) == 0:
            return {"classification_results": []}

        # Classification for each detected object
        detected_labels = []
        processed_count = 0
        
        for box in detect_results[0].boxes:
            if processed_count >= max_detections:
                break
                
            confidence = float(box.conf)
            if confidence > detection_threshold:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Validate crop dimensions
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                # Crop with padding to avoid edge cases
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                cropped_image = image[y1:y2, x1:x2]
                
                # Skip if crop too small
                if cropped_image.shape[0] < 32 or cropped_image.shape[1] < 32:
                    continue
                
                try:
                    # Encode and process for ResNet
                    _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    cropped_tensor = process_image_for_resnet(buffer.tobytes())

                    # Classification
                    with torch.no_grad():
                        outputs = model_resnet(cropped_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        top_prob, top_class = torch.max(probs, dim=1)

                        if top_prob.item() > classification_threshold:
                            label = class_names_resnet[top_class.item()]
                            # Avoid duplicates
                            if label not in detected_labels:
                                detected_labels.append(label)
                                processed_count += 1
                                
                except Exception as crop_error:
                    # Log but continue processing other detections
                    print(f"Error processing crop: {crop_error}")
                    continue

        # Return simple list format for mobile apps
        return {"classification_results": detected_labels}  # Return only top 3 results
 # Direct list instead of dict

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")


@app.post("/detect_and_classify_simple")
async def detect_and_classify_mobile(file: UploadFile = File(...)):
    """
    Ultra-fast endpoint specifically for mobile apps with fixed optimal settings.
    Returns only the first 3 most confident detections for immediate response.
    """
    try:
        # Quick validation
        if file.size > 5 * 1024 * 1024:  # 5MB limit for mobile
            raise HTTPException(413, detail="Image too large for mobile endpoint")
        
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return []
        
        # Aggressive resize for speed
        height, width = image.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Fast detection with high confidence
        detect_results = model_yolo.predict(image, conf=0.7, max_det=3, verbose=False)
        
        if not detect_results[0].boxes:
            return []

        detected_labels = []
        for box in detect_results[0].boxes[:3]:  # Only process top 3
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped_image = image[max(0, y1):min(image.shape[0], y2), 
                                 max(0, x1):min(image.shape[1], x2)]
            
            if cropped_image.shape[0] >= 32 and cropped_image.shape[1] >= 32:
                try:
                    _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    cropped_tensor = process_image_for_resnet(buffer.tobytes())

                    with torch.no_grad():
                        outputs = model_resnet(cropped_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        top_prob, top_class = torch.max(probs, dim=1)

                        if top_prob.item() > 0.7:  # High confidence only
                            label = class_names_resnet[top_class.item()]
                            if label not in detected_labels:
                                detected_labels.append(label)
                except:
                    continue

        return {"classification_results": detected_labels}  # Return only top 3 results
    except Exception as e:
        return {}  # Fail silently for mobile apps


@app.get("/debug")
async def debug_model():
    """Debug endpoint để kiểm tra trạng thái model"""
    try:
        debug_info = {
            "model_status": {},
            "model_weights": {},
            "test_predictions": {}
        }

        # Kiểm tra YOLO
        if model_yolo is not None:
            debug_info["model_status"]["yolo"] = "loaded"
            debug_info["model_status"]["yolo_classes"] = len(model_yolo.names)
        else:
            debug_info["model_status"]["yolo"] = "not_loaded"

        # Kiểm tra ResNet
        if model_resnet is not None:
            debug_info["model_status"]["resnet"] = "loaded"

            # Test với synthetic input
            test_input = torch.randn(1, 3, 224, 224)
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            normalized_input = (test_input - mean) / std

            with torch.no_grad():
                output = model_resnet(normalized_input)
                probs = torch.nn.functional.softmax(output, dim=1)
                top_prob, top_class = torch.max(probs, dim=1)

                debug_info["test_predictions"]["max_confidence"] = float(top_prob.item())
                debug_info["test_predictions"]["predicted_class"] = int(top_class.item())
                debug_info["test_predictions"]["class_name"] = class_names_resnet.get(top_class.item(), "unknown")

                # Entropy check
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).item()
                debug_info["test_predictions"]["entropy"] = float(entropy)

                # Output statistics
                debug_info["model_weights"]["output_mean"] = float(output.mean().item())
                debug_info["model_weights"]["output_std"] = float(output.std().item())

                # Weight statistics
                last_layer = model_resnet.fc[-1]
                debug_info["model_weights"]["last_layer_weight_mean"] = float(last_layer.weight.mean().item())
                debug_info["model_weights"]["last_layer_weight_std"] = float(last_layer.weight.std().item())
                debug_info["model_weights"]["last_layer_bias_mean"] = float(last_layer.bias.mean().item())

        else:
            debug_info["model_status"]["resnet"] = "not_loaded"

        # Diagnostic messages
        diagnostics = []
        if debug_info["test_predictions"].get("max_confidence", 0) < 0.05:
            diagnostics.append("❌ CRITICAL: Model confidence extremely low - weights may not be loaded correctly")
        elif debug_info["test_predictions"].get("max_confidence", 0) < 0.1:
            diagnostics.append("⚠️ WARNING: Model confidence very low")
        else:
            diagnostics.append("✅ Model confidence appears normal")

        if debug_info["test_predictions"].get("entropy", 0) > 3.5:
            diagnostics.append("⚠️ WARNING: High entropy indicates uncertain predictions")

        if debug_info["model_weights"].get("last_layer_weight_std", 0) < 0.001:
            diagnostics.append("❌ WARNING: Weight standard deviation very low - possible initialization issue")

        debug_info["diagnostics"] = diagnostics
        debug_info["recommendations"] = [
            "If confidence is very low (<5%), check if checkpoint file contains trained weights",
            "If entropy is high (>3.5), model may be predicting randomly",
            "Make sure FC layer architecture matches training exactly",
            "Verify class mapping is correct"
        ]

        return debug_info

    except Exception as e:
        return {"error": str(e), "status": "debug_failed"}


@app.get("/test")
async def test():
    return {"status": "API is running!", "timestamp": time.time()}

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
    host = '192.168.1.103'
    port = 8000  # Ensure this matches the exposed port in Docker
    uvicorn.run(
        app,
        host=host,
        port=port
    )
# python -m http.server 3000 --bind 0.0.0.0
# http://localhost:3000/
