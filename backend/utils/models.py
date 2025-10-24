import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
import os
import traceback

# Define base directory for absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Global device variable
DEVICE = None

def get_device():
    """Get the optimal device for model inference"""
    global DEVICE
    if DEVICE is None:
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
            print(f"[INFO] CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            DEVICE = torch.device("cpu")
            print("[WARNING] CUDA not available, using CPU")
    return DEVICE

# Define model paths with absolute paths
MODEL_PATHS = {
    "kerbs": os.path.join(BASE_DIR, "assets", "kerbs.pt"),
    "potholes": os.path.join(BASE_DIR, "assets", "best_ph2.pt"),
    "cracks": os.path.join(BASE_DIR, "assets", "best_ac8_types.pt"),
    "road_infra": os.path.join(BASE_DIR, "assets", "road_infra.pt"),
    "classification": os.path.join(BASE_DIR, "assets", "best_road_classification.pt")
}

# Class mapping for the new classification model
CLASSIFICATION_CLASSES = {
    0: "noroad",  # Non-road images
    1: "road"     # Road images
}

def load_yolo_models():
    """
    Load all YOLO models using ultralytics with proper device and dtype handling
    """
    models = {}
    device = get_device()
    
    for model_name, model_path in MODEL_PATHS.items():
        # Check if the model file exists
        print(f"Looking for model at path: {model_path}")
        if os.path.exists(model_path):
            try:
                # Load model with explicit device specification
                model = YOLO(model_path)
                
                # Move model to device early in the process
                model.to(device)
                
                # Set model to evaluation mode for inference
                try:
                    model.model.eval()
                except AttributeError:
                    # Handle case where model.model might not exist
                    if hasattr(model, 'eval'):
                        model.eval()
                
                # Apply device-specific optimizations
                if device.type == 'cuda':
                    try:
                        # Clear cache before optimization
                        torch.cuda.empty_cache()
                        
                        # Convert model to half precision for CUDA
                        model.model.half()
                        
                        # Warm up the model with a dummy input to ensure everything works
                        dummy_input = torch.randn(1, 3, 640, 640, device=device, dtype=torch.half)
                        with torch.no_grad():
                            _ = model.model(dummy_input)
                        
                        print(f"[INFO] Model {model_name} optimized for CUDA with FP16")
                        
                    except Exception as e:
                        print(f"[WARNING] CUDA optimization failed for {model_name}: {e}")
                        print(f"[INFO] Falling back to FP32 for {model_name}")
                        
                        # Fallback to FP32 on CUDA
                        model.model.float()
                        
                        # Test with FP32 dummy input
                        dummy_input = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float32)
                        with torch.no_grad():
                            _ = model.model(dummy_input)
                        
                        print(f"[SUCCESS] Model {model_name} loaded on CUDA with FP32")
                else:
                    # CPU optimization
                    model.model.float()  # Ensure FP32 for CPU
                    print(f"[SUCCESS] Model {model_name} loaded on CPU with FP32")
                
                models[model_name] = model
                models[f"{model_name}_classes"] = list(model.names.values())
                print(f"[SUCCESS] Loaded model: {model_name} on {device}")
                
            except Exception as e:
                print(f"[ERROR] Error loading model {model_name}: {e}")
                
                # Complete fallback to CPU if CUDA fails
                if device.type == 'cuda':
                    try:
                        print(f"[INFO] Retrying {model_name} on CPU...")
                        model = YOLO(model_path)
                        model.to(torch.device("cpu"))
                        model.model.float()  # Ensure FP32 for CPU
                        model.model.eval()
                        
                        # Test CPU inference
                        dummy_input = torch.randn(1, 3, 640, 640, device=torch.device("cpu"), dtype=torch.float32)
                        with torch.no_grad():
                            _ = model.model(dummy_input)
                        
                        models[model_name] = model
                        models[f"{model_name}_classes"] = list(model.names.values())
                        print(f"[SUCCESS] Loaded model: {model_name} on CPU (fallback)")
                    except Exception as e2:
                        print(f"[ERROR] Failed to load {model_name} on CPU: {e2}")
        else:
            print(f"[WARNING] Model file not found: {model_path}")
    
    return models

def load_midas():
    """
    Load the MiDaS depth estimation model with proper device and dtype handling
    """
    try:
        print("[INFO] Loading MiDaS model...")
        device = get_device()
        
        # First try loading the small model
        try:
            print("[INFO] Attempting to load MiDaS_small model...")
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
            print("[SUCCESS] MiDaS_small model loaded successfully")
        except Exception as e:
            print(f"[WARNING] Failed to load MiDaS_small, error: {str(e)}")
            print("[INFO] Attempting to load DPT_Large model as fallback...")
            try:
                midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True, trust_repo=True)
                print("[SUCCESS] DPT_Large model loaded successfully")
            except Exception as e2:
                print(f"[ERROR] Failed to load DPT_Large model: {str(e2)}")
                return None, None
        
        print(f"[INFO] Moving MiDaS model to device: {device}")
        midas.to(device)
        midas.eval()
        
        # Apply device-specific optimizations
        if device.type == 'cuda':
            try:
                print("[INFO] Optimizing model for CUDA...")
                torch.cuda.empty_cache()
                midas.half()  # Convert to FP16 for faster inference
                print("[SUCCESS] MiDaS model optimized for CUDA with FP16")
            except Exception as e:
                print(f"[WARNING] MiDaS CUDA optimization failed: {str(e)}")
                print("[INFO] Falling back to FP32 for MiDaS")
                midas.float()
                print("[SUCCESS] MiDaS model loaded on CUDA with FP32")
        else:
            midas.float()  # Ensure FP32 for CPU
            print("[SUCCESS] MiDaS model loaded on CPU with FP32")
        
        print("[INFO] Loading MiDaS transforms...")
        try:
            midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            print("[SUCCESS] MiDaS transforms loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load MiDaS transforms: {str(e)}")
            return None, None
        
        # Verify model is loaded correctly
        print("[INFO] Verifying MiDaS model...")
        try:
            # Create a dummy input
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            if device.type == 'cuda' and next(midas.parameters()).dtype == torch.float16:
                dummy_input = dummy_input.half()
            
            # Test inference
            with torch.no_grad():
                _ = midas(dummy_input)
            print("[SUCCESS] MiDaS model verification successful")
        except Exception as e:
            print(f"[ERROR] MiDaS model verification failed: {str(e)}")
            return None, None
        
        print(f"[SUCCESS] MiDaS setup completed successfully on {device}")
        return midas, midas_transform
        
    except Exception as e:
        print(f"[ERROR] Critical error in MiDaS setup: {str(e)}")
        
        # Fallback to CPU if CUDA fails
        if get_device().type == 'cuda':
            try:
                print("[INFO] Retrying MiDaS setup on CPU...")
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
                midas.to(torch.device("cpu"))
                midas.eval()
                midas.float()  # Ensure FP32 for CPU
                midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
                
                # Verify CPU model
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 256, 256)
                    _ = midas(dummy_input)
                
                print("[SUCCESS] MiDaS model loaded successfully on CPU (fallback)")
                return midas, midas_transform
            except Exception as e2:
                print(f"[ERROR] Failed to load MiDaS on CPU: {str(e2)}")
        
        return None, None

def estimate_depth(frame, midas, midas_transform):
    """
    Estimate depth from a single frame using MiDaS with proper dtype handling
    """
    device = get_device()
    
    try:
        # Convert image to RGB for MiDaS
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform image for model input
        img_tensor = midas_transform(img).to(device)
        
        # Use appropriate precision based on model dtype
        if device.type == 'cuda':
            try:
                # Check if model is in half precision
                model_dtype = next(midas.parameters()).dtype
                if model_dtype == torch.float16:
                    img_tensor = img_tensor.half()
                else:
                    img_tensor = img_tensor.float()
            except Exception:
                # Default to float32 if unsure
                img_tensor = img_tensor.float()
        else:
            img_tensor = img_tensor.float()
        
        # Add batch dimension if needed
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            prediction = midas(img_tensor)
        
        # Convert prediction to numpy
        depth_map = prediction.squeeze().cpu().numpy()
        
        # Robust normalization:
        # 1. Remove outliers using percentile clipping
        # 2. Apply non-linear scaling to enhance depth differences
        # 3. Normalize to 0-255 range
        
        # Remove outliers
        p1, p99 = np.percentile(depth_map, (1, 99))
        depth_map = np.clip(depth_map, p1, p99)
        
        # Non-linear scaling (gamma correction)
        depth_map = ((depth_map - p1) / (p99 - p1)) ** 0.5
        
        # Normalize to 0-255 range
        depth_map = (depth_map * 255).astype(np.uint8)
        
        # Resize to match input frame size
        if depth_map.shape != (frame.shape[0], frame.shape[1]):
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        # Apply slight Gaussian blur to reduce noise
        depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
        
        print("[SUCCESS] Depth map generated successfully")
        return depth_map
        
    except Exception as e:
        print(f"[ERROR] Error in depth estimation: {str(e)}")
        return None

def calculate_real_depth(binary_mask, depth_map, pixel_to_cm=0.1, calibration_value=1.6):
    """
    Calculate the real-world depth of a pothole from its binary mask and depth map
    """
    if binary_mask.shape != depth_map.shape:
        binary_mask = cv2.resize(binary_mask, (depth_map.shape[1], depth_map.shape[0]))
    
    # Apply Gaussian smoothing to reduce noise
    smoothed_depth = gaussian_filter(depth_map, sigma=2)
    
    # Get depth values within the pothole mask
    pothole_depths = smoothed_depth[binary_mask > 0]
    
    # Check if we have valid depth values
    if len(pothole_depths) == 0:
        return None
    
    # Calculate depth statistics
    max_depth = np.max(pothole_depths)
    min_depth = np.min(pothole_depths)
    avg_depth = np.mean(pothole_depths)
    std_depth = np.std(pothole_depths)
    
    # More robust depth calculation:
    # 1. Use percentiles instead of absolute min/max to avoid outliers
    # 2. Lower the std threshold to catch shallower potholes
    # 3. Add additional validation checks
    
    # Get depth values at 5th and 95th percentiles to avoid outliers
    p05_depth = np.percentile(pothole_depths, 5)
    p95_depth = np.percentile(pothole_depths, 95)
    
    # Calculate depth using percentile difference
    depth_cm = (p95_depth - p05_depth) * pixel_to_cm * calibration_value
    avg_depth_cm = (avg_depth - min_depth) * pixel_to_cm * calibration_value
    
    # Validation checks:
    # 1. Ensure minimum depth variation (lowered threshold)
    # 2. Ensure depth is positive
    # 3. Ensure depth is within reasonable range
    if std_depth < 0.5 or depth_cm <= 0 or depth_cm > 50:  # Lowered from 5 to 0.5, max depth 50cm
        print(f"Debug - Depth calculation stats: std={std_depth}, depth={depth_cm}cm")
        return None
    
    return {
        'max_depth_cm': round(depth_cm, 2),
        'avg_depth_cm': round(avg_depth_cm, 2)
    }

def calculate_pothole_dimensions(binary_mask, pixel_to_cm=0.1):
    """
    Calculate pothole dimensions (width, length, area) from a binary mask
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    
    width = rect[1][0] * pixel_to_cm
    length = rect[1][1] * pixel_to_cm
    area = cv2.contourArea(largest_contour) * (pixel_to_cm ** 2)
    
    return {
        'width_cm': round(width, 2), 
        'length_cm': round(length, 2), 
        'area_cm2': round(area, 2)
    }

def calculate_area(mask):
    """
    Calculate area from a binary mask.
    Assumes each pixel represents a fixed real-world area.
    """
    # Count non-zero pixels in the binary mask
    pixel_count = cv2.countNonZero(mask)

    # Conversion factor from pixel to cm² — adjust as needed
    # For example, assume 0.25 cm² per pixel
    pixel_to_cm2 = 0.25

    area_cm2 = pixel_count * pixel_to_cm2
    return {"area_cm2": area_cm2}

def classify_road_image(image, models, confidence_threshold=0.3):
    """
    Classify whether an image contains a road using the YOLOv11n classification model.

    Args:
        image: Input image (numpy array)
        models: Dictionary containing loaded models
        confidence_threshold: Minimum confidence for road classification (default: 0.3 - more permissive)

    Returns:
        dict: {
            "is_road": bool,
            "confidence": float,
            "class_name": str
        }
    """
    if not models or "classification" not in models:
        print("[WARNING] Classification model not available")
        return {"is_road": True, "confidence": 1.0, "class_name": "unknown"}  # Default to allow processing

    print(f"[DEBUG] Starting YOLOv11n classification with threshold {confidence_threshold}")
    print(f"[DEBUG] Available models: {list(models.keys())}")
    print(f"[DEBUG] Classification model type: {type(models['classification'])}")
    print(f"[DEBUG] Image shape: {image.shape}")
    print(f"[DEBUG] Image dtype: {image.dtype}")
    print(f"[DEBUG] Image min/max values: {image.min()}/{image.max()}")
    print(f"[DEBUG] Image channels: {image.shape[2] if len(image.shape) == 3 else 'N/A'}")

    # Debug: Print actual class names from the model
    if hasattr(models["classification"], 'names'):
        actual_class_names = models["classification"].names
        print(f"[DEBUG] Actual model class names: {actual_class_names}")
    else:
        print("[DEBUG] Model has no 'names' attribute")

    try:
        device = get_device()

        # Ensure image is in the correct format for the model
        if image.shape[2] == 3:
            # CRITICAL FIX: Test both color formats to handle camera vs uploaded image differences
            # Camera images and uploaded images may have different color channel ordering

            print(f"[DEBUG] Testing both color formats for optimal classification")

            # Test 1: Original image (might already be RGB from camera)
            test_image_1 = image.copy()

            # Test 2: BGR->RGB conversion (standard for OpenCV decoded images)
            test_image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Quick test both formats with the model to see which gives better results
            try:
                # Test format 1 (original)
                results_1 = models["classification"](test_image_1, conf=0.1, device=device)
                conf_1 = 0.0
                class_1 = -1
                if len(results_1) > 0 and results_1[0].probs is not None:
                    confidences_1 = results_1[0].probs.data.cpu().numpy()
                    class_1 = np.argmax(confidences_1)
                    conf_1 = float(confidences_1[class_1])

                # Test format 2 (BGR->RGB)
                results_2 = models["classification"](test_image_2, conf=0.1, device=device)
                conf_2 = 0.0
                class_2 = -1
                if len(results_2) > 0 and results_2[0].probs is not None:
                    confidences_2 = results_2[0].probs.data.cpu().numpy()
                    class_2 = np.argmax(confidences_2)
                    conf_2 = float(confidences_2[class_2])

                print(f"[DEBUG] Format 1 (original): class={class_1}, conf={conf_1:.3f}")
                print(f"[DEBUG] Format 2 (BGR->RGB): class={class_2}, conf={conf_2:.3f}")

                # Choose the format that gives road classification (class 1) with higher confidence
                # Updated for new model: Class 0 = noroad, Class 1 = road
                if class_1 == 1 and class_2 != 1:
                    inference_image = test_image_1
                    print(f"[DEBUG] Using original format (detected road)")
                elif class_2 == 1 and class_1 != 1:
                    inference_image = test_image_2
                    print(f"[DEBUG] Using BGR->RGB format (detected road)")
                elif class_1 == 1 and class_2 == 1:
                    # Both detect road, use the one with higher confidence
                    if conf_1 >= conf_2:
                        inference_image = test_image_1
                        print(f"[DEBUG] Using original format (higher road confidence)")
                    else:
                        inference_image = test_image_2
                        print(f"[DEBUG] Using BGR->RGB format (higher road confidence)")
                else:
                    # Neither detects road clearly, use BGR->RGB as default (standard OpenCV)
                    inference_image = test_image_2
                    print(f"[DEBUG] Using BGR->RGB format (default)")

            except Exception as e:
                print(f"[DEBUG] Error in format testing: {e}, using BGR->RGB as fallback")
                inference_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            inference_image = image
            print(f"[DEBUG] Using image as-is (not 3-channel)")

        # Run classification with proper error handling
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            try:
                results = models["classification"](inference_image, conf=0.1, device=device)
            except RuntimeError as e:
                if "dtype" in str(e):
                    print(f"[WARNING] Classification model dtype error: {e}")
                    print("[INFO] Attempting classification inference with CPU fallback...")
                    results = models["classification"](inference_image, conf=0.1, device='cpu')
                else:
                    raise e

        # Process classification results
        if results and len(results) > 0:
            result = results[0]
            print(f"[DEBUG] Classification result type: {type(result)}")
            print(f"[DEBUG] Result has probs: {hasattr(result, 'probs') and result.probs is not None}")
            print(f"[DEBUG] Result has boxes: {hasattr(result, 'boxes') and result.boxes is not None}")

            # Get the highest confidence prediction
            if result.probs is not None:
                # YOLOv11n Classification model with probabilities
                confidences = result.probs.data.cpu().numpy()
                class_idx = np.argmax(confidences)
                confidence = float(confidences[class_idx])

                print(f"[DEBUG] All confidences: {confidences}")
                print(f"[DEBUG] Selected class_idx: {class_idx}, confidence: {confidence}")

                # Get class names from the model - use actual model names
                if hasattr(models["classification"], 'names'):
                    class_names = list(models["classification"].names.values())
                else:
                    # Updated fallback for new model: Class 0 = noroad, Class 1 = road
                    class_names = list(CLASSIFICATION_CLASSES.values())

                print(f"[DEBUG] Using class names: {class_names}")
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                print(f"[DEBUG] Predicted class: {class_name}")

                # YOLOv11n road classification logic
                # The model should output classes like "road" and "non_road" or similar

                # YOLOv11n road classification logic based on actual class names
                # Class 0: '.ipynb_checkpoints' (training artifact)
                # Class 1: 'No Road'
                # Class 2: 'Road'

                print(f"[DEBUG] class_idx: {class_idx}, class_name: '{class_name}'")

                # Determine if image contains road based on class index and confidence
                # Updated for new model: Class 0 = noroad, Class 1 = road
                if class_idx == 1:  # Class 1 is "road"
                    # For road class, be more permissive with confidence
                    if confidence >= confidence_threshold:
                        is_road = True
                        print(f"[DEBUG] Road class detected with confidence ({confidence:.3f})")
                    else:
                        # Even with lower confidence, if it's classified as road, accept it
                        is_road = True
                        print(f"[DEBUG] Road class with lower confidence ({confidence:.3f}) - still accepting as road")
                elif class_idx == 0:  # Class 0 is "noroad"
                    # For non-road class, require higher confidence to reject
                    if confidence >= 0.7:  # Higher threshold for rejecting
                        is_road = False
                        print(f"[DEBUG] Non-road class detected with high confidence ({confidence:.3f}) - rejecting")
                    else:
                        # If confidence is low for non-road, default to road (be permissive)
                        is_road = True
                        print(f"[DEBUG] Non-road class with low confidence ({confidence:.3f}) - defaulting to road")
                else:
                    # Unknown class - default to road (be permissive)
                    is_road = True
                    print(f"[DEBUG] Unknown class ({class_idx}) - defaulting to road")

                print(f"[DEBUG] confidence: {confidence:.3f}")
                print(f"[DEBUG] confidence_threshold: {confidence_threshold}")
                print(f"[DEBUG] Final is_road: {is_road}")

                return {
                    "is_road": is_road,
                    "confidence": confidence,
                    "class_name": class_name
                }

            elif result.boxes is not None and len(result.boxes) > 0:
                # Detection model format - get highest confidence detection
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                print(f"[DEBUG] Detection format - confidences: {confidences}")
                print(f"[DEBUG] Detection format - class_ids: {class_ids}")

                if len(confidences) > 0:
                    max_conf_idx = np.argmax(confidences)
                    confidence = float(confidences[max_conf_idx])
                    class_id = int(class_ids[max_conf_idx])

                    # Get class names from the model - use actual model names
                    if hasattr(models["classification"], 'names'):
                        class_names = list(models["classification"].names.values())
                    else:
                        # Updated fallback for new model: Class 0 = noroad, Class 1 = road
                        class_names = list(CLASSIFICATION_CLASSES.values())

                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    print(f"[DEBUG] Detection - class_name: {class_name}, confidence: {confidence}")

                    # YOLOv11n road classification logic for detection format
                    # Class 0: '.ipynb_checkpoints' (training artifact)
                    # Class 1: 'No Road'
                    # Class 2: 'Road'

                    print(f"[DEBUG] Detection - class_id: {class_id}, class_name: '{class_name}'")

                    # More permissive logic for road classification (detection format)
                    # Updated for new model: Class 0 = noroad, Class 1 = road
                    if class_id == 1:  # Class 1 is "road"
                        # For road class, be more permissive
                        if confidence >= confidence_threshold:
                            is_road = True
                            print(f"[DEBUG] Detection - Road class detected with confidence ({confidence:.3f})")
                        else:
                            is_road = True  # Accept even with lower confidence
                            print(f"[DEBUG] Detection - Road class with lower confidence ({confidence:.3f}) - still accepting")
                    elif class_id == 0:  # Class 0 is "noroad"
                        # For non-road class, require higher confidence to reject
                        if confidence >= 0.7:
                            is_road = False
                            print(f"[DEBUG] Detection - Non-road class detected with high confidence ({confidence:.3f}) - rejecting")
                        else:
                            is_road = True  # Default to road if low confidence
                            print(f"[DEBUG] Detection - Non-road class with low confidence ({confidence:.3f}) - defaulting to road")
                    else:
                        # Unknown class - default to road (be permissive)
                        is_road = True
                        print(f"[DEBUG] Detection - Unknown class ({class_id}) - defaulting to road")

                    return {
                        "is_road": is_road,
                        "confidence": confidence,
                        "class_name": class_name
                    }

        # If no valid results, default to accepting (be permissive)
        print("[WARNING] No valid classification results found - defaulting to accept as road (permissive mode)")
        return {"is_road": True, "confidence": 0.5, "class_name": "no_detection_default_road"}

    except Exception as e:
        print(f"[ERROR] Error during road classification: {e}")
        traceback.print_exc()
        # In case of error, default to accepting to avoid blocking valid road images
        print("[DEBUG] Classification error - defaulting to accept as road (permissive mode)")
        return {"is_road": True, "confidence": 0.5, "class_name": "error_default_road"}