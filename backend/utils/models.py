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
            print(f"✅ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            DEVICE = torch.device("cpu")
            print("⚠️ CUDA not available, using CPU")
    return DEVICE

# Define model paths with absolute paths
MODEL_PATHS = {
    "kerbs": os.path.join(BASE_DIR, "assets", "kerbs.pt"),
    "potholes": os.path.join(BASE_DIR, "assets", "best_ph2.pt"),
    "cracks": os.path.join(BASE_DIR, "assets", "best_ac8_types.pt"),
    "road_infra": os.path.join(BASE_DIR, "assets", "road_infra.pt"),
    "classification": os.path.join(BASE_DIR, "assets", "classfication.pt")  # Note: keeping original filename with typo
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
                        
                        print(f"🚀 Model {model_name} optimized for CUDA with FP16")
                        
                    except Exception as e:
                        print(f"⚠️ CUDA optimization failed for {model_name}: {e}")
                        print(f"🔄 Falling back to FP32 for {model_name}")
                        
                        # Fallback to FP32 on CUDA
                        model.model.float()
                        
                        # Test with FP32 dummy input
                        dummy_input = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float32)
                        with torch.no_grad():
                            _ = model.model(dummy_input)
                        
                        print(f"✅ Model {model_name} loaded on CUDA with FP32")
                else:
                    # CPU optimization
                    model.model.float()  # Ensure FP32 for CPU
                    print(f"✅ Model {model_name} loaded on CPU with FP32")
                
                models[model_name] = model
                models[f"{model_name}_classes"] = list(model.names.values())
                print(f"✅ Loaded model: {model_name} on {device}")
                
            except Exception as e:
                print(f"❌ Error loading model {model_name}: {e}")
                
                # Complete fallback to CPU if CUDA fails
                if device.type == 'cuda':
                    try:
                        print(f"🔄 Retrying {model_name} on CPU...")
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
                        print(f"✅ Loaded model: {model_name} on CPU (fallback)")
                    except Exception as e2:
                        print(f"❌ Failed to load {model_name} on CPU: {e2}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
    
    return models

def load_midas():
    """
    Load the MiDaS depth estimation model with proper device and dtype handling
    """
    try:
        print("Loading MiDaS model...")
        device = get_device()
        
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(device)
        midas.eval()
        
        # Apply device-specific optimizations
        if device.type == 'cuda':
            try:
                # Clear cache before optimization
                torch.cuda.empty_cache()
                midas.half()  # Convert to FP16 for faster inference
                print("🚀 MiDaS model optimized for CUDA with FP16")
            except Exception as e:
                print(f"⚠️ MiDaS CUDA optimization failed: {e}")
                print("🔄 Falling back to FP32 for MiDaS")
                midas.float()
                print("✅ MiDaS model loaded on CUDA with FP32")
        else:
            midas.float()  # Ensure FP32 for CPU
            print("✅ MiDaS model loaded on CPU with FP32")
        
        midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        print(f"✅ MiDaS model loaded successfully on {device}")
        return midas, midas_transform
        
    except Exception as e:
        print(f"❌ Error loading MiDaS model: {e}")
        
        # Fallback to CPU if CUDA fails
        if get_device().type == 'cuda':
            try:
                print("🔄 Retrying MiDaS on CPU...")
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                midas.to(torch.device("cpu"))
                midas.eval()
                midas.float()  # Ensure FP32 for CPU
                midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
                print("✅ MiDaS model loaded successfully on CPU (fallback)")
                return midas, midas_transform
            except Exception as e2:
                print(f"❌ Failed to load MiDaS on CPU: {e2}")
        return None, None

def estimate_depth(frame, midas, midas_transform):
    """
    Estimate depth from a single frame using MiDaS with proper dtype handling
    """
    device = get_device()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    
    with torch.no_grad():
        prediction = midas(img_tensor)
    
    depth_map = prediction.squeeze().cpu().numpy()
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
    
    return depth_map

def calculate_real_depth(binary_mask, depth_map, pixel_to_cm=0.1, calibration_value=1.6):
    """
    Calculate the real-world depth of a pothole from its binary mask and depth map
    """
    if binary_mask.shape != depth_map.shape:
        binary_mask = cv2.resize(binary_mask, (depth_map.shape[1], depth_map.shape[0]))
    
    smoothed_depth = gaussian_filter(depth_map, sigma=2)
    pothole_depths = smoothed_depth[binary_mask > 0]
    
    if len(pothole_depths) == 0 or np.std(pothole_depths) < 5:
        return None
    
    max_depth = np.max(pothole_depths)
    min_depth = np.min(pothole_depths)
    avg_depth = np.mean(pothole_depths)
    
    depth_cm = (max_depth - min_depth) * pixel_to_cm * calibration_value
    avg_depth_cm = avg_depth * pixel_to_cm * calibration_value
    
    return {'max_depth_cm': round(depth_cm, 2), 'avg_depth_cm': round(avg_depth_cm, 2)}

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
    Classify whether an image contains a road using the classification model.

    Args:
        image: Input image (numpy array)
        models: Dictionary containing loaded models
        confidence_threshold: Minimum confidence for road classification (default: 0.3)

    Returns:
        dict: {
            "is_road": bool,
            "confidence": float,
            "class_name": str
        }
    """
    if not models or "classification" not in models:
        print("⚠️ Classification model not available")
        return {"is_road": True, "confidence": 1.0, "class_name": "unknown"}  # Default to allow processing

    print(f"🔍 DEBUG: Starting classification with threshold {confidence_threshold}")
    print(f"🔍 DEBUG: Available models: {list(models.keys())}")
    print(f"🔍 DEBUG: Classification model type: {type(models['classification'])}")

    # Debug: Print actual class names from the model
    if hasattr(models["classification"], 'names'):
        actual_class_names = models["classification"].names
        print(f"🔍 DEBUG: Actual model class names: {actual_class_names}")
    else:
        print("🔍 DEBUG: Model has no 'names' attribute")

    try:
        device = get_device()

        # Ensure image is in the correct format for the model
        if image.shape[2] == 3:
            inference_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            inference_image = image

        # Run classification with proper error handling
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            try:
                results = models["classification"](inference_image, conf=0.1, device=device)
            except RuntimeError as e:
                if "dtype" in str(e):
                    print(f"⚠️ Classification model dtype error: {e}")
                    print("🔄 Attempting classification inference with CPU fallback...")
                    results = models["classification"](inference_image, conf=0.1, device='cpu')
                else:
                    raise e

        # Process classification results
        if results and len(results) > 0:
            result = results[0]
            print(f"🔍 DEBUG: Classification result type: {type(result)}")
            print(f"🔍 DEBUG: Result has probs: {hasattr(result, 'probs') and result.probs is not None}")
            print(f"🔍 DEBUG: Result has boxes: {hasattr(result, 'boxes') and result.boxes is not None}")

            # Get the highest confidence prediction
            if result.probs is not None:
                # Classification model with probabilities
                confidences = result.probs.data.cpu().numpy()
                class_idx = np.argmax(confidences)
                confidence = float(confidences[class_idx])

                print(f"🔍 DEBUG: All confidences: {confidences}")
                print(f"🔍 DEBUG: Selected class_idx: {class_idx}, confidence: {confidence}")

                # Get class names from the model - use actual model names
                if hasattr(models["classification"], 'names'):
                    class_names = list(models["classification"].names.values())
                else:
                    class_names = models.get("classification_classes", ["no_road", "road"])

                print(f"🔍 DEBUG: Using class names: {class_names}")
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                print(f"🔍 DEBUG: Predicted class: {class_name}")

                # Since this is an ImageNet model (not a road classifier), we need to:
                # 1. Detect clearly NON-road images
                # 2. Detect non-paved surfaces that shouldn't be analyzed for pavement defects

                # Classes that are clearly not roads at all
                non_road_classes = [
                    # Animals
                    "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "elephant", "bear", "lion", "tiger",
                    "monkey", "rabbit", "mouse", "snake", "spider", "butterfly", "bee", "ant",
                    # Indoor objects/scenes
                    "television", "computer", "laptop", "keyboard", "mouse", "monitor", "printer",
                    "refrigerator", "microwave", "oven", "toaster", "dishwasher", "washing_machine",
                    "bed", "chair", "table", "desk", "sofa", "couch", "lamp", "clock",
                    "book", "bottle", "cup", "plate", "bowl", "spoon", "fork", "knife",
                    # Food items
                    "pizza", "burger", "sandwich", "cake", "bread", "fruit", "apple", "banana",
                    "orange", "strawberry", "broccoli", "carrot", "potato",
                    # Clothing
                    "shirt", "pants", "dress", "shoe", "hat", "jacket", "tie", "sock",
                    # Tools/instruments that are clearly not road-related
                    "guitar", "piano", "violin", "drum", "flute", "trumpet",
                    "hammer", "screwdriver", "wrench", "drill", "saw"
                ]

                # Classes that indicate unpaved/natural surfaces (not suitable for pavement analysis)
                unpaved_surface_classes = [
                    # Natural terrain and unpaved surfaces
                    "sandbar", "beach", "lakeside", "seashore", "cliff", "valley", "mountain", "hill",
                    "desert", "dune", "field", "meadow", "pasture", "grassland", "prairie", "plain",
                    "forest", "woodland", "jungle", "swamp", "marsh", "bog", "wetland",
                    "dirt", "soil", "sand", "gravel", "stone", "rock", "boulder", "pebble",
                    "trail", "path", "track", "unpaved", "dirt_road", "gravel_road",
                    # Agricultural and rural scenes
                    "farm", "farmland", "agricultural", "rural", "countryside", "barn", "silo",
                    "tractor", "plow", "harvester", "crop", "wheat", "corn", "rice",
                    # Natural water features
                    "river", "stream", "creek", "pond", "lake", "ocean", "sea", "water",
                    # Additional classes that might indicate non-paved areas
                    "envelope", "cardboard", "carton", "package", "box", "container",
                    "landscape", "scenery", "outdoor", "nature", "terrain", "ground",
                    "earth", "mud", "clay", "dust", "powder", "granule"
                ]

                is_clearly_not_road = any(keyword in class_name.lower() for keyword in non_road_classes)
                is_unpaved_surface = any(keyword in class_name.lower() for keyword in unpaved_surface_classes)

                # More restrictive logic for road detection:
                # 1. Reject if clearly not a road scene with reasonable confidence
                # 2. Reject if unpaved/natural surface with low confidence threshold
                # 3. For very low confidence predictions, be more restrictive

                if is_clearly_not_road and confidence >= 0.3:
                    is_road = False
                elif is_unpaved_surface and confidence >= 0.15:  # Lower threshold for unpaved surfaces
                    is_road = False
                elif confidence < 0.05:  # Very low confidence - likely not a clear road
                    is_road = False
                else:
                    # Only accept if it has reasonable confidence and doesn't match rejection criteria
                    # Additional check: require minimum confidence for acceptance
                    is_road = confidence >= 0.1

                print(f"🔍 DEBUG: is_clearly_not_road: {is_clearly_not_road}")
                print(f"🔍 DEBUG: is_unpaved_surface: {is_unpaved_surface}")
                print(f"🔍 DEBUG: confidence: {confidence:.3f}")
                print(f"🔍 DEBUG: Final is_road: {is_road}")

                return {
                    "is_road": is_road,
                    "confidence": confidence,
                    "class_name": class_name
                }

            elif result.boxes is not None and len(result.boxes) > 0:
                # Detection model format - get highest confidence detection
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                print(f"🔍 DEBUG: Detection format - confidences: {confidences}")
                print(f"🔍 DEBUG: Detection format - class_ids: {class_ids}")

                if len(confidences) > 0:
                    max_conf_idx = np.argmax(confidences)
                    confidence = float(confidences[max_conf_idx])
                    class_id = int(class_ids[max_conf_idx])

                    # Get class names from the model - use actual model names
                    if hasattr(models["classification"], 'names'):
                        class_names = list(models["classification"].names.values())
                    else:
                        class_names = models.get("classification_classes", ["no_road", "road"])

                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    print(f"🔍 DEBUG: Detection - class_name: {class_name}, confidence: {confidence}")

                    # Use same logic as classification model - detect clearly non-road images
                    non_road_classes = [
                        "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "elephant", "bear", "lion", "tiger",
                        "monkey", "rabbit", "mouse", "snake", "spider", "butterfly", "bee", "ant",
                        "television", "computer", "laptop", "keyboard", "mouse", "monitor", "printer",
                        "refrigerator", "microwave", "oven", "toaster", "dishwasher", "washing_machine",
                        "bed", "chair", "table", "desk", "sofa", "couch", "lamp", "clock",
                        "book", "bottle", "cup", "plate", "bowl", "spoon", "fork", "knife",
                        "pizza", "burger", "sandwich", "cake", "bread", "fruit", "apple", "banana",
                        "orange", "strawberry", "broccoli", "carrot", "potato",
                        "shirt", "pants", "dress", "shoe", "hat", "jacket", "tie", "sock",
                        "guitar", "piano", "violin", "drum", "flute", "trumpet",
                        "hammer", "screwdriver", "wrench", "drill", "saw"
                    ]

                    is_clearly_not_road = any(keyword in class_name.lower() for keyword in non_road_classes)

                    if is_clearly_not_road and confidence >= 0.3:
                        is_road = False
                    else:
                        is_road = True  # Be permissive

                    return {
                        "is_road": is_road,
                        "confidence": confidence,
                        "class_name": class_name
                    }

        # If no valid results, default to allowing processing (temporary for debugging)
        print("⚠️ No valid classification results found - defaulting to allow processing")
        return {"is_road": True, "confidence": 0.0, "class_name": "no_detection"}

    except Exception as e:
        print(f"❌ Error during road classification: {e}")
        traceback.print_exc()
        # In case of error, default to allowing processing to avoid blocking the workflow
        return {"is_road": True, "confidence": 0.0, "class_name": "error"}
