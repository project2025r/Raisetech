import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
import os

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
    "road_infra": os.path.join(BASE_DIR, "assets", "road_infra.pt")
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
