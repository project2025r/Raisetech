from flask import Blueprint, request, jsonify, Response, stream_with_context, session
import cv2
import numpy as np
import base64
import os
import json
import traceback
from config.db import connect_to_db, get_gridfs
from utils.models import load_yolo_models, load_midas, estimate_depth, calculate_real_depth, calculate_pothole_dimensions, calculate_area, get_device, classify_road_image
from utils.exif_utils import get_gps_coordinates, format_coordinates
import pandas as pd
import io
from bson import ObjectId
import uuid
import logging
import time
import torch
from datetime import datetime
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError

# Import our comprehensive S3-MongoDB integration
from s3_mongodb_integration import ImageProcessingWorkflow, S3ImageManager, MongoDBImageManager

# Import file validation utilities
from utils.file_validation import validate_upload_file, validate_base64_image, get_context_specific_error_message

pavement_bp = Blueprint('pavement', __name__)

# Global variables for models - lazy loaded or preloaded
models = None
midas = None
midas_transform = None

# Tracking helper functions
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    intersection_x_min = max(x1_min, x2_min)
    intersection_y_min = max(y1_min, y2_min)
    intersection_x_max = min(x1_max, x2_max)
    intersection_y_max = min(y1_max, y2_max)
    
    if intersection_x_max <= intersection_x_min or intersection_y_max <= intersection_y_min:
        return 0.0
    
    intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_box_center(box):
    """Calculate center point of a bounding box"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def match_detections_to_tracks(detections, tracks, iou_threshold=0.3, distance_threshold=100):
    """Match new detections to existing tracks based on IoU and distance"""
    matched_pairs = []
    unmatched_detections = []
    unmatched_tracks = list(tracks.keys())
    
    for detection in detections:
        best_match_id = None
        best_iou = 0.0
        best_distance = float('inf')
        
        detection_center = calculate_box_center(detection['bbox'])
        
        for track_id in tracks:
            if track_id not in unmatched_tracks:
                continue
                
            track = tracks[track_id]
            
            # Only match same type of detections
            if track['type'] != detection['type']:
                continue
            
            # Calculate IoU
            iou = calculate_iou(detection['bbox'], track['bbox'])
            
            # Calculate distance between centers
            track_center = calculate_box_center(track['bbox'])
            distance = calculate_distance(detection_center, track_center)
            
            # Consider it a match if IoU is above threshold OR distance is small
            if iou > iou_threshold or distance < distance_threshold:
                if iou > best_iou or (iou == best_iou and distance < best_distance):
                    best_match_id = track_id
                    best_iou = iou
                    best_distance = distance
        
        if best_match_id:
            matched_pairs.append((detection, best_match_id))
            unmatched_tracks.remove(best_match_id)
        else:
            unmatched_detections.append(detection)
    
    return matched_pairs, unmatched_detections, unmatched_tracks

class DefectTracker:
    """Class to track defects across video frames"""
    
    def __init__(self, max_missing_frames=30, confidence_threshold=0.4):
        self.tracks = {}
        self.next_id = 1
        self.max_missing_frames = max_missing_frames
        self.confidence_threshold = confidence_threshold
        self.unique_detections = []  # Store unique detections for final output
    
    def update(self, detections, frame_count):
        """Update tracks with new detections"""
        # Filter detections by confidence
        filtered_detections = [d for d in detections if d['confidence'] >= self.confidence_threshold]
        
        # Match detections to existing tracks
        matched_pairs, unmatched_detections, unmatched_tracks = match_detections_to_tracks(
            filtered_detections, self.tracks
        )
        
        # Update matched tracks
        for detection, track_id in matched_pairs:
            self.tracks[track_id].update({
                'bbox': detection['bbox'],
                'confidence': max(self.tracks[track_id]['confidence'], detection['confidence']),
                'last_seen': frame_count,
                'times_seen': self.tracks[track_id]['times_seen'] + 1
            })
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'type': detection['type'],
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'first_seen': frame_count,
                'last_seen': frame_count,
                'times_seen': 1,
                'track_id': track_id
            }
            
            # Add to unique detections (only add once per track)
            unique_detection = detection.copy()
            unique_detection['track_id'] = track_id
            unique_detection['first_detected_frame'] = frame_count
            self.unique_detections.append(unique_detection)
        
        # Remove old tracks that haven't been seen for too long
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if frame_count - track['last_seen'] > self.max_missing_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return current frame detections (only for display)
        current_frame_detections = []
        for detection, track_id in matched_pairs:
            current_detection = detection.copy()
            current_detection['track_id'] = track_id
            current_detection['is_tracked'] = True
            current_frame_detections.append(current_detection)
        
        for detection in unmatched_detections:
            current_detection = detection.copy()
            current_detection['is_tracked'] = False
            current_frame_detections.append(current_detection)
        
        return current_frame_detections
    
    def get_unique_detections(self):
        """Get list of unique detections (one per track)"""
        return self.unique_detections
    
    def get_active_tracks(self):
        """Get currently active tracks"""
        return self.tracks

# Video processing global variables
video_processing_stop_flag = False

# Create processed_videos directory if it doesn't exist
PROCESSED_VIDEOS_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed_videos')
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

# Configure logging for video processing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preload_models_on_startup():
    """Eagerly preload YOLO and MiDaS models when the Flask server starts"""
    global models, midas, midas_transform

    print("\n=== Starting Model Preload ===")
    print("🔄 Preloading pavement models on server startup...")
    
    # Load YOLO models
    models = load_yolo_models()
    for model in models.values():
        try:
            model.eval()
        except Exception as e:
            print(f"⚠️ Warning: Could not set model to eval mode: {e}")
    
    # Load MiDaS model with proper error handling
    print("\n=== Loading MiDaS Model ===")
    try:
        midas, midas_transform = load_midas()
        if midas is None or midas_transform is None:
            print("❌ Failed to load MiDaS model - depth estimation will be unavailable")
        else:
            print("✅ MiDaS model loaded successfully")
            if hasattr(midas, 'eval'):
                midas.eval()
                print("✅ MiDaS model set to eval mode")
            
            # Verify MiDaS works with a test input
            device = get_device()
            try:
                print("🔄 Testing MiDaS with dummy input...")
                dummy_input = torch.randn(1, 3, 384, 384).to(device)
                if device.type == 'cuda':
                    dummy_input = dummy_input.half()
                with torch.no_grad():
                    _ = midas(dummy_input)
                print("✅ MiDaS test inference successful")
            except Exception as e:
                print(f"❌ MiDaS test inference failed: {str(e)}")
                midas, midas_transform = None, None
    except Exception as e:
        print(f"❌ Error during MiDaS initialization: {str(e)}")
        midas, midas_transform = None, None
    
    print("\n=== Model Preload Status ===")
    print(f"✓ YOLO Models: {len(models)} loaded")
    print(f"✓ MiDaS: {'Available' if midas else 'Unavailable'}")
    print("=== Preload Complete ===\n")
    
    return models, midas, midas_transform

def preload_models():
    """Preload YOLO and MiDaS models before the first request if not already loaded"""
    global models, midas, midas_transform

    # Only load if models haven't been loaded yet
    if models is None:
        print("\n=== Loading YOLO Models ===")
        models = load_yolo_models()
        for model in models.values():
            try:
                model.eval()
            except Exception as e:
                print(f"⚠️ Warning: Could not set model to eval mode: {e}")

    if midas is None or midas_transform is None:
        print("\n=== Loading MiDaS Model ===")
        try:
            midas, midas_transform = load_midas()
            if midas is None or midas_transform is None:
                print("❌ Failed to load MiDaS model - depth estimation will be unavailable")
            else:
                print("✅ MiDaS model loaded successfully")
                if hasattr(midas, 'eval'):
                    midas.eval()
                    print("✅ MiDaS model set to eval mode")
                
                # Verify MiDaS works with a test input
                device = get_device()
                try:
                    print("🔄 Testing MiDaS with dummy input...")
                    dummy_input = torch.randn(1, 3, 384, 384).to(device)
                    if device.type == 'cuda':
                        dummy_input = dummy_input.half()
                    with torch.no_grad():
                        _ = midas(dummy_input)
                    print("✅ MiDaS test inference successful")
                except Exception as e:
                    print(f"❌ MiDaS test inference failed: {str(e)}")
                    midas, midas_transform = None, None
        except Exception as e:
            print(f"❌ Error during MiDaS initialization: {str(e)}")
            midas, midas_transform = None, None


def get_models():
    """Return preloaded models"""
    global models, midas, midas_transform
    return models, midas, midas_transform


def decode_base64_image(base64_string):
    """Decode a base64 image to cv2 format with automatic AVIF to JPG conversion"""
    print(f"🔍 DEBUG: Base64 string prefix: {base64_string[:50]}...")

    # Check if this is an AVIF image and convert if necessary
    if 'data:image/avif;base64,' in base64_string or 'data:image/;base64,' in base64_string:
        print("🔄 AVIF image detected, converting to JPG...")
        try:
            from utils.image_converter import convert_image_to_yolo_supported
            base64_string = convert_image_to_yolo_supported(base64_string)
            print("✅ AVIF successfully converted to JPG")
        except Exception as e:
            print(f"❌ Error converting AVIF to JPG: {str(e)}")
            return None

    if 'base64,' in base64_string:
        header = base64_string.split('base64,')[0]
        print(f"🔍 DEBUG: Image header: {header}")
        base64_string = base64_string.split('base64,')[1]

    img_data = base64.b64decode(base64_string)
    print(f"🔍 DEBUG: Decoded image data size: {len(img_data)} bytes")

    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is not None:
        print(f"🔍 DEBUG: Decoded image shape: {img.shape}")
        print(f"🔍 DEBUG: Decoded image dtype: {img.dtype}")
        print(f"🔍 DEBUG: Decoded image min/max: {img.min()}/{img.max()}")
    else:
        print("🔍 DEBUG: Failed to decode image!")

    return img

def encode_processed_image(image):
    """Encode a processed image to base64"""
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_image}"


def process_video_frame_pavement(frame, frame_count, selected_model, models, midas, midas_transform, tracker=None):
    """Process a single video frame for pavement defect detection with tracking and CUDA optimization"""
    logger.debug(f"Processing pavement frame {frame_count}")
    
    start_time = time.time()
    detection_frame = frame.copy()
    original_frame = frame.copy()
    all_detections = []
    device = get_device()
    
    # Calculate depth map for this frame if processing potholes and MiDaS is available
    depth_map = None
    if (selected_model == "All" or selected_model == "Potholes") and midas and midas_transform:
        try:
            logger.debug("🔄 Running depth estimation with MiDaS...")
            depth_map = estimate_depth(original_frame, midas, midas_transform)
            if depth_map is not None:
                logger.debug("✅ Depth map generated successfully")
            else:
                logger.warning("⚠️ Depth map generation failed - got None result")
        except Exception as e:
            logger.warning(f"⚠️ Error during depth estimation: {str(e)}")
            depth_map = None
    else:
        logger.warning("⚠️ MiDaS model not available or not needed for selected model")
    
    try:
        # Determine which models to use based on selection
        models_to_use = []
        if selected_model == "All":
            models_to_use = [("potholes", "Pothole"), ("cracks", "Crack"), ("kerbs", "Kerb")]
        elif selected_model == "Potholes":
            models_to_use = [("potholes", "Pothole")]
        elif selected_model == "Alligator Cracks":
            models_to_use = [("cracks", "Crack")]
        elif selected_model == "Kerbs":
            models_to_use = [("kerbs", "Kerb")]
        else:
            # Default to all models
            models_to_use = [("potholes", "Pothole"), ("cracks", "Crack"), ("kerbs", "Kerb")]
        
        # Process each selected model
        for model_key, display_name in models_to_use:
            if model_key not in models:
                logger.warning(f"Model {model_key} not available, skipping")
                continue
            
            # Initialize detection_id for each model
            detection_id = 1
            # Use a fresh copy of the original frame for each model
            inference_frame = original_frame.copy()
            
            # CUDA optimization: Run inference with optimal settings and proper dtype handling
            with torch.no_grad():  # Disable gradient computation for inference
                # Clear GPU cache if using CUDA
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Ensure image is in the correct format for the model
                if inference_frame.shape[2] == 3:
                    inference_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                
                # Run detection with the model with proper error handling
                try:
                    results = models[model_key](inference_frame, conf=0.2, device=device)
                except RuntimeError as e:
                    if "dtype" in str(e):
                        print(f"⚠️ Video processing dtype error for {model_key}: {e}")
                        print(f"🔄 Attempting video inference with CPU fallback for {model_key}...")
                        results = models[model_key](inference_frame, conf=0.2, device='cpu')
                    else:
                        raise e
            
            # Remove per-frame detection_id logic; all_detections will be handled after tracking
            
            # Process results based on model type
            if model_key == "potholes":
                # Handle pothole detection (bounding boxes and segmentation masks)
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        # Check if segmentation masks are available
                        if result.masks is not None:
                            # Process with segmentation masks
                            masks = result.masks.data.cpu().numpy()
                            
                            for mask, box, conf in zip(masks, boxes, confidences):
                                # Resize mask to frame dimensions
                                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                                binary_mask = cv2.resize(binary_mask, (detection_frame.shape[1], detection_frame.shape[0]))
                                
                                # Apply colored overlay only where mask exists (blue for potholes)
                                mask_indices = binary_mask > 0
                                detection_frame[mask_indices] = cv2.addWeighted(
                                    detection_frame[mask_indices], 0.7, 
                                    np.full_like(detection_frame[mask_indices], (255, 0, 0)), 0.3, 0
                                )
                                
                                # Calculate dimensions
                                dimensions = calculate_pothole_dimensions(binary_mask)
                                
                                # Calculate depth metrics if depth map is available
                                depth_metrics = None
                                if depth_map is not None:
                                    try:
                                        depth_metrics = calculate_real_depth(binary_mask, depth_map)
                                        if depth_metrics:
                                            logger.debug(f"✅ Depth calculated successfully: max={depth_metrics['max_depth_cm']}cm, avg={depth_metrics['avg_depth_cm']}cm")
                                        else:
                                            logger.warning("⚠️ Depth calculation returned None - using default values")
                                            depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                                    except Exception as e:
                                        logger.warning(f"⚠️ Error calculating depth: {str(e)}")
                                        depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                                else:
                                    logger.warning("⚠️ No depth map available - using default values")
                                    depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                                
                                # Get detection box coordinates
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Calculate volume and range if we have dimensions
                                if dimensions and depth_metrics:
                                    volume = dimensions["area_cm2"] * depth_metrics["max_depth_cm"]
                                    
                                    # Determine volume range
                                    if volume < 1000:
                                        volume_range = "Small (<1k)"
                                    elif volume < 10000:
                                        volume_range = "Medium (1k - 10k)"
                                    else:
                                        volume_range = "Big (>10k)"
                                    
                                    # Draw bounding box
                                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    # --- Draw label with ID and metrics ---
                                    label = f"ID {detection_id}, A:{dimensions['area_cm2']:.1f}cm^2, D:{depth_metrics['max_depth_cm']:.1f}cm, V:{volume:.1f}cm^3"
                                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                    cv2.rectangle(detection_frame, (x1, y1 - label_h - 6), (x1 + label_w, y1), (0, 255, 0), -1)
                                    cv2.putText(detection_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                    
                                    # Add detection to results with complete measurements
                                    all_detections.append({
                                        'type': 'Pothole',
                                        'bbox': [x1, y1, x2, y2],
                                        'confidence': float(conf),
                                        'frame': frame_count,
                                        'timestamp': frame_count / 30.0,
                                        'has_mask': True,
                                        'area_cm2': float(dimensions["area_cm2"]),
                                        'depth_cm': float(depth_metrics["max_depth_cm"]),
                                        'volume': float(volume),
                                        'volume_range': volume_range,
                                        'detection_id': detection_id
                                    })
                                    detection_id += 1
                                else:
                                    # Draw bounding box
                                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"ID {detection_id}, Pothole {conf:.2f}"
                                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                    cv2.rectangle(detection_frame, (x1, y1 - label_h - 6), (x1 + label_w, y1), (0, 255, 0), -1)
                                    cv2.putText(detection_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                    
                                    # Add detection to results with default values
                                    all_detections.append({
                                        'type': 'Pothole',
                                        'bbox': [x1, y1, x2, y2],
                                        'confidence': float(conf),
                                        'frame': frame_count,
                                        'timestamp': frame_count / 30.0,
                                        'has_mask': True,
                                        'area_cm2': 0.0,
                                        'depth_cm': 0.0,
                                        'volume': 0.0,
                                        'volume_range': 'Unknown',
                                        'detection_id': detection_id
                                    })
                                    detection_id += 1
                        else:
                            # Process only bounding boxes (fallback)
                            for box, conf in zip(boxes, confidences):
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Draw bounding box
                                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"ID {detection_id}, Pothole {conf:.2f}"
                                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                cv2.rectangle(detection_frame, (x1, y1 - label_h - 6), (x1 + label_w, y1), (0, 255, 0), -1)
                                cv2.putText(detection_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                
                                # Add detection to results without measurements
                                all_detections.append({
                                    'type': 'Pothole',
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(conf),
                                    'frame': frame_count,
                                    'timestamp': frame_count / 30.0,
                                    'has_mask': False,
                                    'area_cm2': 0.0,
                                    'depth_cm': 0.0,
                                    'volume': 0.0,
                                    'volume_range': 'Unknown',
                                    'detection_id': detection_id
                                })
                                detection_id += 1
            
            elif model_key == "cracks":
                # Handle crack detection (segmentation masks)
                CRACK_TYPES = {
                    0: {"name": "Alligator Crack", "color": (0, 0, 255)},
                    1: {"name": "Edge Crack", "color": (0, 255, 255)},
                    2: {"name": "Hairline Cracks", "color": (255, 0, 0)},
                    3: {"name": "Longitudinal Cracking", "color": (0, 255, 0)},
                    4: {"name": "Transverse Cracking", "color": (128, 0, 128)}
                }
                
                for result in results:
                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        for mask, box, cls, conf in zip(masks, boxes, classes, confidences):
                            # Resize mask to frame dimensions
                            binary_mask = (mask > 0.5).astype(np.uint8) * 255
                            binary_mask = cv2.resize(binary_mask, (detection_frame.shape[1], detection_frame.shape[0]))
                            
                            # Get crack type
                            crack_type = CRACK_TYPES.get(int(cls), {"name": "Unknown Crack", "color": (128, 128, 128)})
                            
                            # Create colored mask overlay
                            colored_mask = np.zeros_like(detection_frame)
                            colored_mask[binary_mask > 0] = crack_type["color"]
                            
                            # Apply mask with transparency
                            detection_frame = cv2.addWeighted(detection_frame, 0.7, colored_mask, 0.3, 0)
                            
                            # Calculate area
                            area_data = calculate_area(binary_mask)
                            area_cm2 = area_data["area_cm2"] if area_data else 0
                            
                            # Determine area range
                            if area_cm2 < 50:
                                area_range = "Small (<50 cm²)"
                            elif area_cm2 < 200:
                                area_range = "Medium (50-200 cm²)"
                            else:
                                area_range = "Large (>200 cm²)"
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), crack_type["color"], 2)
                            # --- Draw label with ID and metrics ---
                            label = f"ID {detection_id}, {crack_type['name']}, A:{area_cm2:.1f}cm^2"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(detection_frame, (x1, y1 - label_h - 6), (x1 + label_w, y1), crack_type["color"], -1)
                            cv2.putText(detection_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            # Add detection to results
                            all_detections.append({
                                'type': crack_type['name'],
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'frame': frame_count,
                                'timestamp': frame_count / 30.0,
                                'has_mask': True,
                                'area_cm2': float(area_cm2),
                                'area_range': area_range,
                                'detection_id': detection_id,
                                # 'coordinates': coordinates,
                                # 'username': username,
                                # 'role': role
                            })
                            detection_id += 1
            
            elif model_key == "kerbs":
                # Handle kerb detection (bounding boxes)
                kerb_types = {
                    0: {"name": "Damaged Kerbs", "color": (0, 0, 255)},
                    1: {"name": "Faded Kerbs", "color": (0, 165, 255)},
                    2: {"name": "Normal Kerbs", "color": (0, 255, 0)}
                }
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        for box, cls, conf in zip(boxes, classes, confidences):
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Get kerb type
                            kerb_type = kerb_types.get(int(cls), {"name": "Unknown Kerb", "color": (128, 128, 128)})
                            
                            # Calculate approximate length from bounding box
                            box_width = x2 - x1
                            box_height = y2 - y1
                            # Use the maximum dimension as approximation of length
                            length_pixels = max(box_width, box_height)
                            # Convert to meters (rough approximation, 1 pixel ≈ 0.01 meters)
                            length_m = length_pixels * 0.01
                            
                            # Draw bounding box
                            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), kerb_type["color"], 2)
                            # --- Draw label with ID and metrics ---
                            label = f"ID {detection_id}, {kerb_type['name']}, L:{length_m:.2f}m"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(detection_frame, (x1, y1 - label_h - 6), (x1 + label_w, y1), kerb_type["color"], -1)
                            cv2.putText(detection_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            # Add detection to results
                            all_detections.append({
                                'type': kerb_type['name'],
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'frame': frame_count,
                                'timestamp': frame_count / 30.0,
                                'kerb_type': 'Concrete Kerb',  # Default kerb type
                                'condition': kerb_type['name'],
                                'length_m': float(length_m),
                                'detection_id': detection_id,
                                # 'coordinates': coordinates,
                                # 'username': username,
                                # 'role': role
                            })
                            detection_id += 1
        
        # Apply tracking if tracker is provided
        if tracker:
            # Update tracker with current detections
            tracked_detections = tracker.update(all_detections, frame_count)
            
            # Use tracked detections for display (includes track IDs)
            display_detections = tracked_detections
        else:
            # No tracking - use raw detections
            display_detections = all_detections
        
        # --- Draw bounding boxes and labels using stable track_id ---
        for detection in display_detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection.get('track_id', None)
            defect_type = detection.get('type', '')
            color = (0, 255, 0)  # Default green
            label = f"ID {track_id if track_id is not None else ''}"
            # Compose label and color based on defect type
            if defect_type == 'Pothole':
                area = detection.get('area_cm2', 0)
                depth = detection.get('depth_cm', 0)
                volume = detection.get('volume', 0)
                label += f", A:{area:.1f}cm^2, D:{depth:.1f}cm, V:{volume:.1f}cm^3"
                color = (0, 255, 0)
            elif 'Crack' in defect_type:
                area = detection.get('area_cm2', 0)
                label += f", {defect_type}, A:{area:.1f}cm^2"
                color = (0, 0, 255) if 'Alligator' in defect_type else (0, 255, 255)
            elif 'Kerb' in defect_type:
                length = detection.get('length_m', 0)
                condition = detection.get('condition', '')
                label += f", {condition}, L:{length:.2f}m"
                color = (0, 0, 255) if 'Damaged' in condition else (0, 255, 0)
            else:
                conf = detection.get('confidence', 0)
                label += f", {defect_type} {conf:.2f}"
            # Draw bounding box
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), color, 2)
            # Draw label background and text
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(detection_frame, (x1, y1 - label_h - 6), (x1 + label_w, y1), color, -1)
            cv2.putText(detection_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Log processing time and performance metrics
        processing_time = time.time() - start_time
        logger.debug(f"Frame {frame_count} processed in {processing_time:.3f} seconds")
        
        # Log GPU memory usage if CUDA is being used
        if device.type == 'cuda':
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            logger.debug(f"GPU memory used: {gpu_memory_used:.1f} MB")
        
        return detection_frame, display_detections
        
    except Exception as e:
        logger.error(f"Error processing frame {frame_count}: {str(e)}")
        return frame, []


def process_pavement_video(video_path, selected_model, coordinates, video_timestamp=None, aws_folder=None, s3_folder=None, username=None, role=None, video_id=None, original_video_name=None):
    """Process pavement video and yield frames with detection results using CUDA optimization"""
    global video_processing_stop_flag
    
    try:
        # Reset stop flag
        video_processing_stop_flag = False
        
        # Get models and device info
        models, midas, midas_transform = get_models()
        device = get_device()
        
        if not models:
            yield "data: " + json.dumps({"success": False, "message": "Failed to load models"}) + "\n\n"
            return
        
        timestamp = datetime.now().isoformat()
        db = connect_to_db()
        
        # Determine which models will be run
        models_to_run = []
        if selected_model == "All":
            models_to_run = ["potholes", "cracks", "kerbs"]
        elif selected_model == "Potholes":
            models_to_run = ["potholes"]
        elif selected_model == "Alligator Cracks":
            models_to_run = ["cracks"]
        elif selected_model == "Kerbs":
            models_to_run = ["kerbs"]
        
        # Create initial video processing document if not already present
        if db is not None and video_id and not db.video_processing.find_one({"video_id": video_id}):
            video_doc = {
                "video_id": video_id,
                "original_video_url": None,  # Will be updated after S3 upload
                "processed_video_url": None,  # Will be updated after processing
                # s3_path removed; use only relative paths for video URLs
                "role": role,
                "username": username,
                "timestamp": timestamp,
                "models_run": models_to_run,
                "status": "processing",
                "model_outputs": {
                    "potholes": [],
                    "cracks": [],
                    "kerbs": []
                },
                "created_at": timestamp,
                "updated_at": timestamp
            }
            db.video_processing.insert_one(video_doc)
        
        # Log device and model information
        logger.info(f"Starting video processing on device: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Clear GPU cache before processing
            torch.cuda.empty_cache()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if db is not None:
                db.video_processing.update_one(
                    {"video_id": video_id},
                    {"$set": {"status": "failed", "error": "Could not open video file"}}
                )
            yield "data: " + json.dumps({"success": False, "message": "Could not open video file"}) + "\n\n"
            return
        
        # Get video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {frame_width}x{frame_height}, {frame_rate} FPS, {total_frames} frames")
        
        # Set up output video writer with timestamp-based naming
        if video_timestamp:
            output_filename = f"video_{video_timestamp}_processed.mp4"
        else:
            output_filename = f"processed_video_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join("processed_videos", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
        
        # Initialize variables
        frame_count = 0
        all_detections = []
        processing_times = []
        representative_frame = None
        representative_frame_detections = []
        max_detections_count = 0

        # Initialize tracking
        tracker = DefectTracker(max_missing_frames=30, confidence_threshold=0.3)

        # Performance monitoring
        start_time = time.time()
        
        while cap.isOpened() and not video_processing_stop_flag:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break
            
            frame_count += 1
            frame_start_time = time.time()
            
            # Process frame with tracking
            detection_frame, detections = process_video_frame_pavement(
                frame, frame_count, selected_model, models, midas, midas_transform, tracker
            )
            
            # Add detections to overall list (these include track IDs)
            all_detections.extend(detections)

            # Check if this frame has more detections than previous frames (for representative frame)
            current_detections_count = len(detections)
            if current_detections_count > max_detections_count:
                max_detections_count = current_detections_count
                # Store the frame as base64 for MongoDB storage
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                success, buffer = cv2.imencode('.jpg', detection_frame, encode_params)
                if success:
                    representative_frame = base64.b64encode(buffer).decode('utf-8')
                    representative_frame_detections = detections.copy()

            # If no detections found yet and this is the middle frame, use it as representative
            elif representative_frame is None and frame_count == total_frames // 2:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                success, buffer = cv2.imencode('.jpg', detection_frame, encode_params)
                if success:
                    representative_frame = base64.b64encode(buffer).decode('utf-8')
                    representative_frame_detections = detections.copy()

            # Write frame to output video
            out.write(detection_frame)

            # Track processing time
            frame_processing_time = time.time() - frame_start_time
            processing_times.append(frame_processing_time)
            
            # Convert frame to base64 for streaming (compress for better performance)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Reduce quality for faster streaming
            success, buffer = cv2.imencode('.jpg', detection_frame, encode_params)
            if success:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                logger.debug(f"Streaming frame {frame_count}/{total_frames}, Progress: {(frame_count / total_frames) * 100:.1f}%")
                
                # Calculate performance metrics
                avg_processing_time = np.mean(processing_times[-10:])  # Last 10 frames
                estimated_remaining = avg_processing_time * (total_frames - frame_count)
                
                # Send frame data with performance metrics
                frame_data = {
                    "frame": frame_base64,
                    "frame_count": frame_count,
                    "total_frames": total_frames,
                    "detections": detections,
                    "progress": (frame_count / total_frames) * 100,
                    "output_path": output_path,
                    "video_id": video_id,
                    "performance": {
                        "avg_frame_time": avg_processing_time,
                        "estimated_remaining": estimated_remaining,
                        "fps": 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
                        "device": str(device)
                    }
                }
                
                # Add GPU memory info if using CUDA
                if device.type == 'cuda':
                    frame_data["performance"]["gpu_memory_used"] = torch.cuda.memory_allocated() / 1024**2  # MB
                    frame_data["performance"]["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**2  # MB
                
                yield f"data: {json.dumps(frame_data)}\n\n"
            else:
                logger.warning(f"Failed to encode frame {frame_count}")
            
            # Periodic GPU memory cleanup
            if device.type == 'cuda' and frame_count % 50 == 0:
                torch.cuda.empty_cache()
        
        # Handle early stop case
        if video_processing_stop_flag:
            logger.info("Video processing stopped by user")
            unique_detections = tracker.get_unique_detections()
            stop_data = {
                "success": True,
                "message": "Processing stopped by user",
                "total_frames": frame_count,
                "all_detections": unique_detections,
                "frame_detections": all_detections,
                "total_unique_detections": len(unique_detections),
                "total_frame_detections": len(all_detections),
                "stopped_early": True,
                "output_path": output_path,
                "video_id": video_id
            }
            
            # Update MongoDB document with stopped status
            if db is not None:
                db.video_processing.update_one(
                    {"video_id": video_id},
                    {"$set": {
                        "status": "stopped",
                        "updated_at": datetime.now().isoformat()
                    }}
                )
            
            yield f"data: {json.dumps(stop_data)}\n\n"
        
        # Release resources
        cap.release()
        out.release()
        
        # Get unique detections from tracker (removes duplicates)
        unique_detections = tracker.get_unique_detections()
        
        # Calculate final performance metrics
        total_processing_time = time.time() - start_time
        avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        # Upload processed video to S3 if parameters provided
        processed_video_s3_url = None
        if aws_folder and s3_folder and video_timestamp:
            try:
                # Extract original name from the original video name for consistency
                if original_video_name:
                    original_name_part = original_video_name.replace(f"_{video_timestamp}.mp4", "")
                    processed_video_name = f"{original_name_part}_{video_timestamp}_processed.mp4"
                else:
                    processed_video_name = f"video_{video_timestamp}_processed.mp4"
                s3_key_processed = f"{s3_folder}/{processed_video_name}"
                upload_success, s3_url_or_error = upload_video_to_s3(output_path, aws_folder, s3_key_processed)
                if upload_success:
                    logger.info(f"Uploaded processed video to S3: {s3_url_or_error}")
                    # Store only the relative S3 path (role/username/video_xxx_processed.mp4)
                    processed_video_s3_url = s3_key_processed
                    # Update MongoDB document with processed video RELATIVE URL
                    # The full S3 URL should be constructed in the frontend/API consumer using a common base URL
                    if db is not None:
                        db.video_processing.update_one(
                            {"video_id": video_id},
                            {"$set": {
                                "processed_video_url": processed_video_s3_url,
                                "status": "completed",
                                "updated_at": datetime.now().isoformat()
                            }}
                        )
                    # Clean up local processed video after successful upload
                    try:
                        os.remove(output_path)
                        logger.info(f"Cleaned up local processed video: {output_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not remove local processed video {output_path}: {cleanup_error}")
                else:
                    logger.error(f"Failed to upload processed video to S3: {s3_url_or_error}")
                    if db is not None:
                        db.video_processing.update_one(
                            {"video_id": video_id},
                            {"$set": {
                                "status": "failed",
                                "error": f"Failed to upload processed video: {s3_url_or_error}",
                                "updated_at": datetime.now().isoformat()
                            }}
                        )
            except Exception as upload_error:
                logger.error(f"Error uploading processed video: {upload_error}")
                if db is not None:
                    db.video_processing.update_one(
                        {"video_id": video_id},
                        {"$set": {
                            "status": "failed",
                            "error": f"Error uploading processed video: {str(upload_error)}",
                            "updated_at": datetime.now().isoformat()
                        }}
                    )
        
        # Organize detections by model type
        model_outputs = {
            "potholes": [],
            "cracks": [],
            "kerbs": []
        }
        
        for detection in unique_detections:
            if "type" in detection:
                if detection["type"] == "Pothole":
                    model_outputs["potholes"].append(detection)
                elif "Crack" in detection["type"]:
                    model_outputs["cracks"].append(detection)
                elif "Kerb" in detection["type"]:
                    model_outputs["kerbs"].append(detection)
        
        # Update MongoDB document with model outputs and representative frame
        if db is not None:
            update_data = {
                "model_outputs": model_outputs,
                "updated_at": datetime.now().isoformat()
            }

            # Add representative frame if we captured one
            if representative_frame is not None:
                update_data["representative_frame"] = representative_frame
                update_data["representative_frame_detections"] = representative_frame_detections
                update_data["representative_frame_detection_count"] = len(representative_frame_detections)

            db.video_processing.update_one(
                {"video_id": video_id},
                {"$set": update_data}
            )
        
        # Send final results with performance summary
        final_data = {
            "success": True,
            "total_frames": frame_count,
            "all_detections": unique_detections,
            "frame_detections": all_detections,
            "total_unique_detections": len(unique_detections),
            "total_frame_detections": len(all_detections),
            "output_path": output_path,
            "processed_video_s3_url": processed_video_s3_url,
            "stopped_early": video_processing_stop_flag,
            "selected_model": selected_model,
            "coordinates": coordinates,
            "completed": True,
            "video_id": video_id,
            "performance_summary": {
                "total_time": total_processing_time,
                "avg_fps": avg_fps,
                "avg_frame_time": np.mean(processing_times) if processing_times else 0,
                "device": str(device)
            }
        }
        
        # Add final GPU memory info
        if device.type == 'cuda':
            final_data["performance_summary"]["final_gpu_memory"] = torch.cuda.memory_allocated() / 1024**2
        
        yield f"data: {json.dumps(final_data)}\n\n"
        
        # Send explicit end signal
        yield f"data: {json.dumps({'end': True})}\n\n"
        
        logger.info(f"Video processing completed: {frame_count} frames in {total_processing_time:.2f}s (avg {avg_fps:.1f} FPS)")
        
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}", exc_info=True)
        # Update MongoDB document with error status
        if 'db' in locals() and 'video_id' in locals():
            db.video_processing.update_one(
                {"video_id": video_id},
                {"$set": {
                    "status": "failed",
                    "error": str(e),
                    "updated_at": datetime.now().isoformat()
                }}
            )
        yield f"data: {json.dumps({'success': False, 'message': str(e)})}\n\n"
        # Send end signal even on error
        yield f"data: {json.dumps({'end': True})}\n\n"
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        video_processing_stop_flag = False
        
        # Clean up GPU memory if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Clean up temporary input video file
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up temporary video file: {video_path}")
        except Exception as e:
            logger.warning(f"Could not remove temporary file {video_path}: {e}")


def generate_timestamp_filename():
    """Generate a timestamp-based filename with conflict resolution"""
    base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check if file already exists and add suffix if needed
    counter = 0
    timestamp = base_timestamp
    temp_dir = os.path.dirname(__file__)
    
    while True:
        temp_path = os.path.join(temp_dir, f"video_{timestamp}.mp4")
        if not os.path.exists(temp_path):
            break
        counter += 1
        timestamp = f"{base_timestamp}_{counter:02d}"
        
        # Safety check to prevent infinite loop
        if counter > 99:
            timestamp = f"{base_timestamp}_{uuid.uuid4().hex[:8]}"
            break
    
    return timestamp


def cleanup_old_processed_videos(max_age_hours=24):
    """Clean up old processed video files that are older than max_age_hours"""
    try:
        processed_videos_dir = os.path.join(os.path.dirname(__file__), '..', 'processed_videos')
        if not os.path.exists(processed_videos_dir):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for filename in os.listdir(processed_videos_dir):
            if filename.endswith('.mp4'):
                file_path = os.path.join(processed_videos_dir, filename)
                try:
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old processed video: {filename}")
                except Exception as e:
                    logger.warning(f"Could not clean up file {filename}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old processed video files")
            
    except Exception as e:
        logger.warning(f"Error during processed videos cleanup: {e}")


# Call cleanup on module load to clean any leftover files
cleanup_old_processed_videos()

@pavement_bp.route('/detect-potholes', methods=['POST'])
def detect_potholes():
    """
    API endpoint to detect potholes in an uploaded image with CUDA optimization
    """
    try:
        # Get models and device info
        models, midas, midas_transform = get_models()
        device = get_device()
        
        print("\n=== Starting Pothole Detection ===")
        print(f"🔧 Using device: {device}")
        
        if not models or "potholes" not in models:
            return jsonify({
                "success": False,
                "message": "Failed to load pothole detection model"
            }), 500
        
        # Check MiDaS availability
        if midas is None or midas_transform is None:
            print("⚠️ MiDaS model not available - attempting to reload...")
            try:
                midas, midas_transform = load_midas()
                if midas is None or midas_transform is None:
                    print("❌ MiDaS reload failed - depth estimation will use default values")
                else:
                    print("✅ MiDaS reload successful")
                    midas.eval()
            except Exception as e:
                print(f"❌ Error reloading MiDaS: {str(e)}")
        else:
            print("✅ MiDaS model is available")
        
        if 'image' not in request.json:
            return jsonify({
                "success": False,
                "message": "No image data provided"
            }), 400

        # Get image data and validate it
        image_data = request.json['image']
        is_valid, error_message = validate_base64_image(image_data, 'pothole_detection')
        if not is_valid:
            logger.warning(f"Image validation failed for pothole detection: {error_message}")
            return jsonify({
                "success": False,
                "message": error_message
            }), 400

        # Extract coordinates if provided
        client_coordinates = request.json.get('coordinates', 'Not Available')

        # Get user information
        username = request.json.get('username', 'Unknown')
        role = request.json.get('role', 'Unknown')

        # Check if road classification should be skipped
        skip_road_classification = request.json.get('skip_road_classification', False)
        image = decode_base64_image(image_data)
        
        if image is None:
            return jsonify({
                "success": False,
                "message": "Invalid image data"
            }), 400
        
        print("✅ Image decoded successfully")
        print(f"📊 Image shape: {image.shape}")
        
        # Try to extract EXIF GPS data from the image
        lat, lon = get_gps_coordinates(image_data)
        coordinates = format_coordinates(lat, lon) if lat and lon else client_coordinates
        
        # Process the image
        processed_image = image.copy()

        # Classify the image to check if it contains a road (unless skipped)
        if skip_road_classification:
            # Skip classification and assume it's a road
            classification_result = {"is_road": True, "confidence": 1.0, "class_name": "skipped"}
        else:
            # Perform road classification with lower confidence threshold
            classification_result = classify_road_image(processed_image, models, confidence_threshold=0.4)

            # If no road detected, return classification info without processing
            if not classification_result["is_road"]:
                return jsonify({
                    "success": True,
                    "processed": False,
                    "message": "No road detected in the image. Image not processed.",
                    "classification": classification_result,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "potholes": [],
                    "processed_image": None
                }), 200
        
        # Run depth estimation if MiDaS is available
        depth_map = None
        if midas and midas_transform:
            try:
                print("🔄 Running depth estimation...")
                depth_map = estimate_depth(processed_image, midas, midas_transform)
                if depth_map is not None:
                    print(f"✅ Depth map generated successfully - shape: {depth_map.shape}")
                    print(f"📊 Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")
                else:
                    print("❌ Depth map generation failed - got None result")
            except Exception as e:
                print(f"❌ Error during depth estimation: {str(e)}")
                depth_map = None
        else:
            print("⚠️ MiDaS not available - skipping depth estimation")
        
        # Detect potholes with CUDA optimization and proper dtype handling
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Ensure image is in the correct format for the model
            if processed_image.shape[2] == 3:
                inference_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                inference_image = processed_image
            
            # Run inference with proper error handling
            try:
                results = models["potholes"](inference_image, conf=0.2, device=device)
            except RuntimeError as e:
                if "dtype" in str(e):
                    print(f"⚠️ Dtype error detected: {e}")
                    print("🔄 Attempting inference with CPU fallback...")
                    results = models["potholes"](inference_image, conf=0.2, device='cpu')
                else:
                    raise e
        
        # Process results
        pothole_results = []
        pothole_id = 1
        timestamp = pd.Timestamp.now().isoformat()
        image_upload_id = str(uuid.uuid4())
        
        # We'll upload images to S3 after processing
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    
                    for mask, box, conf in zip(masks, boxes, confidences):
                        binary_mask = (mask > 0.5).astype(np.uint8) * 255
                        binary_mask = cv2.resize(binary_mask, (processed_image.shape[1], processed_image.shape[0]))
                        
                        # Apply colored overlay
                        mask_indices = binary_mask > 0
                        processed_image[mask_indices] = cv2.addWeighted(
                            processed_image[mask_indices], 0.7,
                            np.full_like(processed_image[mask_indices], (255, 0, 0)), 0.3, 0
                        )
                        
                        # Calculate dimensions
                        dimensions = calculate_pothole_dimensions(binary_mask)
                        
                        # Calculate depth metrics if depth map is available
                        depth_metrics = None
                        if depth_map is not None:
                            try:
                                depth_metrics = calculate_real_depth(binary_mask, depth_map)
                                if depth_metrics:
                                    print(f"✅ Depth calculated successfully: max={depth_metrics['max_depth_cm']}cm, avg={depth_metrics['avg_depth_cm']}cm")
                                else:
                                    print("⚠️ Depth calculation returned None - using default values")
                                    depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                            except Exception as e:
                                print(f"❌ Error calculating depth: {str(e)}")
                                depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                        else:
                            print("⚠️ No depth map available - using default values")
                            depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                        
                        if dimensions and depth_metrics:
                            x1, y1, x2, y2 = map(int, box[:4])
                            volume = dimensions["area_cm2"] * depth_metrics["max_depth_cm"]
                            
                            if volume < 1000:
                                volume_range = "Small (<1k)"
                            elif volume < 10000:
                                volume_range = "Medium (1k - 10k)"
                            else:
                                volume_range = "Big (>10k)"
                            
                            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text = f"ID {pothole_id}, A:{dimensions['area_cm2']:.1f}cm^2, D:{depth_metrics['max_depth_cm']:.1f}cm, V:{volume:.1f}cm^3"
                            cv2.putText(processed_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            pothole_info = {
                                "pothole_id": pothole_id,
                                "area_cm2": float(dimensions["area_cm2"]),
                                "depth_cm": float(depth_metrics["max_depth_cm"]),
                                "volume": float(volume),
                                "volume_range": volume_range,
                                "confidence": float(conf),
                                "coordinates": coordinates,
                                "username": username,
                                "role": role,
                                "has_mask": True
                            }
                            pothole_results.append(pothole_info)
                            pothole_id += 1
                else:
                    # Process bounding boxes only
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"ID {pothole_id}, Pothole, C:{conf:.2f}"
                        cv2.putText(processed_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        pothole_info = {
                            "pothole_id": pothole_id,
                            "area_cm2": 0.0,
                            "depth_cm": 0.0,
                            "volume": 0.0,
                            "volume_range": "Unknown",
                            "confidence": float(conf),
                            "coordinates": coordinates,
                            "username": username,
                            "role": role,
                            "has_mask": False
                        }
                        pothole_results.append(pothole_info)
                        pothole_id += 1
        
        # Use comprehensive S3-MongoDB integration workflow
        try:
            # Initialize the workflow manager
            workflow = ImageProcessingWorkflow()

            # Prepare metadata
            metadata = {
                'username': username,
                'role': role,
                'coordinates': coordinates,
                'timestamp': timestamp
            }

            # Execute complete workflow: S3 upload + MongoDB storage
            workflow_success, workflow_result = workflow.process_and_store_images(
                image, processed_image, metadata, pothole_results, 'pothole'
            )

            if not workflow_success:
                return jsonify({
                    "success": False,
                    "message": f"Failed to process and store images: {workflow_result}"
                }), 500

            # Extract S3 URLs from workflow result for response
            original_s3_url = workflow_result['original_s3_url']
            processed_s3_url = workflow_result['processed_s3_url']

            logger.info(f"✅ Complete workflow successful for pothole detection: {workflow_result['image_id']}")

        except Exception as e:
            logger.error(f"❌ Error in comprehensive workflow: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Error in image processing workflow: {str(e)}"
            }), 500
        
        # Return results
        return jsonify({
            "success": True,
            "processed": True,
            "classification": classification_result,
            "message": f"Detected {len(pothole_results)} potholes",
            "processed_image": encode_processed_image(processed_image),
            "potholes": pothole_results,
            "coordinates": coordinates,
            "username": username,
            "role": role
        })
        
    except Exception as e:
        print(f"❌ Critical error in pothole detection: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }), 500

@pavement_bp.route('/detect-cracks', methods=['POST'])
def detect_cracks():
    """
    API endpoint to detect cracks in an uploaded image using segmentation masks with CUDA optimization
    """
    models, _, _ = get_models()
    device = get_device()

    if not models or "cracks" not in models:
        return jsonify({
            "success": False,
            "message": "Failed to load crack detection model"
        }), 500

    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "message": "No image data provided"
        }), 400

    # Get image data and validate it
    image_data = request.json['image']
    is_valid, error_message = validate_base64_image(image_data, 'crack_detection')
    if not is_valid:
        logger.warning(f"Image validation failed for crack detection: {error_message}")
        return jsonify({
            "success": False,
            "message": error_message
        }), 400

    client_coordinates = request.json.get('coordinates', 'Not Available')
    username = request.json.get('username', 'Unknown')
    role = request.json.get('role', 'Unknown')
    skip_road_classification = request.json.get('skip_road_classification', False)

    try:
        image_data = request.json['image']
        image = decode_base64_image(image_data)

        if image is None:
            return jsonify({
                "success": False,
                "message": "Invalid image data"
            }), 400

        lat, lon = get_gps_coordinates(image_data)
        coordinates = format_coordinates(lat, lon) if lat and lon else client_coordinates
        processed_image = image.copy()

        # Classify the image to check if it contains a road (unless skipped)
        if skip_road_classification:
            # Skip classification and assume it's a road
            classification_result = {"is_road": True, "confidence": 1.0, "class_name": "skipped"}
        else:
            # Perform road classification with lower confidence threshold
            classification_result = classify_road_image(processed_image, models, confidence_threshold=0.4)

            # If no road detected, return classification info without processing
            if not classification_result["is_road"]:
                return jsonify({
                    "success": True,
                    "processed": False,
                    "message": "No road detected in the image. Image not processed.",
                    "classification": classification_result,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "cracks": [],
                    "processed_image": None
                }), 200

        # Run crack detection with CUDA optimization and proper dtype handling
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Ensure image is in the correct format for the model
            if processed_image.shape[2] == 3:
                inference_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                inference_image = processed_image
            
            # Run inference with proper error handling
            try:
                results = models["cracks"](inference_image, conf=0.2, device=device)
            except RuntimeError as e:
                if "dtype" in str(e):
                    print(f"⚠️ Crack model dtype error: {e}")
                    print("🔄 Attempting crack inference with CPU fallback...")
                    results = models["cracks"](inference_image, conf=0.2, device='cpu')
                else:
                    raise e

        CRACK_TYPES = {
            0: {"name": "Alligator Crack", "color": (0, 0, 255)},
            1: {"name": "Edge Crack", "color": (0, 255, 255)},
            2: {"name": "Hairline Cracks", "color": (255, 0, 0)},
            3: {"name": "Longitudinal Cracking", "color": (0, 255, 0)},
            4: {"name": "Transverse Cracking", "color": (128, 0, 128)}
        }

        crack_results = []
        crack_id = 1
        timestamp = pd.Timestamp.now().isoformat()
        image_upload_id = str(uuid.uuid4())

        # We'll upload images to S3 after processing

        current_detections = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for mask, box, cls, conf in zip(masks, boxes, classes, confidences):
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                binary_mask = cv2.resize(binary_mask, (processed_image.shape[1], processed_image.shape[0]))

                crack_type = CRACK_TYPES.get(int(cls), {"name": "Unknown", "color": (128, 128, 128)})

                current_detections.append({
                    "mask": binary_mask,
                    "box": box,
                    "type": crack_type,
                    "class_id": int(cls),
                    "confidence": float(conf)
                })

        processed_detections = []
        for det in current_detections:
            matched = False
            for existing in processed_detections:
                box1 = det["box"]
                box2 = existing["box"]
                xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
                xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
                inter_area = max(0, xB - xA) * max(0, yB - yA)
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                iou = inter_area / float(box1_area + box2_area - inter_area)

                if iou > 0.5 and det["class_id"] == existing["class_id"]:
                    combined_mask = cv2.bitwise_or(det["mask"], existing["mask"])
                    new_box = [
                        min(box1[0], box2[0]), min(box1[1], box2[1]),
                        max(box1[2], box2[2]), max(box1[3], box2[3])
                    ]
                    existing["mask"] = combined_mask
                    existing["box"] = new_box
                    matched = True
                    break
            if not matched:
                processed_detections.append(det)

        condition_counts = {v["name"]: 0 for v in CRACK_TYPES.values()}

        for det in processed_detections:
            area_data = calculate_area(det["mask"])
            area_cm2 = area_data["area_cm2"] if area_data else 0

            if area_cm2 < 50:
                area_range = "Small (<50 cm²)"
            elif area_cm2 < 200:
                area_range = "Medium (50-200 cm²)"
            else:
                area_range = "Large (>200 cm²)"

            x1, y1, x2, y2 = map(int, det["box"][:4])
            color = det["type"]["color"]

            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
            label = f"ID {crack_id}, {det['type']['name']} {det['confidence']:.2f}"
            cv2.putText(processed_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw segmentation mask overlay
            colored_mask = np.zeros_like(processed_image)
            for i in range(3):
                colored_mask[:, :, i] = color[i] * (det["mask"] > 0)
            processed_image = cv2.addWeighted(processed_image, 1.0, colored_mask, 0.4, 0)

            crack_results.append({
                "crack_id": crack_id,
                "crack_type": det["type"]["name"],
                "area_cm2": area_cm2,
                "area_range": area_range,
                "coordinates": coordinates,
                "confidence": det["confidence"],
                "username": username,
                "role": role
            })
            condition_counts[det["type"]["name"]] += 1
            crack_id += 1

        # Upload images to S3
        original_s3_url, processed_s3_url, upload_success, upload_error = upload_images_to_s3(
            image, processed_image, image_upload_id, role, username
        )

        if not upload_success:
            return jsonify({
                "success": False,
                "message": f"Failed to upload images to S3: {upload_error}"
            }), 500

        db = connect_to_db()
        if db is not None and crack_results:
            try:
                db.crack_images.insert_one({
                    "image_id": image_upload_id,
                    "timestamp": timestamp,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "original_image_s3_url": original_s3_url,
                    "processed_image_s3_url": processed_s3_url,
                    "crack_count": len(crack_results),
                    "cracks": crack_results,
                    "type_counts": condition_counts
                })
            except Exception as e:
                print(f"Error saving image data: {e}")

        encoded_image = encode_processed_image(processed_image)

        return jsonify({
            "success": True,
            "processed": True,
            "classification": classification_result,
            "message": f"Detected {len(crack_results)} cracks",
            "processed_image": encoded_image,
            "cracks": crack_results,
            "type_counts": condition_counts,
            "coordinates": coordinates,
            "username": username,
            "role": role
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }), 500


@pavement_bp.route('/detect-kerbs', methods=['POST'])
def detect_kerbs():
    """
    API endpoint to detect kerbs and assess their condition in an uploaded image with CUDA optimization
    """
    # Get models and device info
    models, _, _ = get_models()
    device = get_device()
    
    if not models or "kerbs" not in models:
        return jsonify({
            "success": False,
            "message": "Failed to load kerb detection model"
        }), 500
    
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "message": "No image data provided"
        }), 400

    # Get image data and validate it
    image_data = request.json['image']
    is_valid, error_message = validate_base64_image(image_data, 'kerb_detection')
    if not is_valid:
        logger.warning(f"Image validation failed for kerb detection: {error_message}")
        return jsonify({
            "success": False,
            "message": error_message
        }), 400

    # Extract coordinates if provided
    client_coordinates = request.json.get('coordinates', 'Not Available')

    # Get user information
    username = request.json.get('username', 'Unknown')
    role = request.json.get('role', 'Unknown')
    skip_road_classification = request.json.get('skip_road_classification', False)

    # Get image data
    try:
        image_data = request.json['image']
        image = decode_base64_image(image_data)
        
        if image is None:
            return jsonify({
                "success": False,
                "message": "Invalid image data"
            }), 400
            
        # Try to extract EXIF GPS data from the image
        lat, lon = get_gps_coordinates(image_data)
        coordinates = format_coordinates(lat, lon) if lat and lon else client_coordinates
            
        # Process the image
        processed_image = image.copy()

        # Classify the image to check if it contains a road (unless skipped)
        if skip_road_classification:
            # Skip classification and assume it's a road
            classification_result = {"is_road": True, "confidence": 1.0, "class_name": "skipped"}
        else:
            # Perform road classification with lower confidence threshold
            classification_result = classify_road_image(processed_image, models, confidence_threshold=0.4)

            # If no road detected, return classification info without processing
            if not classification_result["is_road"]:
                return jsonify({
                    "success": True,
                    "processed": False,
                    "message": "No road detected in the image. Image not processed.",
                    "classification": classification_result,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "kerbs": [],
                    "processed_image": None
                }), 200

        # Detect kerbs using YOLOv8 model with CUDA optimization and proper dtype handling
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Ensure image is in the correct format for the model
            if processed_image.shape[2] == 3:
                inference_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                inference_image = processed_image
            
            # Run inference with proper error handling
            try:
                results = models["kerbs"](inference_image, conf=0.5, device=device)
            except RuntimeError as e:
                if "dtype" in str(e):
                    print(f"⚠️ Kerb model dtype error: {e}")
                    print("🔄 Attempting kerb inference with CPU fallback...")
                    results = models["kerbs"](inference_image, conf=0.5, device='cpu')
                else:
                    raise e
        
        # Process results
        kerb_results = []
        kerb_id = 1
        timestamp = pd.Timestamp.now().isoformat()
        image_upload_id = str(uuid.uuid4())  # Create a unique ID for this image upload
        
        # We'll upload images to S3 after processing
        
        # Initialize condition counts
        condition_counts = {
            "Normal Kerbs": 0,
            "Faded Kerbs": 0,
            "Damaged Kerbs": 0
        }
        
        kerb_types = {
            0: {"name": "Damaged Kerbs", "color": (0, 0, 255)},   # Red (in BGR)
            1: {"name": "Faded Kerbs", "color": (0, 165, 255)},   # Orange
            2: {"name": "Normal Kerbs", "color": (0, 255, 0)}     # Green
        }
        
        for result in results:
            # For YOLO mask-based results (if available)
            if result.masks:
                for mask, cls, conf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
                    mask = np.array(mask, dtype=np.int32)
                    
                    # Create an overlay for the mask
                    overlay = processed_image.copy()
                    kerb_type = kerb_types[int(cls)]
                    color = kerb_type["color"]
                    cv2.fillPoly(overlay, [mask], color=color)
                    
                    # Blend overlay with original image
                    alpha = 0.3
                    processed_image = cv2.addWeighted(overlay, alpha, processed_image, 1 - alpha, 0)
                    
                    # Draw outline
                    cv2.polylines(processed_image, [mask], isClosed=True, color=color, thickness=2)
                    
                    # Calculate kerb length (approx. based on perimeter/2)
                    perimeter = cv2.arcLength(mask, True)
                    length_m = perimeter / 100  # Convert pixels to meters (approximate)
                    
                    # Draw ID and condition on the image
                    x, y = mask.mean(axis=0).astype(int)
                    text = f"ID {kerb_id}: {kerb_type['name']}"
                    cv2.putText(processed_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add to kerb results
                    kerb_info = {
                        "kerb_id": kerb_id,
                        "kerb_type": "Concrete Kerb",  # Default type since subtype data isn't available
                        "length_m": length_m,
                        "condition": kerb_type["name"],
                        "confidence": float(conf),
                        "coordinates": coordinates,
                        "username": username,
                        "role": role
                    }
                    
                    # Update condition counts
                    condition_counts[kerb_type["name"]] += 1
                    
                    kerb_results.append(kerb_info)
                    kerb_id += 1
            else:
                # For regular YOLO box-based results
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    x1, y1, x2, y2 = map(int, box[:4])
                    kerb_type = kerb_types[int(cls)]
                    color = kerb_type["color"]
                    
                    # Draw bounding box
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Calculate approximate length based on box dimensions
                    length_m = max((x2 - x1), (y2 - y1)) / 100  # Convert pixels to meters (approximate)
                    
                    # Label
                    text = f"ID {kerb_id}: {kerb_type['name']}"
                    cv2.putText(processed_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add to kerb results
                    kerb_info = {
                        "kerb_id": kerb_id,
                        "kerb_type": "Concrete Kerb",  # Default type
                        "length_m": length_m,
                        "condition": kerb_type["name"],
                        "confidence": float(conf),
                        "coordinates": coordinates,
                        "username": username,
                        "role": role
                    }
                    
                    # Update condition counts
                    condition_counts[kerb_type["name"]] += 1
                    
                    kerb_results.append(kerb_info)
                    kerb_id += 1
        
        # Upload images to S3
        original_s3_url, processed_s3_url, upload_success, upload_error = upload_images_to_s3(
            image, processed_image, image_upload_id, role, username
        )

        if not upload_success:
            return jsonify({
                "success": False,
                "message": f"Failed to upload images to S3: {upload_error}"
            }), 500

        # Store consolidated entry in the database
        db = connect_to_db()
        if db is not None and kerb_results:
            try:
                db.kerb_images.insert_one({
                    "image_id": image_upload_id,
                    "timestamp": timestamp,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "original_image_s3_url": original_s3_url,
                    "processed_image_s3_url": processed_s3_url,
                    "kerb_count": len(kerb_results),
                    "kerbs": kerb_results,
                    "condition_counts": condition_counts
                })
            except Exception as e:
                print(f"Error saving image data: {e}")
        
        # Encode the processed image
        encoded_image = encode_processed_image(processed_image)
        
        return jsonify({
            "success": True,
            "processed": True,
            "classification": classification_result,
            "message": f"Detected {len(kerb_results)} kerbs",
            "processed_image": encoded_image,
            "kerbs": kerb_results,
            "condition_counts": condition_counts,
            "coordinates": coordinates,
            "username": username,
            "role": role
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }), 500

@pavement_bp.route('/get-image/<image_id>', methods=['GET'])
def get_image(image_id):
    """
    API endpoint to retrieve an image from GridFS by its ID
    """
    try:
        # Connect to MongoDB and get GridFS instance
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
            
        fs = get_gridfs()
        
        # Check if image exists
        if not fs.exists(ObjectId(image_id)):
            return jsonify({
                "success": False,
                "message": "Image not found"
            }), 404
        
        # Get image file from GridFS
        file = fs.get(ObjectId(image_id))
        
        # Create a Flask response with the image data
        from flask import send_file, Response
        
        # Return the image as a response
        return Response(
            file.read(),
            mimetype=file.content_type,
            headers={"Content-Disposition": f"inline; filename={file.filename}"}
        )
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error retrieving image: {str(e)}"
        }), 500


@pavement_bp.route('/get-image-url/<path:s3_key>', methods=['GET'])
def get_image_url(s3_key):
    """
    API endpoint to generate a public S3 URL for an image
    """
    try:
        # Generate the public S3 URL
        image_url = generate_s3_url(s3_key)

        return jsonify({
            "success": True,
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating image URL: {str(e)}"
        }), 500


@pavement_bp.route('/get-s3-image/<path:s3_key>', methods=['GET'])
def get_s3_image(s3_key):
    """
    API endpoint to proxy S3 images through the backend
    This solves the issue of S3 bucket not being publicly accessible
    """
    try:
        # Add detailed logging for debugging
        logger.info(f"🔍 S3 Image Request - S3 Key: {s3_key}")

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )

        # Get AWS folder and bucket info
        aws_folder = os.environ.get('AWS_FOLDER', 'aispry-project/2024_Oct_YNMSafety_RoadSafetyAudit/audit/raisetech')
        aws_folder = aws_folder.strip('/')
        parts = aws_folder.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''

        # Smart S3 key handling to avoid path duplication
        if prefix and s3_key.startswith(prefix):
            # S3 key already contains the full path (e.g., from database with full path)
            full_s3_key = s3_key
            logger.info(f"🔍 S3 key already contains full path: {s3_key}")
        elif prefix and s3_key.startswith(f"{prefix}/"):
            # S3 key starts with prefix/ (another variation)
            full_s3_key = s3_key
            logger.info(f"🔍 S3 key already contains full path with slash: {s3_key}")
        else:
            # S3 key is relative, add prefix
            full_s3_key = f"{prefix}/{s3_key}" if prefix else s3_key
            logger.info(f"🔍 Added prefix to relative S3 key: {prefix}/{s3_key}")

        logger.info(f"🔍 S3 Configuration - Bucket: {bucket}, Prefix: {prefix}")
        logger.info(f"🔍 Full S3 Key: {full_s3_key}")

        # Fetch image from S3
        response = s3_client.get_object(Bucket=bucket, Key=full_s3_key)
        image_data = response['Body'].read()
        content_type = response.get('ContentType', 'image/jpeg')

        logger.info(f"✅ Successfully fetched image from S3 - Size: {len(image_data)} bytes, Content-Type: {content_type}")

        # Return image with proper headers
        from flask import Response
        return Response(
            image_data,
            mimetype=content_type,
            headers={
                'Cache-Control': 'public, max-age=3600',  # Cache for 1 hour
                'Content-Length': str(len(image_data))
            }
        )

    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"❌ S3 ClientError - Code: {error_code}, Message: {str(e)}")
        logger.error(f"❌ Failed S3 Key: {full_s3_key if 'full_s3_key' in locals() else s3_key}")

        if error_code == 'NoSuchKey':
            return jsonify({
                "success": False,
                "message": f"Image not found in S3: {full_s3_key if 'full_s3_key' in locals() else s3_key}"
            }), 404
        else:
            return jsonify({
                "success": False,
                "message": f"S3 error: {str(e)}"
            }), 500
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching S3 image: {str(e)}")
        logger.error(f"❌ Failed S3 Key: {full_s3_key if 'full_s3_key' in locals() else s3_key}")
        return jsonify({
            "success": False,
            "message": f"Error fetching S3 image: {str(e)}"
        }), 500


@pavement_bp.route('/potholes/list', methods=['GET'])
def list_potholes():
    """
    API endpoint to list all detected potholes
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        all_potholes = []
        
        # Get potholes from the old structure
        old_potholes = list(db.potholes.find({}, {
            "_id": 1,
            "pothole_id": 1,
            "area_cm2": 1,
            "depth_cm": 1,
            "volume": 1,
            "volume_range": 1,
            "coordinates": 1,
            "timestamp": 1,
            "original_image_id": 1,
            "processed_image_id": 1
        }))
        
        # Convert ObjectId to string for old potholes
        for pothole in old_potholes:
            pothole["_id"] = str(pothole["_id"])
            pothole["data_structure"] = "old"
            all_potholes.append(pothole)
        
        # Get potholes from the new structure
        image_entries = list(db.pothole_images.find({}))
        
        for image in image_entries:
            image_id = image["image_id"]
            timestamp = image["timestamp"]
            original_image_id = image["original_image_id"]
            processed_image_id = image["processed_image_id"]
            
            # Add each pothole from the image with the image data
            for pothole in image["potholes"]:
                pothole_data = {
                    **pothole,
                    "image_id": image_id,
                    "timestamp": timestamp,
                    "original_image_id": original_image_id,
                    "processed_image_id": processed_image_id,
                    "data_structure": "new"
                }
                all_potholes.append(pothole_data)
        
        # Sort all potholes by timestamp, newest first
        all_potholes.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return jsonify({
            "success": True,
            "potholes": all_potholes
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving potholes: {str(e)}"
        }), 500

@pavement_bp.route('/potholes/recent', methods=['GET'])
def get_recent_potholes():
    """
    API endpoint to list only the most recently detected potholes (from the most recent image)
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        # Check if we have data in the new pothole_images collection
        latest_image = db.pothole_images.find_one(
            {},
            sort=[("timestamp", -1)]  # Sort by timestamp descending
        )
        
        if latest_image:
            # Get all potholes from the most recent image using the new data structure
            image_data = {
                "image_id": latest_image.get("image_id"),
                "timestamp": latest_image.get("timestamp")
            }

            # Handle both old GridFS-based and new S3-based data structures
            if "original_image_id" in latest_image:
                image_data["original_image_id"] = latest_image["original_image_id"]
            if "processed_image_id" in latest_image:
                image_data["processed_image_id"] = latest_image["processed_image_id"]
            if "original_image_s3_url" in latest_image:
                image_data["original_image_s3_url"] = latest_image["original_image_s3_url"]
            if "processed_image_s3_url" in latest_image:
                image_data["processed_image_s3_url"] = latest_image["processed_image_s3_url"]

            return jsonify({
                "success": True,
                "potholes": latest_image["potholes"],
                "image_data": image_data
            })
        
        # Fallback to old structure if no data in pothole_images collection
        latest_pothole = db.potholes.find_one(
            {},
            {"timestamp": 1},
            sort=[("timestamp", -1)]  # Sort by timestamp descending
        )
        
        if not latest_pothole:
            return jsonify({
                "success": True,
                "potholes": []
            })
        
        latest_timestamp = latest_pothole["timestamp"]
        
        # Get all potholes with that timestamp (from the same image)
        potholes = list(db.potholes.find(
            {"timestamp": latest_timestamp},
            {
                "_id": 1,
                "pothole_id": 1,
                "area_cm2": 1,
                "depth_cm": 1,
                "volume": 1,
                "volume_range": 1,
                "coordinates": 1,
                "timestamp": 1,
                "original_image_id": 1,
                "processed_image_id": 1
            }
        ))
        
        # Convert ObjectId to string
        for pothole in potholes:
            pothole["_id"] = str(pothole["_id"])
        
        return jsonify({
            "success": True,
            "potholes": potholes
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving recent potholes: {str(e)}"
        }), 500

@pavement_bp.route('/cracks/list', methods=['GET'])
def list_cracks():
    """
    API endpoint to list all detected cracks
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        all_cracks = []
        
        # Get cracks from the old structure
        old_cracks = list(db.cracks.find({}, {
            "_id": 1,
            "crack_id": 1,
            "crack_type": 1,
            "area_cm2": 1,
            "area_range": 1,
            "coordinates": 1,
            "timestamp": 1,
            "original_image_id": 1,
            "processed_image_id": 1
        }))
        
        # Convert ObjectId to string for old cracks
        for crack in old_cracks:
            crack["_id"] = str(crack["_id"])
            crack["data_structure"] = "old"
            all_cracks.append(crack)
        
        # Get cracks from the new structure
        image_entries = list(db.crack_images.find({}))
        
        for image in image_entries:
            image_id = image["image_id"]
            timestamp = image["timestamp"]
            original_image_id = image["original_image_id"]
            processed_image_id = image["processed_image_id"]
            
            # Add each crack from the image with the image data
            for crack in image["cracks"]:
                crack_data = {
                    **crack,
                    "image_id": image_id,
                    "timestamp": timestamp,
                    "original_image_id": original_image_id,
                    "processed_image_id": processed_image_id,
                    "data_structure": "new"
                }
                all_cracks.append(crack_data)
        
        # Sort all cracks by timestamp, newest first
        all_cracks.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return jsonify({
            "success": True,
            "cracks": all_cracks
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving cracks: {str(e)}"
        }), 500

@pavement_bp.route('/cracks/recent', methods=['GET'])
def get_recent_cracks():
    """
    API endpoint to list only the most recently detected cracks (from the most recent image)
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        # Check if we have data in the new crack_images collection
        latest_image = db.crack_images.find_one(
            {},
            sort=[("timestamp", -1)]  # Sort by timestamp descending
        )
        
        if latest_image:
            # Get all cracks from the most recent image using the new data structure
            return jsonify({
                "success": True,
                "cracks": latest_image["cracks"],
                "image_data": {
                    "image_id": latest_image["image_id"],
                    "timestamp": latest_image["timestamp"],
                    "original_image_id": latest_image["original_image_id"],
                    "processed_image_id": latest_image["processed_image_id"],
                    "type_counts": latest_image.get("type_counts", {})
                }
            })
        
        # Fallback to old structure if no data in crack_images collection
        latest_crack = db.cracks.find_one(
            {},
            {"timestamp": 1},
            sort=[("timestamp", -1)]  # Sort by timestamp descending
        )
        
        if not latest_crack:
            return jsonify({
                "success": True,
                "cracks": []
            })
        
        latest_timestamp = latest_crack["timestamp"]
        
        # Get all cracks with that timestamp (from the same image)
        cracks = list(db.cracks.find(
            {"timestamp": latest_timestamp},
            {
                "_id": 1,
                "crack_id": 1,
                "crack_type": 1,
                "area_cm2": 1,
                "area_range": 1,
                "coordinates": 1,
                "timestamp": 1,
                "original_image_id": 1,
                "processed_image_id": 1
            }
        ))
        
        # Convert ObjectId to string
        for crack in cracks:
            crack["_id"] = str(crack["_id"])
        
        return jsonify({
            "success": True,
            "cracks": cracks
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving recent cracks: {str(e)}"
        }), 500

@pavement_bp.route('/kerbs/list', methods=['GET'])
def list_kerbs():
    """
    API endpoint to list all detected kerbs
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        all_kerbs = []
        
        # Get kerbs from the old structure
        old_kerbs = list(db.kerbs.find({}, {
            "_id": 1,
            "kerb_id": 1,
            "kerb_type": 1,
            "length_m": 1,
            "condition": 1,
            "coordinates": 1,
            "timestamp": 1,
            "original_image_id": 1,
            "processed_image_id": 1
        }))
        
        # Convert ObjectId to string for old kerbs
        for kerb in old_kerbs:
            kerb["_id"] = str(kerb["_id"])
            kerb["data_structure"] = "old"
            all_kerbs.append(kerb)
        
        # Get kerbs from the new structure
        image_entries = list(db.kerb_images.find({}))
        
        for image in image_entries:
            image_id = image["image_id"]
            timestamp = image["timestamp"]
            original_image_id = image["original_image_id"]
            processed_image_id = image["processed_image_id"]
            
            # Add each kerb from the image with the image data
            for kerb in image["kerbs"]:
                kerb_data = {
                    **kerb,
                    "image_id": image_id,
                    "timestamp": timestamp,
                    "original_image_id": original_image_id,
                    "processed_image_id": processed_image_id,
                    "data_structure": "new"
                }
                all_kerbs.append(kerb_data)
        
        # Sort all kerbs by timestamp, newest first
        all_kerbs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return jsonify({
            "success": True,
            "kerbs": all_kerbs
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving kerbs: {str(e)}"
        }), 500

@pavement_bp.route('/kerbs/recent', methods=['GET'])
def get_recent_kerbs():
    """
    API endpoint to list only the most recently detected kerbs (from the most recent image)
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        # Check if we have data in the new kerb_images collection
        latest_image = db.kerb_images.find_one(
            {},
            sort=[("timestamp", -1)]  # Sort by timestamp descending
        )
        
        if latest_image:
            # Get all kerbs from the most recent image using the new data structure
            return jsonify({
                "success": True,
                "kerbs": latest_image["kerbs"],
                "image_data": {
                    "image_id": latest_image["image_id"],
                    "timestamp": latest_image["timestamp"],
                    "original_image_id": latest_image["original_image_id"],
                    "processed_image_id": latest_image["processed_image_id"],
                    "condition_counts": latest_image.get("condition_counts", {})
                }
            })
        
        # Fallback to old structure if no data in kerb_images collection
        latest_kerb = db.kerbs.find_one(
            {},
            {"timestamp": 1},
            sort=[("timestamp", -1)]  # Sort by timestamp descending
        )
        
        if not latest_kerb:
            return jsonify({
                "success": True,
                "kerbs": []
            })
        
        latest_timestamp = latest_kerb["timestamp"]
        
        # Get all kerbs with that timestamp (from the same image)
        kerbs = list(db.kerbs.find(
            {"timestamp": latest_timestamp},
            {
                "_id": 1,
                "kerb_id": 1,
                "kerb_type": 1,
                "length_m": 1,
                "condition": 1,
                "coordinates": 1,
                "timestamp": 1,
                "original_image_id": 1,
                "processed_image_id": 1
            }
        ))
        
        # Convert ObjectId to string
        for kerb in kerbs:
            kerb["_id"] = str(kerb["_id"])
        
        return jsonify({
            "success": True,
            "kerbs": kerbs
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving recent kerbs: {str(e)}"
        }), 500

@pavement_bp.route('/images/list', methods=['GET'])
def list_all_images():
    """
    API endpoint to list all uploaded images with their metadata
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        all_images = []
        
        # Get pothole images
        pothole_images = list(db.pothole_images.find({}, {
            "_id": 1,
            "image_id": 1,
            "timestamp": 1,
            "coordinates": 1,
            "username": 1,
            "role": 1,
            "pothole_count": 1,
            "original_image_id": 1,
            "processed_image_id": 1
        }).sort([("timestamp", -1)]))
        
        # Add type information
        for image in pothole_images:
            image["_id"] = str(image["_id"])
            image["type"] = "pothole"
            all_images.append(image)
        
        # Get crack images
        crack_images = list(db.crack_images.find({}, {
            "_id": 1,
            "image_id": 1,
            "timestamp": 1,
            "coordinates": 1,
            "username": 1,
            "role": 1,
            "crack_count": 1,
            "original_image_id": 1,
            "processed_image_id": 1,
            "type_counts": 1
        }).sort([("timestamp", -1)]))
        
        # Add type information
        for image in crack_images:
            image["_id"] = str(image["_id"])
            image["type"] = "crack"
            image["defect_count"] = image.get("crack_count", 0)
            all_images.append(image)
        
        # Get kerb images
        kerb_images = list(db.kerb_images.find({}, {
            "_id": 1,
            "image_id": 1,
            "timestamp": 1,
            "coordinates": 1,
            "username": 1,
            "role": 1,
            "kerb_count": 1,
            "original_image_id": 1,
            "processed_image_id": 1,
            "condition_counts": 1
        }).sort([("timestamp", -1)]))
        
        # Add type information
        for image in kerb_images:
            image["_id"] = str(image["_id"])
            image["type"] = "kerb"
            image["defect_count"] = image.get("kerb_count", 0)
            all_images.append(image)
        
        # Sort all images by timestamp, newest first
        all_images.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return jsonify({
            "success": True,
            "images": all_images
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving images: {str(e)}"
        }), 500

@pavement_bp.route('/images/<image_id>', methods=['GET'])
def get_image_details(image_id):
    """
    API endpoint to get detailed information about a specific image by its ID,
    including all defects detected in the image
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        # Initialize combined image data
        combined_image_data = None
        image_type = None
        detected_defects = []
        
        # Check in all collections for this image_id
        pothole_image = db.pothole_images.find_one({"image_id": image_id})
        crack_image = db.crack_images.find_one({"image_id": image_id})
        kerb_image = db.kerb_images.find_one({"image_id": image_id})
        
        # If no image found in any collection
        if not pothole_image and not crack_image and not kerb_image:
            return jsonify({
                "success": False,
                "message": f"Image with ID {image_id} not found"
            }), 404
        
        # Use the first available image as base and combine defect data
        base_image = pothole_image or crack_image or kerb_image
        combined_image_data = {
            "_id": str(base_image["_id"]),
            "image_id": base_image["image_id"],
            "timestamp": base_image["timestamp"],
            "coordinates": base_image["coordinates"],
            "username": base_image.get("username", "Unknown"),
            "role": base_image.get("role", "Unknown"),
            "original_image_id": base_image.get("original_image_id"),
            "processed_image_id": base_image.get("processed_image_id"),
            "detection_type": base_image.get("detection_type", "unknown"),
            # Initialize all defect arrays
            "potholes": [],
            "cracks": [],
            "kerbs": [],
            "pothole_count": 0,
            "crack_count": 0,
            "kerb_count": 0,
            "type_counts": {},
            "condition_counts": {},
            "detected_defects": [],
            "multi_defect_image": False
        }
        
        # Add pothole data if present
        if pothole_image:
            detected_defects.append("potholes")
            combined_image_data["potholes"] = pothole_image.get("potholes", [])
            combined_image_data["pothole_count"] = pothole_image.get("pothole_count", 0)
            
        # Add crack data if present
        if crack_image:
            detected_defects.append("cracks")
            combined_image_data["cracks"] = crack_image.get("cracks", [])
            combined_image_data["crack_count"] = crack_image.get("crack_count", 0)
            combined_image_data["type_counts"] = crack_image.get("type_counts", {})
            
        # Add kerb data if present
        if kerb_image:
            detected_defects.append("kerbs")
            combined_image_data["kerbs"] = kerb_image.get("kerbs", [])
            combined_image_data["kerb_count"] = kerb_image.get("kerb_count", 0)
            combined_image_data["condition_counts"] = kerb_image.get("condition_counts", {})
        
        # Set combined metadata
        combined_image_data["detected_defects"] = detected_defects
        combined_image_data["multi_defect_image"] = len(detected_defects) > 1
        
        # Determine primary type (for backward compatibility)
        if pothole_image:
            image_type = "pothole"
        elif crack_image:
            image_type = "crack"
        elif kerb_image:
            image_type = "kerb"
        else:
            image_type = "unknown"
            
        return jsonify({
            "success": True,
            "image": combined_image_data,
            "type": image_type
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving image details: {str(e)}"
        }), 500 

@pavement_bp.route('/detect-all', methods=['POST'])
def detect_all():
    """
    API endpoint to detect all types of defects (potholes, cracks, kerbs) in an uploaded image with CUDA optimization
    
    CRITICAL FIX: Model Isolation Implementation
    ===========================================
    
    This function implements complete model isolation to prevent interference between
    the three AI models (potholes, cracks, kerbs) when processing the same image.
    
    Key Isolation Strategies:
    1. **Image Data Isolation**: Each model receives a completely independent copy 
       of the original image via fresh .copy() operations
    2. **Processing Independence**: Models run on separate image instances, preventing
       cross-contamination from preprocessing or inference modifications
    3. **Visualization Separation**: All overlays are applied to a separate display_image
       that is not used for inference, only for visualization
    4. **Memory Management**: Each model works with its own memory space without
       sharing variables or objects
    
    This ensures that the "All" option produces identical detection results as if 
    each model was run individually on the same clean image.
    """
    # Get models and device info
    models, midas, midas_transform = get_models()
    device = get_device()
    
    if not models or any(model_type not in models for model_type in ["potholes", "cracks", "kerbs"]):
        return jsonify({
            "success": False,
            "message": "Failed to load one or more detection models"
        }), 500
    
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "message": "No image data provided"
        }), 400

    # Get image data and validate it
    image_data = request.json['image']
    is_valid, error_message = validate_base64_image(image_data, 'all_defects_detection')
    if not is_valid:
        logger.warning(f"Image validation failed for all defects detection: {error_message}")
        return jsonify({
            "success": False,
            "message": error_message
        }), 400

    # Extract coordinates if provided
    client_coordinates = request.json.get('coordinates', 'Not Available')

    # Get user information
    username = request.json.get('username', 'Unknown')
    role = request.json.get('role', 'Unknown')
    skip_road_classification = request.json.get('skip_road_classification', False)

    # Get image data
    try:
        image_data = request.json['image']
        image = decode_base64_image(image_data)
        
        if image is None:
            return jsonify({
                "success": False,
                "message": "Invalid image data"
            }), 400
            
        # Try to extract EXIF GPS data from the image
        lat, lon = get_gps_coordinates(image_data)
        coordinates = format_coordinates(lat, lon) if lat and lon else client_coordinates
        
        # CRITICAL FIX: Create separate copies for each model to prevent interference
        # Keep original image unchanged for model processing
        original_image = image.copy()

        # Classify the image to check if it contains a road (unless skipped)
        if skip_road_classification:
            # Skip classification and assume it's a road
            classification_result = {"is_road": True, "confidence": 1.0, "class_name": "skipped"}
        else:
            # Perform road classification with lower confidence threshold
            classification_result = classify_road_image(original_image, models, confidence_threshold=0.4)

            # If no road detected, return classification info without processing
            if not classification_result["is_road"]:
                return jsonify({
                    "success": True,
                    "processed": False,
                    "message": "No road detected in the image. Image not processed.",
                    "classification": classification_result,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "potholes": [],
                    "cracks": [],
                    "kerbs": [],
                    "processed_image": None
                }), 200

        # Debug: Log image shape and hash for verification
        print(f"MODEL ISOLATION DEBUG: Original image shape: {original_image.shape}")
        original_hash = hash(original_image.tobytes())
        print(f"MODEL ISOLATION DEBUG: Original image hash: {original_hash}")

        # Run depth estimation if MiDaS is available (using original image)
        depth_map = None
        if midas and midas_transform:
            depth_map = estimate_depth(original_image, midas, midas_transform)
        else:
            print("MiDaS model not available, skipping depth estimation")
        
        # Initialize results containers
        all_results = {
            "potholes": [],
            "cracks": [],
            "kerbs": [],
            "model_errors": {},
            "processed_image": None,
            "success": True,
            "processed": True,
            "classification": classification_result,
            "coordinates": coordinates,
            "username": username,
            "role": role
        }
        
        timestamp = pd.Timestamp.now().isoformat()
        image_upload_id = str(uuid.uuid4())  # Create a unique ID for this image upload
        
        # We'll upload images to S3 after processing all detections
        
        # Track which models succeeded
        successful_models = []
        
        # Initialize visualization image for overlays (separate from inference)
        display_image = original_image.copy()
        print(f"MODEL ISOLATION DEBUG: Display image initialized with hash: {hash(display_image.tobytes())}")
        
        # === POTHOLE DETECTION ===
        try:
            # CRITICAL FIX: Use fresh copy of original image for pothole detection
            pothole_inference_image = original_image.copy()
            pothole_hash = hash(pothole_inference_image.tobytes())
            print(f"MODEL ISOLATION DEBUG: Pothole model receiving image with hash: {pothole_hash}")
            print(f"MODEL ISOLATION DEBUG: Pothole image matches original: {pothole_hash == original_hash}")
            
            # Run pothole detection with CUDA optimization and proper dtype handling
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Ensure image is in the correct format for the model
                if pothole_inference_image.shape[2] == 3:
                    pothole_inference_image = cv2.cvtColor(pothole_inference_image, cv2.COLOR_BGR2RGB)
                
                # Run inference with proper error handling
                try:
                    pothole_results = models["potholes"](pothole_inference_image, conf=0.2, device=device)
                except RuntimeError as e:
                    if "dtype" in str(e):
                        print(f"⚠️ Pothole model dtype error: {e}")
                        print("🔄 Attempting pothole inference with CPU fallback...")
                        pothole_results = models["potholes"](pothole_inference_image, conf=0.2, device='cpu')
                    else:
                        raise e
            pothole_id = 1
            
            for result in pothole_results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    # Check if segmentation masks are available
                    if result.masks is not None:
                        # Process with segmentation masks
                        masks = result.masks.data.cpu().numpy()
                        
                        for mask, box, conf in zip(masks, boxes, confidences):
                            # Process the segmentation mask
                            binary_mask = (mask > 0.5).astype(np.uint8) * 255
                            binary_mask = cv2.resize(binary_mask, (display_image.shape[1], display_image.shape[0]))
                            
                            # Apply colored overlay only where mask exists (blue for potholes)
                            mask_indices = binary_mask > 0
                            display_image[mask_indices] = cv2.addWeighted(
                                display_image[mask_indices], 0.7, 
                                np.full_like(display_image[mask_indices], (255, 0, 0)), 0.3, 0
                            )
                            
                            # Calculate dimensions
                            dimensions = calculate_pothole_dimensions(binary_mask)
                            
                            # Calculate depth metrics if depth map is available
                            depth_metrics = None
                            if depth_map is not None:
                                depth_metrics = calculate_real_depth(binary_mask, depth_map)
                            else:
                                # Provide default depth values when MiDaS is not available
                                depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}  # Default values
                            
                            if dimensions and depth_metrics:
                                # Get detection box coordinates
                                x1, y1, x2, y2 = map(int, box[:4])
                                volume = dimensions["area_cm2"] * depth_metrics["max_depth_cm"]
                                
                                # Determine volume range
                                if volume < 1000:
                                    volume_range = "Small (<1k)"
                                elif volume < 10000:
                                    volume_range = "Medium (1k - 10k)"
                                else:
                                    volume_range = "Big (>10k)"
                                
                                # Draw bounding box on display image
                                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # Add text with ID and measurements
                                text = f"ID {pothole_id}, A:{dimensions['area_cm2']:.1f}cm^2, D:{depth_metrics['max_depth_cm']:.1f}cm, V:{volume:.1f}cm^3"
                                cv2.putText(display_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Store pothole data in database
                                pothole_data = {
                                    "pothole_id": pothole_id,
                                    "area_cm2": dimensions["area_cm2"],
                                    "depth_cm": depth_metrics["max_depth_cm"],
                                    "volume": volume,
                                    "volume_range": volume_range,
                                    "confidence": float(conf),
                                    "coordinates": coordinates,
                                    "timestamp": timestamp,
                                    "bbox": [x1, y1, x2, y2],
                                    "username": username,
                                    "role": role,
                                    "image_upload_id": image_upload_id,
                                    "has_mask": True
                                }
                                
                                # Create a copy for JSON response (without ObjectId references)
                                pothole_data_for_response = {
                                    "pothole_id": pothole_id,
                                    "area_cm2": dimensions["area_cm2"],
                                    "depth_cm": depth_metrics["max_depth_cm"],
                                    "volume": volume,
                                    "volume_range": volume_range,
                                    "confidence": float(conf),
                                    "coordinates": coordinates,
                                    "bbox": [x1, y1, x2, y2],
                                    "has_mask": True
                                }
                                
                                # Insert into database
                                db = connect_to_db()
                                if db is not None:
                                    db.potholes.insert_one(pothole_data)
                                
                                all_results["potholes"].append(pothole_data_for_response)
                                pothole_id += 1
                    else:
                        # Process only bounding boxes (fallback)
                        for box, conf in zip(boxes, confidences):
                            x1, y1, x2, y2 = map(int, box[:4])
                            
                            # Draw bounding box on display image
                            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text = f"ID {pothole_id}, Pothole, C:{conf:.2f}"
                            cv2.putText(display_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Store pothole data in database (without detailed measurements)
                            pothole_data = {
                                "pothole_id": pothole_id,
                                "area_cm2": 0.0,  # Not available without mask
                                "depth_cm": 0.0,  # Not available without mask
                                "volume": 0.0,    # Not available without mask
                                "volume_range": "Unknown",
                                "confidence": float(conf),
                                "coordinates": coordinates,
                                "timestamp": timestamp,
                                "bbox": [x1, y1, x2, y2],
                                "username": username,
                                "role": role,
                                "image_upload_id": image_upload_id,
                                "has_mask": False
                            }
                            
                            # Create a copy for JSON response (without ObjectId references)
                            pothole_data_for_response = {
                                "pothole_id": pothole_id,
                                "area_cm2": 0.0,
                                "depth_cm": 0.0,
                                "volume": 0.0,
                                "volume_range": "Unknown",
                                "confidence": float(conf),
                                "coordinates": coordinates,
                                "bbox": [x1, y1, x2, y2],
                                "has_mask": False
                            }
                            
                            # Insert into database
                            db = connect_to_db()
                            if db is not None:
                                db.potholes.insert_one(pothole_data)
                            
                            all_results["potholes"].append(pothole_data_for_response)
                            pothole_id += 1
            
            successful_models.append("potholes")
            
        except Exception as e:
            print(f"Error in pothole detection: {str(e)}")
            all_results["model_errors"]["potholes"] = str(e)
        
        # === CRACK DETECTION ===
        try:
            # CRITICAL FIX: Use fresh copy of original image for crack detection
            crack_inference_image = original_image.copy()
            crack_hash = hash(crack_inference_image.tobytes())
            print(f"MODEL ISOLATION DEBUG: Crack model receiving image with hash: {crack_hash}")
            print(f"MODEL ISOLATION DEBUG: Crack image matches original: {crack_hash == original_hash}")
            
            # Run crack detection with CUDA optimization and proper dtype handling
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Ensure image is in the correct format for the model
                if crack_inference_image.shape[2] == 3:
                    crack_inference_image = cv2.cvtColor(crack_inference_image, cv2.COLOR_BGR2RGB)
                
                # Run inference with proper error handling
                try:
                    crack_results = models["cracks"](crack_inference_image, conf=0.2, device=device)
                except RuntimeError as e:
                    if "dtype" in str(e):
                        print(f"⚠️ Crack model dtype error: {e}")
                        print("🔄 Attempting crack inference with CPU fallback...")
                        crack_results = models["cracks"](crack_inference_image, conf=0.2, device='cpu')
                    else:
                        raise e
            crack_id = 1
            
            # Define crack types mapping (same as original code)
            CRACK_TYPES = {
                0: {"name": "Alligator Crack", "color": (0, 0, 255)},
                1: {"name": "Edge Crack", "color": (0, 255, 255)},
                2: {"name": "Hairline Cracks", "color": (255, 0, 0)},
                3: {"name": "Longitudinal Cracking", "color": (0, 255, 0)},
                4: {"name": "Transverse Cracking", "color": (128, 0, 128)}
            }
            
            # Track crack types
            type_counts = {v["name"]: 0 for v in CRACK_TYPES.values()}
            
            for result in crack_results:
                if result.masks is None:
                    continue
                    
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for mask, box, cls, conf in zip(masks, boxes, classes, confidences):
                    # Process the segmentation mask
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255
                    binary_mask = cv2.resize(binary_mask, (display_image.shape[1], display_image.shape[0]))
                    
                    # Create colored overlay for cracks (green) - apply to display image only
                    colored_mask = np.zeros_like(display_image)
                    colored_mask[:, :, 1] = binary_mask  # Green channel
                    
                    # Blend the display image with the mask (not the inference image)
                    display_image = cv2.addWeighted(display_image, 1.0, colored_mask, 0.4, 0)
                    
                    # Calculate area safely
                    area_data = calculate_area(binary_mask)
                    area_cm2 = area_data["area_cm2"] if area_data else 0
                    
                    # Determine area range
                    if area_cm2 < 50:
                        area_range = "Small (<50 cm²)"
                    elif area_cm2 < 200:
                        area_range = "Medium (50-200 cm²)"
                    else:
                        area_range = "Large (>200 cm²)"
                    
                    # Get crack type safely using dictionary lookup
                    crack_type_info = CRACK_TYPES.get(int(cls), {"name": "Unknown", "color": (128, 128, 128)})
                    crack_type = crack_type_info["name"]
                    type_counts[crack_type] += 1
                    
                    # Get detection box coordinates
                    x1, y1, x2, y2 = map(int, box[:4])

                    # Draw bounding box and text on display image
                    color = crack_type_info["color"]
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                    text = f"ID {crack_id}, {crack_type}, A:{area_cm2:.1f}cm^2"
                    cv2.putText(display_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Store crack data
                    crack_data = {
                        "crack_id": crack_id,
                        "crack_type": crack_type,
                        "area_cm2": round(area_cm2, 2),
                        "area_range": area_range,
                        "confidence": float(conf),
                        "coordinates": coordinates,
                        "timestamp": timestamp,
                        "bbox": [x1, y1, x2, y2],
                        "username": username,
                        "role": role,
                        "image_upload_id": image_upload_id
                    }
                    
                    # Create a copy for JSON response (without ObjectId references)
                    crack_data_for_response = {
                        "crack_id": crack_id,
                        "crack_type": crack_type,
                        "area_cm2": round(area_cm2, 2),
                        "area_range": area_range,
                        "confidence": float(conf),
                        "coordinates": coordinates,
                        "bbox": [x1, y1, x2, y2]
                    }
                        
                    # Insert into database
                    db = connect_to_db()
                    if db is not None:
                        db.cracks.insert_one(crack_data)
                        
                    all_results["cracks"].append(crack_data_for_response)
                    crack_id += 1
            
            all_results["type_counts"] = type_counts
            successful_models.append("cracks")
            
        except Exception as e:
            print(f"Error in crack detection: {str(e)}")
            all_results["model_errors"]["cracks"] = str(e)
        
        # === KERB DETECTION ===
        try:
            # CRITICAL FIX: Use fresh copy of original image for kerb detection
            kerb_inference_image = original_image.copy()
            kerb_hash = hash(kerb_inference_image.tobytes())
            print(f"MODEL ISOLATION DEBUG: Kerb model receiving image with hash: {kerb_hash}")
            print(f"MODEL ISOLATION DEBUG: Kerb image matches original: {kerb_hash == original_hash}")
            
            # Run kerb detection with CUDA optimization and proper dtype handling
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Ensure image is in the correct format for the model
                if kerb_inference_image.shape[2] == 3:
                    kerb_inference_image = cv2.cvtColor(kerb_inference_image, cv2.COLOR_BGR2RGB)
                
                # Run inference with proper error handling
                try:
                    kerb_results = models["kerbs"](kerb_inference_image, conf=0.5, device=device)
                except RuntimeError as e:
                    if "dtype" in str(e):
                        print(f"⚠️ Kerb model dtype error: {e}")
                        print("🔄 Attempting kerb inference with CPU fallback...")
                        kerb_results = models["kerbs"](kerb_inference_image, conf=0.5, device='cpu')
                    else:
                        raise e
            kerb_id = 1
            
            # Define kerb types mapping (same as original code)
            kerb_types = {
                0: {"name": "Damaged Kerbs", "color": (0, 0, 255)},   # Red (in BGR)
                1: {"name": "Faded Kerbs", "color": (0, 165, 255)},   # Orange
                2: {"name": "Normal Kerbs", "color": (0, 255, 0)}     # Green
            }
            
            # Track kerb conditions
            condition_counts = {
                "Normal Kerbs": 0,
                "Faded Kerbs": 0,
                "Damaged Kerbs": 0
            }
            
            for result in kerb_results:
                # Handle both mask-based and box-based results
                if hasattr(result, 'masks') and result.masks is not None:
                    # Mask-based processing
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for mask, box, cls, conf in zip(masks, boxes, classes, confidences):
                        # Process the segmentation mask
                        binary_mask = (mask > 0.5).astype(np.uint8) * 255
                        binary_mask = cv2.resize(binary_mask, (display_image.shape[1], display_image.shape[0]))
                        
                        # Create colored overlay for kerbs (blue) - apply to display image only
                        colored_mask = np.zeros_like(display_image)
                        colored_mask[:, :, 0] = binary_mask  # Blue channel
                        
                        # Blend the display image with the mask (not the inference image)
                        display_image = cv2.addWeighted(display_image, 1.0, colored_mask, 0.4, 0)
                        
                        # Calculate length based on mask perimeter
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            perimeter = cv2.arcLength(contours[0], True)
                            estimated_length_m = perimeter / 100  # Convert pixels to meters (approximate)
                        else:
                            estimated_length_m = 1.0  # Default value
                        
                        # Get kerb condition safely
                        kerb_type_info = kerb_types.get(int(cls), {"name": "Unknown", "color": (128, 128, 128)})
                        kerb_condition = kerb_type_info["name"]
                        condition_counts[kerb_condition] += 1
                        
                        # Get detection box coordinates
                        x1, y1, x2, y2 = map(int, box[:4])

                        # Draw bounding box and text on display image
                        color = kerb_type_info["color"]
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                        text = f"ID {kerb_id}, {kerb_condition}, L:{estimated_length_m:.1f}m"
                        cv2.putText(display_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Store kerb data
                        kerb_data = {
                            "kerb_id": kerb_id,
                            "kerb_type": "Concrete Kerb",  # Default type
                            "length_m": estimated_length_m,
                            "condition": kerb_condition,
                            "confidence": float(conf),
                            "coordinates": coordinates,
                            "timestamp": timestamp,
                            "bbox": [x1, y1, x2, y2],
                            "username": username,
                            "role": role,
                            "image_upload_id": image_upload_id
                        }
                        
                        # Create a copy for JSON response (without ObjectId references)
                        kerb_data_for_response = {
                            "kerb_id": kerb_id,
                            "kerb_type": "Concrete Kerb",  # Default type
                            "length_m": estimated_length_m,
                            "condition": kerb_condition,
                            "confidence": float(conf),
                            "coordinates": coordinates,
                            "bbox": [x1, y1, x2, y2]
                        }
                        
                        # Insert into database
                        db = connect_to_db()
                        if db is not None:
                            db.kerbs.insert_one(kerb_data)
                        
                        all_results["kerbs"].append(kerb_data_for_response)
                        kerb_id += 1
                else:
                    # Box-based processing (fallback)
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        for box, cls, conf in zip(boxes, classes, confidences):
                            x1, y1, x2, y2 = map(int, box[:4])
                            
                            # Calculate approximate length based on box dimensions
                            estimated_length_m = max((x2 - x1), (y2 - y1)) / 100  # Convert pixels to meters
                            
                            # Get kerb condition safely
                            kerb_type_info = kerb_types.get(int(cls), {"name": "Unknown", "color": (128, 128, 128)})
                            kerb_condition = kerb_type_info["name"]
                            condition_counts[kerb_condition] += 1

                            # Draw bounding box and text on display image
                            color = kerb_type_info["color"]
                            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                            text = f"ID {kerb_id}, {kerb_condition}, L:{estimated_length_m:.1f}m"
                            cv2.putText(display_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            # Store kerb data
                            kerb_data = {
                                "kerb_id": kerb_id,
                                "kerb_type": "Concrete Kerb",  # Default type
                                "length_m": estimated_length_m,
                                "condition": kerb_condition,
                                "confidence": float(conf),
                                "coordinates": coordinates,
                                "timestamp": timestamp,
                                "bbox": [x1, y1, x2, y2],
                                "username": username,
                                "role": role,
                                "image_upload_id": image_upload_id
                            }
                            
                            # Create a copy for JSON response (without ObjectId references)
                            kerb_data_for_response = {
                                "kerb_id": kerb_id,
                                "kerb_type": "Concrete Kerb",  # Default type
                                "length_m": estimated_length_m,
                                "condition": kerb_condition,
                                "confidence": float(conf),
                                "coordinates": coordinates,
                                "bbox": [x1, y1, x2, y2]
                            }
                            
                            # Insert into database
                            db = connect_to_db()
                            if db is not None:
                                db.kerbs.insert_one(kerb_data)
                            
                            all_results["kerbs"].append(kerb_data_for_response)
                            kerb_id += 1
            
            all_results["condition_counts"] = condition_counts
            successful_models.append("kerbs")
            
        except Exception as e:
            print(f"Error in kerb detection: {str(e)}")
            all_results["model_errors"]["kerbs"] = str(e)
        
        # Upload images to S3
        original_s3_url, processed_s3_url, upload_success, upload_error = upload_images_to_s3(
            original_image, display_image, image_upload_id, role, username
        )

        if not upload_success:
            return jsonify({
                "success": False,
                "message": f"Failed to upload images to S3: {upload_error}"
            }), 500
        
        # Encode processed image for response
        all_results["processed_image"] = encode_processed_image(display_image)
        
        # Final validation: Ensure original image integrity is maintained
        final_hash = hash(original_image.tobytes())
        print(f"MODEL ISOLATION DEBUG: Original image hash after all processing: {final_hash}")
        print(f"MODEL ISOLATION DEBUG: Original image integrity maintained: {final_hash == original_hash}")
        
        if final_hash != original_hash:
            print("⚠️  WARNING: Original image was modified during processing - this indicates a bug!")
        else:
            print("✅ SUCCESS: Original image integrity maintained throughout processing")
        
        # Smart categorization logic - Store image in appropriate categories based on detected defects
        db = connect_to_db()
        if db is not None:
            # Base image data for all categories
            image_data_base = {
                "image_id": image_upload_id,
                "timestamp": timestamp,
                "coordinates": coordinates,
                "username": username,
                "role": role,
                "original_image_s3_url": original_s3_url,
                "processed_image_s3_url": processed_s3_url,
                "detection_type": "all"
            }
            
            # Determine which defects were actually detected (non-zero counts)
            detected_defects = []
            
            # Check if potholes were detected
            if "potholes" in successful_models and len(all_results["potholes"]) > 0:
                detected_defects.append("potholes")
            
            # Check if cracks were detected
            if "cracks" in successful_models and len(all_results["cracks"]) > 0:
                detected_defects.append("cracks")
            
            # Check if kerbs were detected
            if "kerbs" in successful_models and len(all_results["kerbs"]) > 0:
                detected_defects.append("kerbs")
            
            # Store image in each category that has detected defects
            # This enables smart categorization where images appear in multiple categories
            # if they contain multiple defect types
            
            if "potholes" in detected_defects:
                pothole_image_data = image_data_base.copy()
                pothole_image_data.update({
                    "pothole_count": len(all_results["potholes"]),
                    "potholes": all_results["potholes"],  # Store individual pothole data
                    "detected_defects": detected_defects,  # Track all defects in this image
                    "multi_defect_image": len(detected_defects) > 1  # Flag for multi-defect images
                })
                db.pothole_images.insert_one(pothole_image_data)
            
            if "cracks" in detected_defects:
                crack_image_data = image_data_base.copy()
                crack_image_data.update({
                    "crack_count": len(all_results["cracks"]),
                    "cracks": all_results["cracks"],  # Store individual crack data
                    "type_counts": all_results.get("type_counts", {}),
                    "detected_defects": detected_defects,  # Track all defects in this image
                    "multi_defect_image": len(detected_defects) > 1  # Flag for multi-defect images
                })
                db.crack_images.insert_one(crack_image_data)
            
            if "kerbs" in detected_defects:
                kerb_image_data = image_data_base.copy()
                kerb_image_data.update({
                    "kerb_count": len(all_results["kerbs"]),
                    "kerbs": all_results["kerbs"],  # Store individual kerb data
                    "condition_counts": all_results.get("condition_counts", {}),
                    "detected_defects": detected_defects,  # Track all defects in this image
                    "multi_defect_image": len(detected_defects) > 1  # Flag for multi-defect images
                })
                db.kerb_images.insert_one(kerb_image_data)
            
            # Add categorization info to response
            all_results["categorization"] = {
                "detected_defects": detected_defects,
                "categories_stored": len(detected_defects),
                "multi_defect_image": len(detected_defects) > 1
            }
        
        # Determine overall success
        if not successful_models:
            all_results["success"] = False
            all_results["message"] = "All detection models failed"
        elif len(successful_models) < 3:
            # Partial success
            failed_models = set(["potholes", "cracks", "kerbs"]) - set(successful_models)
            all_results["message"] = f"Partial success. Failed models: {', '.join(failed_models)}"
        else:
            all_results["message"] = "All detection models completed successfully"
        
        return jsonify(all_results)
        
    except Exception as e:
        print(f"Error in detect_all: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Error during detection: {str(e)}"
        }), 500 


def upload_video_to_s3(local_path, aws_folder, s3_key):
    """
    Upload a video file to S3 at the specified bucket/key using put_object.

    Args:
        local_path: Local file path to upload
        aws_folder: AWS folder path (e.g., 'aispry-project/2024_Oct_YNMSafety_RoadSafetyAudit/audit/raisetech')
        s3_key: S3 key path (e.g., 'role/username/video_xxx.mp4')

    Returns:
        tuple: (success: bool, s3_url_or_error: str)
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )

        # Extract bucket and prefix from aws_folder
        aws_folder = aws_folder.strip('/')
        parts = aws_folder.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''

        # Compose full S3 key
        full_s3_key = f"{prefix}/{s3_key}" if prefix else s3_key

        logger.info(f"🔄 Uploading video to S3 - Bucket: {bucket}, Key: {full_s3_key}")

        # Check if local file exists
        if not os.path.exists(local_path):
            error_msg = f"Local video file not found: {local_path}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        # Upload video file to S3 using put_object
        with open(local_path, 'rb') as f:
            s3_client.put_object(
                Bucket=bucket,
                Key=full_s3_key,
                Body=f,
                ContentType='video/mp4'
            )

        logger.info(f"✅ Successfully uploaded video to S3: {full_s3_key}")
        return True, f's3://{bucket}/{full_s3_key}'

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = f"S3 upload error ({error_code}): {str(e)}"
        logger.error(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during video upload: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return False, error_msg


def upload_image_to_s3(image_buffer, aws_folder, s3_key, content_type='image/jpeg'):
    """
    Upload an image buffer to S3 at the specified bucket/key.

    Args:
        image_buffer: Image data as bytes (from cv2.imencode)
        aws_folder: AWS folder path (e.g., 'aispry-project/2024_Oct_YNMSafety_RoadSafetyAudit/audit/raisetech')
        s3_key: S3 key path (e.g., 'role/username/image_xxx_original.jpg')
        content_type: MIME type for the image

    Returns:
        tuple: (success: bool, s3_url_or_error: str)
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION')
        )

        # Extract bucket and prefix from aws_folder
        aws_folder = aws_folder.strip('/')
        parts = aws_folder.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''

        # Compose full S3 key
        full_s3_key = f"{prefix}/{s3_key}" if prefix else s3_key

        # Upload image buffer to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=full_s3_key,
            Body=image_buffer,
            ContentType=content_type
        )

        return True, f's3://{bucket}/{full_s3_key}'
    except ClientError as e:
        logger.error(f"S3 image upload error: {e}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Unexpected error during S3 image upload: {e}")
        return False, str(e)


def upload_images_to_s3(original_image, processed_image, image_upload_id, role, username):
    """
    Upload both original and processed images to S3 with organized folder structure.

    Args:
        original_image: Original image as numpy array
        processed_image: Processed image as numpy array
        image_upload_id: Unique identifier for this image upload
        role: User role for folder structure
        username: Username for folder structure

    Returns:
        tuple: (original_s3_url: str, processed_s3_url: str, success: bool, error_msg: str)
    """
    try:
        # Get AWS configuration
        aws_folder = os.environ.get('AWS_FOLDER', 'aispry-project/2024_Oct_YNMSafety_RoadSafetyAudit/audit/raisetech')

        # Encode images to JPEG format
        _, original_buffer = cv2.imencode('.jpg', original_image)
        _, processed_buffer = cv2.imencode('.jpg', processed_image)

        # Create S3 keys with organized folder structure
        # Structure: {AWS_FOLDER}/{role}/{username}/original/image_{id}.jpg
        #           {AWS_FOLDER}/{role}/{username}/processed/image_{id}.jpg
        original_s3_key = f"{role}/{username}/original/image_{image_upload_id}.jpg"
        processed_s3_key = f"{role}/{username}/processed/image_{image_upload_id}.jpg"

        # Upload original image
        original_success, original_result = upload_image_to_s3(
            original_buffer.tobytes(), aws_folder, original_s3_key
        )

        if not original_success:
            return None, None, False, f"Failed to upload original image: {original_result}"

        # Upload processed image
        processed_success, processed_result = upload_image_to_s3(
            processed_buffer.tobytes(), aws_folder, processed_s3_key
        )

        if not processed_success:
            return None, None, False, f"Failed to upload processed image: {processed_result}"

        logger.info(f"Successfully uploaded images to S3: {original_s3_key}, {processed_s3_key}")

        # Return the relative S3 paths (not full S3 URLs)
        return original_s3_key, processed_s3_key, True, None

    except Exception as e:
        logger.error(f"Error uploading images to S3: {e}")
        return None, None, False, str(e)


def generate_s3_url(s3_key, aws_folder=None):
    """
    Generate a public S3 URL from an S3 key.

    Args:
        s3_key: S3 key path (e.g., 'role/username/image_xxx_original.jpg')
        aws_folder: AWS folder path (optional, will use env var if not provided)

    Returns:
        str: Full S3 URL
    """
    if aws_folder is None:
        aws_folder = os.environ.get('AWS_FOLDER', 'aispry-project/2024_Oct_YNMSafety_RoadSafetyAudit/audit/raisetech')

    # Extract bucket and prefix from aws_folder
    aws_folder = aws_folder.strip('/')
    parts = aws_folder.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''

    # Compose full S3 key
    full_s3_key = f"{prefix}/{s3_key}" if prefix else s3_key

    # Generate public URL
    region = os.environ.get('AWS_REGION', 'us-east-1')
    return f"https://{bucket}.s3.{region}.amazonaws.com/{full_s3_key}"


def download_video_from_s3(s3_key, aws_folder=None):
    """
    Download a video file from S3 using get_object.

    Args:
        s3_key: S3 key path (e.g., 'role/username/video_xxx.mp4')
        aws_folder: AWS folder path (optional, will use env var if not provided)

    Returns:
        tuple: (success: bool, video_data_or_error: bytes/str, content_type: str)
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )

        if aws_folder is None:
            aws_folder = os.environ.get('AWS_FOLDER', 'aispry-project/2024_Oct_YNMSafety_RoadSafetyAudit/audit/raisetech')

        # Extract bucket and prefix from aws_folder
        aws_folder = aws_folder.strip('/')
        parts = aws_folder.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''

        # Compose full S3 key
        full_s3_key = f"{prefix}/{s3_key}" if prefix else s3_key

        logger.info(f"🔄 Downloading video from S3 - Bucket: {bucket}, Key: {full_s3_key}")

        # Check if object exists first
        try:
            s3_client.head_object(Bucket=bucket, Key=full_s3_key)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False, f"Video file not found in storage: {s3_key}", None
            elif error_code == '403':
                return False, f"Access denied to video file: {s3_key}", None
            else:
                return False, f"Storage access error ({error_code}): {s3_key}", None

        # Download the video using get_object
        response = s3_client.get_object(Bucket=bucket, Key=full_s3_key)
        video_data = response['Body'].read()

        # Always return video/mp4 content type regardless of what S3 returns
        # S3 sometimes stores videos as binary/octet-stream which causes download issues
        content_type = 'video/mp4'

        logger.info(f"✅ Successfully downloaded video from S3 - Size: {len(video_data)} bytes")
        logger.info(f"🎬 Forcing Content-Type to: {content_type}")
        return True, video_data, content_type

    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"❌ S3 ClientError downloading video: {e}")
        return False, f"Storage service error ({error_code})", None
    except Exception as e:
        logger.error(f"❌ Unexpected error downloading video from S3: {e}")
        return False, f"Video download error: {str(e)}", None


@pavement_bp.route('/detect-video', methods=['POST'])
def detect_video():
    """
    API endpoint to detect pavement defects in uploaded video and return SSE stream
    """
    logger.info("Received video detection request")
    # Get parameters from request
    coordinates = request.form.get('coordinates', 'Not Available')
    selected_model = request.form.get('selectedModel', 'All')
    # Extract user/role/id from session or request (like image endpoints)
    username = (
        session.get('username')
        or request.form.get('username')
        or (request.json.get('username') if request.is_json else None)
        or 'Unknown'
    )
    role = (
        session.get('role')
        or request.form.get('role')
        or (request.json.get('role') if request.is_json else None)
        or 'Unknown'
    )
    # Use username as id for S3 folder structure
    s3_role = role.capitalize() if role else 'UnknownRole'
    s3_id = username
    logger.info(f"Processing video with model: {selected_model}")
    logger.info(f"Coordinates: {coordinates}")
    try:
        # Check if we have video data
        if 'video' not in request.files or not request.files['video']:
            return jsonify({
                "success": False,
                "message": "No video file provided"
            }), 400

        video_file = request.files['video']
        logger.info(f"Processing video file: {video_file.filename}")

        # Validate video file
        is_valid, error_message = validate_upload_file(video_file, 'video')
        if not is_valid:
            logger.warning(f"Video validation failed: {error_message}")
            return jsonify({
                "success": False,
                "message": error_message
            }), 400
        
        # Generate timestamp-based filename with conflict resolution, preserving original name
        video_timestamp = generate_timestamp_filename()
        original_filename = video_file.filename or "video.mp4"
        # Clean the filename to remove any path separators and ensure it's safe
        original_filename = os.path.basename(original_filename)
        name_without_ext = os.path.splitext(original_filename)[0]
        # Create a unique filename that includes both timestamp and original name
        original_video_name = f"{name_without_ext}_{video_timestamp}.mp4"
        temp_video_path = os.path.join(os.path.dirname(__file__), original_video_name)
        logger.info(f"Saving video to temporary path: {temp_video_path} (original: {original_filename})")
        video_file.save(temp_video_path)
        
        # Get AWS configuration
        aws_folder = os.environ.get('AWS_FOLDER', 'LTA')
        s3_folder = f"{s3_role}/{s3_id}"
        
        # Generate a unique video_id here and pass it to process_pavement_video
        video_id = str(ObjectId())
        
        # Upload original video to S3 immediately
        original_video_s3_url = None
        try:
            s3_key_original = f"{s3_folder}/{original_video_name}"
            upload_success, s3_url_or_error = upload_video_to_s3(temp_video_path, aws_folder, s3_key_original)
            if upload_success:
                logger.info(f"Uploaded original video to S3: {s3_url_or_error}")
                # Store only the relative S3 path (role/username/video_xxx.mp4)
                original_video_s3_url = s3_key_original
                # Update MongoDB document with original video RELATIVE URL using video_id
                db = connect_to_db()
                if db is not None:
                    db.video_processing.update_one(
                        {"video_id": video_id},
                        {"$set": {"original_video_url": original_video_s3_url}}
                    )
            else:
                logger.error(f"Failed to upload original video to S3: {s3_url_or_error}")
        except Exception as e:
            logger.error(f"Error uploading original video: {e}")
        
        # Create initial video_processing document here
        timestamp = datetime.now().isoformat()
        models_to_run = []
        if selected_model == "All":
            models_to_run = ["potholes", "cracks", "kerbs"]
        elif selected_model == "Potholes":
            models_to_run = ["potholes"]
        elif selected_model == "Alligator Cracks":
            models_to_run = ["cracks"]
        elif selected_model == "Kerbs":
            models_to_run = ["kerbs"]
        db = connect_to_db()
        if db is not None:
            video_doc = {
                "video_id": video_id,
                "original_video_url": original_video_s3_url,
                "processed_video_url": None,
                "role": role,
                "username": username,
                "timestamp": timestamp,
                "models_run": models_to_run,
                "status": "processing",
                "model_outputs": {
                    "potholes": [],
                    "cracks": [],
                    "kerbs": []
                },
                "created_at": timestamp,
                "updated_at": timestamp
            }
            db.video_processing.insert_one(video_doc)
        
        # Process video and return SSE stream
        # The processed video will be uploaded to S3 automatically when processing completes
        sse_response = Response(
            stream_with_context(process_pavement_video(
                temp_video_path,
                selected_model,
                coordinates,
                video_timestamp,
                aws_folder,
                s3_folder,
                username,
                role,
                video_id,  # Pass video_id to processing function
                original_video_name  # Pass original video name for S3 naming
            )),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
        
        return sse_response
        
    except Exception as e:
        logger.error(f"Error during video processing setup: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500

@pavement_bp.route('/stop-video-processing', methods=['POST'])
def stop_video_processing():
    """
    API endpoint to stop video processing
    """
    global video_processing_stop_flag
    
    try:
        logger.info("Received request to stop video processing")
        
        # Set stop flag
        video_processing_stop_flag = True
        
        return jsonify({
            "success": True,
            "message": "Video processing stop signal sent"
        })
        
    except Exception as e:
        logger.error(f"Error stopping video processing: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@pavement_bp.route('/video-processing/<video_id>', methods=['GET'])
def get_video_processing_status(video_id):
    """
    API endpoint to get video processing status and results
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        # Find video processing document
        video_doc = db.video_processing.find_one({"video_id": video_id})
        if not video_doc:
            return jsonify({
                "success": False,
                "message": f"Video processing record not found for ID: {video_id}"
            }), 404
        
        # Convert ObjectId to string for JSON serialization
        video_doc["_id"] = str(video_doc["_id"])
        
        return jsonify({
            "success": True,
            "video_processing": video_doc
        })
        
    except Exception as e:
        logger.error(f"Error retrieving video processing status: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error retrieving video processing status: {str(e)}"
        }), 500


@pavement_bp.route('/get-s3-video/<video_id>/<video_type>', methods=['GET'])
def get_s3_video(video_id, video_type):
    """
    API endpoint to proxy S3 videos through the backend
    This solves the issue of S3 bucket not being publicly accessible
    video_type: 'original' or 'processed'
    """
    try:
        # Add detailed logging for debugging
        logger.info(f"🔍 S3 Video Request - Video ID: {video_id}, Type: {video_type}")

        # Validate video_type parameter
        if video_type not in ['original', 'processed']:
            return jsonify({
                "success": False,
                "message": "Invalid video type. Use 'original' or 'processed'"
            }), 400

        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500

        # Find video processing document
        from bson import ObjectId
        video_doc = None
        try:
            # Try to find by MongoDB ObjectId first
            video_doc = db.video_processing.find_one({"_id": ObjectId(video_id)})
        except Exception as e:
            logger.debug(f"Could not find by ObjectId: {e}")
            # Fallback to video_id field if ObjectId conversion fails
            video_doc = db.video_processing.find_one({"video_id": video_id})

        if not video_doc:
            logger.error(f"❌ Video processing record not found for ID: {video_id}")
            return jsonify({
                "success": False,
                "message": f"Video not found in storage. Please check if the video exists or contact support."
            }), 404

        logger.info(f"🔍 Found video document for user: {video_doc.get('username', 'unknown')}")

        # Get the appropriate S3 key from MongoDB
        s3_key = None
        if video_type == 'original':
            s3_key = video_doc.get('original_video_url')
        elif video_type == 'processed':
            s3_key = video_doc.get('processed_video_url')

        if not s3_key:
            logger.error(f"❌ {video_type.capitalize()} video URL not found in document")
            logger.error(f"❌ Available fields: {list(video_doc.keys())}")
            return jsonify({
                "success": False,
                "message": f"{video_type.capitalize()} video not found in storage. The video may not have been processed yet."
            }), 404

        logger.info(f"🔍 S3 Video Key from DB: {s3_key}")

        # Download video from S3 using helper function
        success, video_data_or_error, content_type = download_video_from_s3(s3_key)

        if not success:
            # video_data_or_error contains the error message
            error_message = video_data_or_error

            # Determine appropriate HTTP status code based on error type
            if "not found" in error_message.lower():
                status_code = 404
                user_message = f"Video not found in storage. The {video_type} video file may have been moved or deleted."
            elif "access denied" in error_message.lower():
                status_code = 403
                user_message = "Access denied to video storage. Please contact support."
            else:
                status_code = 500
                user_message = "Video download error. Please contact support."

            logger.error(f"❌ Video download failed: {error_message}")
            return jsonify({
                "success": False,
                "message": user_message
            }), status_code

        # video_data_or_error contains the actual video data
        video_data = video_data_or_error
        if content_type is None:
            content_type = 'video/mp4'

        # Extract original filename from S3 key or create a meaningful one
        if s3_key:
            original_filename = s3_key.split('/')[-1]
            if not original_filename.endswith('.mp4'):
                original_filename += '.mp4'
        else:
            original_filename = f"{video_type}_video_{video_id[:8]}.mp4"

        # Return video with proper headers for direct download
        from flask import Response
        return Response(
            video_data,
            mimetype='video/mp4',  # Force video/mp4 mimetype
            headers={
                'Content-Disposition': f'attachment; filename="{original_filename}"',
                'Content-Length': str(len(video_data)),
                'Content-Type': 'video/mp4',  # Explicit content type
                'Cache-Control': 'no-cache, no-store, must-revalidate',  # Prevent caching issues
                'Pragma': 'no-cache',  # HTTP/1.0 compatibility
                'Expires': '0',  # Prevent caching
                'Accept-Ranges': 'bytes',  # Enable range requests for video streaming
                'Content-Transfer-Encoding': 'binary',  # Ensure binary transfer
                'X-Content-Type-Options': 'nosniff',  # Prevent MIME type sniffing
                'Access-Control-Allow-Origin': '*',  # Allow CORS for frontend
                'Access-Control-Allow-Headers': 'Content-Type, Accept',
                'Access-Control-Allow-Methods': 'GET'
            }
        )

    except Exception as e:
        logger.error(f"❌ Unexpected error in get_s3_video: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Video download service error. Please contact support."
        }), 500

@pavement_bp.route('/debug-videos', methods=['GET'])
def debug_videos():
    """
    Debug endpoint to check what videos are in the database
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500

        # Get all video processing documents
        videos = list(db.video_processing.find({}).limit(10))

        # Convert ObjectId to string for JSON serialization
        for video in videos:
            if '_id' in video:
                video['_id'] = str(video['_id'])

        return jsonify({
            "success": True,
            "count": len(videos),
            "videos": videos
        })

    except Exception as e:
        logger.error(f"Error fetching debug videos: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error fetching debug videos: {str(e)}"
        }), 500

@pavement_bp.route('/video-processing/list', methods=['GET'])
def list_video_processing():
    """
    API endpoint to list all video processing records
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500

        # Get optional query parameters
        username = request.args.get('username')
        role = request.args.get('role')
        status = request.args.get('status')

        # Build query filter
        query = {}
        if username:
            query["username"] = username
        if role:
            query["role"] = role
        if status:
            query["status"] = status

        # Get all video processing records, sorted by timestamp descending
        video_docs = list(db.video_processing.find(
            query,
            sort=[("timestamp", -1)]
        ))
        
        # Convert ObjectId to string for JSON serialization
        for doc in video_docs:
            doc["_id"] = str(doc["_id"])
        
        return jsonify({
            "success": True,
            "video_processing_records": video_docs
        })
        
    except Exception as e:
        logger.error(f"Error listing video processing records: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error listing video processing records: {str(e)}"
        }), 500

@pavement_bp.route('/test-classification', methods=['POST'])
def test_classification():
    """Test route to check road classification only"""
    try:
        # Get image data
        image_data = request.json['image']
        image = decode_base64_image(image_data)

        if image is None:
            return jsonify({
                "success": False,
                "message": "Invalid image data"
            }), 400

        # Test classification with different thresholds
        results = {}
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            classification_result = classify_road_image(image, models, confidence_threshold=threshold)
            results[f"threshold_{threshold}"] = classification_result

        return jsonify({
            "success": True,
            "image_shape": image.shape,
            "classification_results": results
        })

    except Exception as e:
        print(f"❌ Error in test classification: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500