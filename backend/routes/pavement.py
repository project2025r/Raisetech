from fastapi import APIRouter, Request, Body, HTTPException, Depends, UploadFile, File, Form, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import base64
import os
import json
import traceback
from config.db import connect_to_db, get_gridfs
from utils.models import load_yolo_models, load_midas, estimate_depth, calculate_real_depth, calculate_pothole_dimensions, calculate_area, get_device, classify_road_image
from utils.exif_utils import get_gps_coordinates, format_coordinates, get_comprehensive_exif_data
from utils.video_metadata_utils import extract_video_metadata, get_video_gps_coordinates
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
from pydantic import BaseModel
from typing import Optional, List

# Import S3 URL generation function from dashboard
from routes.dashboard import generate_s3_url_for_dashboard
from botocore.exceptions import ClientError

# Import our comprehensive S3-MongoDB integration
from s3_mongodb_integration import ImageProcessingWorkflow, S3ImageManager, MongoDBImageManager

# Import file validation utilities
from utils.file_validation import validate_upload_file, validate_base64_image, get_context_specific_error_message

router = APIRouter()

class PotholeDetectionPayload(BaseModel):
    image: str
    coordinates: Optional[str] = 'Not Available'
    username: Optional[str] = 'Unknown'
    role: Optional[str] = 'Unknown'
    skip_road_classification: Optional[bool] = False
    deviceCoordinates: Optional[dict] = {}

class CrackDetectionPayload(BaseModel):
    image: str
    coordinates: Optional[str] = 'Not Available'
    username: Optional[str] = 'Unknown'
    role: Optional[str] = 'Unknown'
    skip_road_classification: Optional[bool] = False

class KerbDetectionPayload(BaseModel):
    image: str
    coordinates: Optional[str] = 'Not Available'
    username: Optional[str] = 'Unknown'
    role: Optional[str] = 'Unknown'
    skip_road_classification: Optional[bool] = False

def extract_media_metadata(media_data, media_type='image'):
    """
    Extract comprehensive metadata from uploaded media (image or video).

    Args:
        media_data: Base64 encoded media data
        media_type: 'image' or 'video'

    Returns:
        dict: Comprehensive metadata including EXIF, GPS, timestamp, etc.
    """
    try:
        if media_type == 'image':
            return get_comprehensive_exif_data(media_data)
        elif media_type == 'video':
            return extract_video_metadata(media_data)
        else:
            logger.warning(f"Unknown media type: {media_type}")
            return {}
    except Exception as e:
        logger.error(f"Error extracting metadata for {media_type}: {e}")
        return {}

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
    """Eagerly preload YOLO and MiDaS models when the server starts"""
    global models, midas, midas_transform

    print("\n=== Starting Model Preload ===")
    print(" Preloading pavement models on server startup...")
    
    # Load YOLO models
    models = load_yolo_models()
    for model in models.values():
        try:
            model.eval()
        except Exception as e:
            print(f" Warning: Could not set model to eval mode: {e}")
    
    # Load MiDaS model with proper error handling
    print("\n=== Loading MiDaS Model ===")
    try:
        midas, midas_transform = load_midas()
        if midas is None or midas_transform is None:
            print(" Failed to load MiDaS model - depth estimation will be unavailable")
        else:
            print(" MiDaS model loaded successfully")
            if hasattr(midas, 'eval'):
                midas.eval()
                print(" MiDaS model set to eval mode")
            
            # Verify MiDaS works with a test input
            device = get_device()
            try:
                print(" Testing MiDaS with dummy input...")
                dummy_input = torch.randn(1, 3, 384, 384).to(device)
                if device.type == 'cuda':
                    dummy_input = dummy_input.half()
                with torch.no_grad():
                    _ = midas(dummy_input)
                print(" MiDaS test inference successful")
            except Exception as e:
                print(f" MiDaS test inference failed: {str(e)}")
                midas, midas_transform = None, None
    except Exception as e:
        print(f" Error during MiDaS initialization: {str(e)}")
        midas, midas_transform = None, None
    
    print("\n=== Model Preload Status ===")
    print(f" YOLO Models: {len(models)} loaded")
    print(f" MiDaS: {'Available' if midas else 'Unavailable'}")
    print("=== Preload Complete ===\n")
    
    return models, midas, midas_transform

def get_models():
    """Return preloaded models"""
    global models, midas, midas_transform
    if models is None or midas is None:
        preload_models_on_startup()
    return models, midas, midas_transform

def decode_base64_image(base64_string):
    """Decode a base64 image to cv2 format with automatic AVIF to JPG conversion"""
    print(f" DEBUG: Base64 string prefix: {base64_string[:50]}...")

    # Check if this is an AVIF image and convert if necessary
    if 'data:image/avif;base64,' in base64_string or 'data:image/;base64,' in base64_string:
        print(" AVIF image detected, converting to JPG...")
        try:
            from utils.image_converter import convert_image_to_yolo_supported
            base64_string = convert_image_to_yolo_supported(base64_string)
            print(" AVIF successfully converted to JPG")
        except Exception as e:
            print(f" Error converting AVIF to JPG: {str(e)}")
            return None

    if 'base64,' in base64_string:
        header = base64_string.split('base64,')[0]
        print(f" DEBUG: Image header: {header}")
        base64_string = base64_string.split('base64,')[1]

    img_data = base64.b64decode(base64_string)
    print(f" DEBUG: Decoded image data size: {len(img_data)} bytes")

    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is not None:
        print(f" DEBUG: Decoded image shape: {img.shape}")
        print(f" DEBUG: Decoded image dtype: {img.dtype}")
        print(f" DEBUG: Decoded image min/max: {img.min()}/{img.max()}")
    else:
        print(" DEBUG: Failed to decode image!")

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
            logger.debug(" Running depth estimation with MiDaS...")
            depth_map = estimate_depth(original_frame, midas, midas_transform)
            if depth_map is not None:
                logger.debug(" Depth map generated successfully")
            else:
                logger.warning(" Depth map generation failed - got None result")
        except Exception as e:
            logger.warning(f" Error during depth estimation: {str(e)}")
            depth_map = None
    else:
        logger.warning(" MiDaS model not available or not needed for selected model")
    
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
                        print(f" Video processing dtype error for {model_key}: {e}")
                        print(f" Attempting video inference with CPU fallback for {model_key}...")
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
                                            logger.debug(f" Depth calculated successfully: max={depth_metrics['max_depth_cm']}cm, avg={depth_metrics['avg_depth_cm']}cm")
                                        else:
                                            logger.warning(" Depth calculation returned None - using default values")
                                            depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                                    except Exception as e:
                                        logger.warning(f" Error calculating depth: {str(e)}")
                                        depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                                else:
                                    logger.warning(" No depth map available - using default values")
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
                "coordinates": coordinates,  # Add coordinates to initial document
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
                    representative_frame = base64.b66encode(buffer).decode('utf-8')
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
                    {"$set": {"status": "stopped", "updated_at": datetime.now().isoformat()}}
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
        
        # Upload original video to S3
        original_video_s3_url = None
        if aws_folder and s3_folder and video_timestamp:
            try:
                workflow = ImageProcessingWorkflow()
                original_video_name = f"video_{video_timestamp}.mp4"
                s3_key_original = f"{s3_folder}/{original_video_name}"
                upload_success, s3_url_or_error = workflow.s3_manager.upload_video_to_s3(video_path, aws_folder, s3_key_original)
                if upload_success:
                    logger.info(f"Uploaded original video to S3: {s3_url_or_error}")
                    original_video_s3_url = s3_key_original
                    if db is not None:
                        db.video_processing.update_one(
                            {"video_id": video_id},
                            {"$set": {"original_video_url": original_video_s3_url}}
                        )
                else:
                    logger.error(f"Failed to upload original video to S3: {s3_url_or_error}")
            except Exception as upload_error:
                logger.error(f"Error uploading original video: {upload_error}")

        # Upload processed video to S3 if parameters provided
        processed_video_s3_url = None
        if aws_folder and s3_folder and video_timestamp:
            try:
                workflow = ImageProcessingWorkflow()
                # Extract original name from the original video name for consistency
                if original_video_name:
                    original_name_part = original_video_name.replace(f"_{video_timestamp}.mp4", "")
                    processed_video_name = f"{original_name_part}_{video_timestamp}_processed.mp4"
                else:
                    processed_video_name = f"video_{video_timestamp}_processed.mp4"
                s3_key_processed = f"{s3_folder}/{processed_video_name}"
                upload_success, s3_url_or_error = workflow.s3_manager.upload_video_to_s3(output_path, aws_folder, s3_key_processed)
                if upload_success:
                    logger.info(f"Uploaded processed video to S3: {s3_url_or_error}")
                    # Store only the relative S3 path (role/username/video_xxx_processed.mp4)
                    processed_video_s3_url = s3_key_processed
                    # Update MongoDB document with processed video RELATIVE URL
                    # The full S3 URL should be constructed in the frontend/API consumer using a common base URL
                    if db is not None:
                        db.video_processing.update_one(
                            {"video_id": video_id},
                            {"$set": {"processed_video_url": processed_video_s3_url, "status": "completed", "updated_at": datetime.now().isoformat()}}
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
                            {"$set": {"status": "failed", "error": f"Failed to upload processed video: {s3_url_or_error}", "updated_at": datetime.now().isoformat()}}
                        )
            except Exception as upload_error:
                logger.error(f"Error uploading processed video: {upload_error}")
                if db is not None:
                    db.video_processing.update_one(
                        {"video_id": video_id},
                        {"$set": {"status": "failed", "error": f"Error uploading processed video: {str(upload_error)}", "updated_at": datetime.now().isoformat()}}
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
                {"$set": {"status": "failed", "error": str(e), "updated_at": datetime.now().isoformat()}}
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

@router.post('/detect-potholes')
def detect_potholes(payload: PotholeDetectionPayload):
    """API endpoint to detect potholes in an uploaded image with CUDA optimization"""
    try:
        # Get models and device info
        models, midas, midas_transform = get_models()
        device = get_device()
        
        print("\n=== Starting Pothole Detection ===")
        print(f" Using device: {device}")
        
        if not models or "potholes" not in models:
            raise HTTPException(status_code=500, detail="Failed to load pothole detection model")
        
        # Check MiDaS availability
        if midas is None or midas_transform is None:
            print(" MiDaS model not available - attempting to reload...")
            try:
                midas, midas_transform = load_midas()
                if midas is None or midas_transform is None:
                    print(" MiDaS reload failed - depth estimation will use default values")
                else:
                    print(" MiDaS reload successful")
                    midas.eval()
            except Exception as e:
                print(f" Error reloading MiDaS: {str(e)}")
        else:
            print(" MiDaS model is available")
        
        # Get image data and validate it
        image_data = payload.image
        is_valid, error_message = validate_base64_image(image_data, 'pothole_detection')
        if not is_valid:
            logger.warning(f"Image validation failed for pothole detection: {error_message}")
            raise HTTPException(status_code=400, detail=error_message)

        # Extract coordinates if provided
        client_coordinates = payload.coordinates

        # Get user information
        username = payload.username
        role = payload.role

        # Check if road classification should be skipped
        skip_road_classification = payload.skip_road_classification
        image = decode_base64_image(image_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        print(" Image decoded successfully")
        print(f" Image shape: {image.shape}")
        
        # Enhanced coordinate handling with device integration
        device_coordinates = payload.deviceCoordinates

        # Try to extract EXIF GPS data from the image
        lat, lon = get_gps_coordinates(image_data)
        exif_coordinates = format_coordinates(lat, lon) if lat is not None and lon is not None else None

        # Coordinate priority: EXIF GPS > Device GPS > Device IP > Client provided
        final_coordinates = None
        coordinate_source = "unknown"

        if exif_coordinates:
            final_coordinates = exif_coordinates
            coordinate_source = "exif_gps"
            logger.info(f" Using EXIF GPS coordinates: {final_coordinates}")
        elif device_coordinates and device_coordinates.get('source') == 'GPS':
            final_coordinates = device_coordinates.get('formatted', client_coordinates)
            coordinate_source = "device_gps"
            logger.info(f" Using device GPS coordinates: {final_coordinates}")
        elif device_coordinates and device_coordinates.get('source') == 'IP':
            final_coordinates = device_coordinates.get('formatted', client_coordinates)
            coordinate_source = "device_ip"
            logger.info(f" Using device IP coordinates: {final_coordinates}")
        else:
            final_coordinates = client_coordinates
            coordinate_source = "client_provided"
            logger.info(f" Using client-provided coordinates: {final_coordinates}")

        # Extract comprehensive EXIF metadata
        exif_metadata = extract_media_metadata(image_data, 'image')

        # Enhance metadata with device coordinate information
        if device_coordinates:
            exif_metadata['device_coordinates'] = device_coordinates
            exif_metadata['coordinate_source'] = coordinate_source

        print(f" Extracted EXIF metadata: {bool(exif_metadata)}")
        print(f" Final coordinates: {final_coordinates} (source: {coordinate_source})")
        
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
                return {
                    "success": True,
                    "processed": False,
                    "message": "No road detected in the image. Image not processed.",
                    "classification": classification_result,
                    "coordinates": final_coordinates,
                    "username": username,
                    "role": role,
                    "potholes": [],
                    "processed_image": None
                }
        
        # Run depth estimation if MiDaS is available
        depth_map = None
        if midas and midas_transform:
            try:
                print(" Running depth estimation...")
                depth_map = estimate_depth(processed_image, midas, midas_transform)
                if depth_map is not None:
                    print(f" Depth map generated successfully - shape: {depth_map.shape}")
                    print(f" Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")
                else:
                    print(" Depth map generation failed - got None result")
            except Exception as e:
                print(f" Error during depth estimation: {str(e)}")
                depth_map = None
        else:
            print(" MiDaS not available - skipping depth estimation")
        
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
                    print(f" Dtype error detected: {e}")
                    print(" Attempting inference with CPU fallback...")
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
                                    print(f" Depth calculated successfully: max={depth_metrics['max_depth_cm']}cm, avg={depth_metrics['avg_depth_cm']}cm")
                                else:
                                    print(" Depth calculation returned None - using default values")
                                    depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                            except Exception as e:
                                print(f" Error calculating depth: {str(e)}")
                                depth_metrics = {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}
                        else:
                            print(" No depth map available - using default values")
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
                                "coordinates": final_coordinates,
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
                            "coordinates": final_coordinates,
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

            # Prepare metadata including EXIF data and coordinate information
            metadata = {
                'username': username,
                'role': role,
                'coordinates': final_coordinates,
                'coordinate_source': coordinate_source,
                'device_coordinates': device_coordinates,
                'timestamp': timestamp,
                'exif_data': exif_metadata,
                'metadata': exif_metadata,
                'media_type': 'image'
            }

            # Execute complete workflow: S3 upload + MongoDB storage
            workflow_success, workflow_result = workflow.process_and_store_images(
                image, processed_image, metadata, pothole_results, 'pothole'
            )

            if not workflow_success:
                raise HTTPException(status_code=500, detail=f"Failed to process and store images: {workflow_result}")

            # Extract S3 URLs from workflow result for response
            original_s3_url = workflow_result['original_s3_url']
            processed_s3_url = workflow_result['processed_s3_url']

            logger.info(f" Complete workflow successful for pothole detection: {workflow_result['image_id']}")

        except Exception as e:
            logger.error(f" Error in comprehensive workflow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in image processing workflow: {str(e)}")
        
        # Return results
        return {
            "success": True,
            "processed": True,
            "classification": classification_result,
            "message": f"Detected {len(pothole_results)} potholes",
            "processed_image": encode_processed_image(processed_image),
            "potholes": pothole_results,
            "coordinates": final_coordinates,
            "coordinate_info": {
                "final_coordinates": final_coordinates,
                "coordinate_source": coordinate_source,
                "device_coordinates": device_coordinates,
                "exif_coordinates": exif_coordinates,
                "client_coordinates": client_coordinates
            },
            "username": username,
            "role": role
        }
        
    except Exception as e:
        print(f" Critical error in pothole detection: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post('/detect-cracks')
def detect_cracks(payload: CrackDetectionPayload):
    """API endpoint to detect cracks in an uploaded image using segmentation masks with CUDA optimization"""
    models, _, _ = get_models()
    device = get_device()

    if not models or "cracks" not in models:
        raise HTTPException(status_code=500, detail="Failed to load crack detection model")

    # Get image data and validate it
    image_data = payload.image
    is_valid, error_message = validate_base64_image(image_data, 'crack_detection')
    if not is_valid:
        logger.warning(f"Image validation failed for crack detection: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)

    client_coordinates = payload.coordinates
    username = payload.username
    role = payload.role
    skip_road_classification = payload.skip_road_classification

    try:
        image = decode_base64_image(image_data)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        lat, lon = get_gps_coordinates(image_data)
        coordinates = format_coordinates(lat, lon) if lat is not None and lon is not None else client_coordinates

        # Log coordinate extraction results
        if lat is not None and lon is not None:
            logger.info(f" Using EXIF GPS coordinates: {coordinates}")
        else:
            logger.info(f" Using client-provided coordinates: {client_coordinates}")
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
                return {
                    "success": True,
                    "processed": False,
                    "message": "No road detected in the image. Image not processed.",
                    "classification": classification_result,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "cracks": [],
                    "processed_image": None
                }

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
                    print(f" Crack model dtype error: {e}")
                    print(" Attempting crack inference with CPU fallback...")
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

        # Use comprehensive S3-MongoDB integration workflow
        try:
            workflow = ImageProcessingWorkflow()
            exif_metadata = extract_media_metadata(image_data, 'image')
            metadata = {
                'username': username,
                'role': role,
                'coordinates': coordinates,
                'timestamp': timestamp,
                'exif_data': exif_metadata,
                'media_type': 'image'
            }
            workflow_success, workflow_result = workflow.process_and_store_images(
                image, processed_image, metadata, crack_results, 'crack'
            )
            if not workflow_success:
                raise HTTPException(status_code=500, detail=f"Failed to process and store images: {workflow_result}")
            logger.info(f"Complete workflow successful for crack detection: {workflow_result['image_id']}")
        except Exception as e:
            logger.error(f"Error in comprehensive workflow for cracks: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in image processing workflow: {str(e)}")

        encoded_image = encode_processed_image(processed_image)

        return {
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
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post('/detect-kerbs')
def detect_kerbs(payload: KerbDetectionPayload):
    """API endpoint to detect kerbs and assess their condition in an uploaded image with CUDA optimization"""
    # Get models and device info
    models, _, _ = get_models()
    device = get_device()
    
    if not models or "kerbs" not in models:
        raise HTTPException(status_code=500, detail="Failed to load kerb detection model")
    
    # Get image data and validate it
    image_data = payload.image
    is_valid, error_message = validate_base64_image(image_data, 'kerb_detection')
    if not is_valid:
        logger.warning(f"Image validation failed for kerb detection: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)

    # Extract coordinates if provided
    client_coordinates = payload.coordinates

    # Get user information
    username = payload.username
    role = payload.role
    skip_road_classification = payload.skip_road_classification

    # Get image data
    try:
        image = decode_base64_image(image_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        # Try to extract EXIF GPS data from the image
        lat, lon = get_gps_coordinates(image_data)
        coordinates = format_coordinates(lat, lon) if lat is not None and lon is not None else client_coordinates

        # Log coordinate extraction results
        if lat is not None and lon is not None:
            logger.info(f" Using EXIF GPS coordinates: {coordinates}")
        else:
            logger.info(f" Using client-provided coordinates: {client_coordinates}")

        # Extract comprehensive EXIF metadata
        exif_metadata = extract_media_metadata(image_data, 'image')
            
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
                return {
                    "success": True,
                    "processed": False,
                    "message": "No road detected in the image. Image not processed.",
                    "classification": classification_result,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "kerbs": [],
                    "processed_image": None
                }

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
                    print(f" Kerb model dtype error: {e}")
                    print(" Attempting kerb inference with CPU fallback...")
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

                    # Add text and other details
                    text = f"ID {kerb_id}: {kerb_type['name']}"
                    cv2.putText(processed_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    kerb_info = {
                        "kerb_id": kerb_id,
                        "kerb_type": "Concrete Kerb",
                        "length_m": (x2 - x1) * 0.01,  # Approximate length
                        "condition": kerb_type["name"],
                        "confidence": float(conf),
                        "coordinates": coordinates,
                        "username": username,
                        "role": role
                    }
                    
                    condition_counts[kerb_type["name"]] += 1
                    kerb_results.append(kerb_info)
                    kerb_id += 1
        
        # Use comprehensive S3-MongoDB integration workflow
        try:
            workflow = ImageProcessingWorkflow()
            metadata = {
                'username': username,
                'role': role,
                'coordinates': coordinates,
                'timestamp': timestamp,
                'exif_data': exif_metadata,
                'media_type': 'image'
            }
            workflow_success, workflow_result = workflow.process_and_store_images(
                image, processed_image, metadata, kerb_results, 'kerb'
            )
            if not workflow_success:
                raise HTTPException(status_code=500, detail=f"Failed to process and store images: {workflow_result}")
            logger.info(f"Complete workflow successful for kerb detection: {workflow_result['image_id']}")
        except Exception as e:
            logger.error(f"Error in comprehensive workflow for kerbs: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in image processing workflow: {str(e)}")

        return {
            "success": True,
            "processed": True,
            "classification": classification_result,
            "message": f"Detected {len(kerb_results)} kerbs",
            "processed_image": encode_processed_image(processed_image),
            "kerbs": kerb_results,
            "type_counts": condition_counts,
            "coordinates": coordinates,
            "username": username,
            "role": role
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get('/potholes')
def list_potholes():
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    potholes = list(db.pothole_images.find({}, {'_id': 0}))
    return {"success": True, "potholes": potholes}

@router.get('/potholes/recent')
def get_recent_potholes():
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    recent_potholes = list(db.pothole_images.find({}, {'_id': 0}).sort("timestamp", -1).limit(10))
    return {"success": True, "potholes": recent_potholes}

@router.get('/cracks')
def list_cracks():
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cracks = list(db.crack_images.find({}, {'_id': 0}))
    return {"success": True, "cracks": cracks}

@router.get('/cracks/recent')
def get_recent_cracks():
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

@router.get('/videos/{video_id}/representative_frame')
def get_representative_frame(video_id: str):
    """API endpoint to retrieve the representative frame and details of a video"""
    try:
        db = connect_to_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        video_doc = db.video_processing.find_one({"video_id": video_id})

        if video_doc:
            # Construct an image-like object from the video document
            image_obj = {
                "image_id": video_doc.get("video_id"),
                "timestamp": video_doc.get("timestamp"),
                "coordinates": video_doc.get("coordinates"),
                "username": video_doc.get("username"),
                "role": video_doc.get("role"),
                "media_type": "video",
                "representative_frame": video_doc.get("representative_frame"),
                "potholes": video_doc.get("model_outputs", {}).get("potholes", []),
                "cracks": video_doc.get("model_outputs", {}).get("cracks", []),
                "kerbs": video_doc.get("model_outputs", {}).get("kerbs", []),
                "exif_data": video_doc.get("metadata"), # Assuming metadata is stored here
                "metadata": video_doc.get("metadata")
            }
            return {"success": True, "image": image_obj, "type": "video"}
        else:
            raise HTTPException(status_code=404, detail="Video not found")
            
    except Exception as e:
        logger.error(f"Error retrieving representative frame for video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving representative frame: {str(e)}")
    
    recent_cracks = list(db.crack_images.find({}, {'_id': 0}).sort("timestamp", -1).limit(10))
    return {"success": True, "cracks": recent_cracks}

@router.get('/kerbs')
def list_kerbs():
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    kerbs = list(db.kerb_images.find({}, {'_id': 0}))
    return {"success": True, "kerbs": kerbs}

@router.get('/kerbs/recent')
def get_recent_kerbs():
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    recent_kerbs = list(db.kerb_images.find({}, {'_id': 0}).sort("timestamp", -1).limit(10))
    return {"success": True, "kerbs": recent_kerbs}

@router.api_route("/get-s3-image/{s3_key:path}", methods=["GET", "HEAD"])
async def get_s3_image(s3_key: str):
    """API endpoint to download an image from S3."""
    try:
        workflow = ImageProcessingWorkflow()
        image_data = workflow.s3_manager.download_image_from_s3(s3_key)

        return StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")

@router.get("/get-s3-video/{video_id}/{video_type}")
async def get_s3_video(video_id: str, video_type: str):
    """API endpoint to download a video from S3."""
    try:
        db = connect_to_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Try to find by ObjectId first
        try:
            from bson import ObjectId
            video_doc = db.video_processing.find_one({"_id": ObjectId(video_id)})
        except Exception:
            video_doc = None

        # If not found by ObjectId, try by video_id
        if not video_doc:
            video_doc = db.video_processing.find_one({"video_id": video_id})

        if not video_doc:
            raise HTTPException(status_code=404, detail="Video not found")

        if video_type == "original":
            s3_key = video_doc.get("original_video_url")
        elif video_type == "processed":
            s3_key = video_doc.get("processed_video_url")
        else:
            raise HTTPException(status_code=400, detail="Invalid video type specified")

        if not s3_key:
            raise HTTPException(status_code=404, detail=f"{video_type} video URL not found")

        workflow = ImageProcessingWorkflow()
        video_data = workflow.s3_manager.download_image_from_s3(s3_key)

        return StreamingResponse(io.BytesIO(video_data), media_type="video/mp4")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")

@router.post('/detect-all')
def detect_all(payload: PotholeDetectionPayload):
    """API endpoint to detect all types of defects in an uploaded image."""
    try:
        # Get models and device info
        models, midas, midas_transform = get_models()
        device = get_device()

        if not models:
            raise HTTPException(status_code=500, detail="Failed to load models")

        # Decode image
        image = decode_base64_image(payload.image)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        processed_image = image.copy()
        all_results = {"potholes": [], "cracks": [], "kerbs": []}
        
        # Pothole detection
        pothole_results = detect_potholes(payload)
        if pothole_results and pothole_results.get('potholes'):
            all_results['potholes'] = pothole_results['potholes']
            processed_image = decode_base64_image(pothole_results['processed_image'])

        # Crack detection
        crack_payload = CrackDetectionPayload(**payload.dict())
        crack_results = detect_cracks(crack_payload)
        if crack_results and crack_results.get('cracks'):
            all_results['cracks'] = crack_results['cracks']
            processed_image = decode_base64_image(crack_results['processed_image'])

        # Kerb detection
        kerb_payload = KerbDetectionPayload(**payload.dict())
        kerb_results = detect_kerbs(kerb_payload)
        if kerb_results and kerb_results.get('kerbs'):
            all_results['kerbs'] = kerb_results['kerbs']
            processed_image = decode_base64_image(kerb_results['processed_image'])

        return {
            "success": True,
            "processed": True,
            "message": f"Detected {len(all_results['potholes'])} potholes, {len(all_results['cracks'])} cracks, and {len(all_results['kerbs'])} kerbs",
            "processed_image": encode_processed_image(processed_image),
            "potholes": all_results['potholes'],
            "cracks": all_results['cracks'],
            "kerbs": all_results['kerbs'],
            "coordinates": payload.coordinates,
            "username": payload.username,
            "role": payload.role,
            "classification": {'is_road': True} # Assuming road classification is handled elsewhere or skipped
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/detect-video")
async def detect_video(
    video: UploadFile = File(...),
    selectedModel: str = Form(...),
    coordinates: str = Form(None),
    username: str = Form('Unknown'),
    role: str = Form('Unknown')
):
    """API endpoint to process video for defect detection."""
    try:
        # Generate a unique filename for the temporary video file
        video_timestamp = generate_timestamp_filename()
        temp_video_path = os.path.join(os.path.dirname(__file__), f"video_{video_timestamp}.mp4")

        # Save the uploaded video to the temporary file
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await video.read())

        # Get AWS folder and S3 folder from role and username
        aws_folder = role
        s3_folder = f"{role}/{username}"
        video_id = str(uuid.uuid4())
        original_video_name = video.filename

        # Return a streaming response to send updates
        return StreamingResponse(
            process_pavement_video(
                temp_video_path,
                selectedModel,
                coordinates,
                video_timestamp,
                aws_folder,
                s3_folder,
                username,
                role,
                video_id,
                original_video_name
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@router.post("/stop-video-processing")
async def stop_video_processing():
    """API endpoint to stop ongoing video processing."""
    global video_processing_stop_flag
    video_processing_stop_flag = True
    return {"success": True, "message": "Video processing stopped"}


class DetectAllPayload(PotholeDetectionPayload):
    pass

@router.post('/detect-all')
def detect_all(payload: DetectAllPayload):
    """
    API endpoint to detect all pavement defects (potholes, cracks, kerbs) in an uploaded image.
    This endpoint combines the functionality of the individual detection endpoints.
    """
    try:
        # 1. Initialization
        models, midas, midas_transform = get_models()
        device = get_device()
        if not models:
            raise HTTPException(status_code=500, detail="Failed to load detection models")

        # 2. Image Validation and Decoding
        image_data = payload.image
        is_valid, error_message = validate_base64_image(image_data, 'all_detection')
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        image = decode_base64_image(image_data)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        processed_image = image.copy()
        timestamp = pd.Timestamp.now().isoformat()
        image_upload_id = str(uuid.uuid4())

        # 3. Metadata Extraction
        username = payload.username
        role = payload.role
        lat, lon = get_gps_coordinates(image_data)
        coordinates = format_coordinates(lat, lon) if lat is not None and lon is not None else payload.coordinates
        exif_metadata = extract_media_metadata(image_data, 'image')

        # 4. Road Classification
        if not payload.skip_road_classification:
            classification_result = classify_road_image(processed_image, models, confidence_threshold=0.4)
            if not classification_result["is_road"]:
                return {"success": True, "processed": False, "message": "No road detected.", "classification": classification_result}
        else:
            classification_result = {"is_road": True, "confidence": 1.0, "class_name": "skipped"}

        all_results = {"potholes": [], "cracks": [], "kerbs": []}
        pothole_id_counter = 1
        crack_id_counter = 1
        kerb_id_counter = 1

        # --- 5. Pothole Detection ---
        if "potholes" in models:
            depth_map = estimate_depth(image, midas, midas_transform) if midas else None
            with torch.no_grad():
                results = models["potholes"](cv2.cvtColor(image, cv2.COLOR_BGR2RGB), conf=0.2, device=device)
            
            for result in results:
                if result.boxes is not None and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for mask, box, conf in zip(masks, boxes, confidences):
                        binary_mask = (mask > 0.5).astype(np.uint8) * 255
                        binary_mask = cv2.resize(binary_mask, (processed_image.shape[1], processed_image.shape[0]))
                        
                        mask_indices = binary_mask > 0
                        processed_image[mask_indices] = cv2.addWeighted(processed_image[mask_indices], 0.7, np.full_like(processed_image[mask_indices], (255, 0, 0)), 0.3, 0)
                        
                        dimensions = calculate_pothole_dimensions(binary_mask)
                        depth_metrics = calculate_real_depth(binary_mask, depth_map) if depth_map is not None else {"max_depth_cm": 5.0, "avg_depth_cm": 3.0}

                        if dimensions and depth_metrics:
                            x1, y1, x2, y2 = map(int, box[:4])
                            volume = dimensions["area_cm2"] * depth_metrics["max_depth_cm"]
                            volume_range = "Small (<1k)" if volume < 1000 else ("Medium (1k - 10k)" if volume < 10000 else "Big (>10k)")
                            
                            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text = f"P{pothole_id_counter}, A:{dimensions['area_cm2']:.1f}, D:{depth_metrics['max_depth_cm']:.1f}"
                            cv2.putText(processed_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            all_results["potholes"].append({
                                "pothole_id": pothole_id_counter, "area_cm2": float(dimensions["area_cm2"]), "depth_cm": float(depth_metrics["max_depth_cm"]),
                                "volume": float(volume), "volume_range": volume_range, "confidence": float(conf), "coordinates": coordinates,
                                "username": username, "role": role, "has_mask": True
                            })
                            pothole_id_counter += 1

        # --- 6. Crack Detection ---
        if "cracks" in models:
            with torch.no_grad():
                results = models["cracks"](cv2.cvtColor(image, cv2.COLOR_BGR2RGB), conf=0.2, device=device)
            
            CRACK_TYPES = {0: {"name": "Alligator", "color": (0, 0, 255)}, 1: {"name": "Edge", "color": (0, 255, 255)}, 2: {"name": "Hairline", "color": (255, 0, 0)}, 3: {"name": "Longitudinal", "color": (0, 255, 0)}, 4: {"name": "Transverse", "color": (128, 0, 128)}}

            for result in results:
                if result.masks is not None:
                    for mask, box, cls, conf in zip(result.masks.data.cpu().numpy(), result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                        binary_mask = cv2.resize((mask > 0.5).astype(np.uint8) * 255, (processed_image.shape[1], processed_image.shape[0]))
                        crack_type = CRACK_TYPES.get(int(cls), {"name": "Unknown", "color": (128, 128, 128)})
                        
                        area_data = calculate_area(binary_mask)
                        area_cm2 = area_data["area_cm2"] if area_data else 0
                        area_range = "Small" if area_cm2 < 50 else ("Medium" if area_cm2 < 200 else "Large")
                        
                        x1, y1, x2, y2 = map(int, box[:4])
                        color = crack_type["color"]
                        cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                        label = f"C{crack_id_counter}: {crack_type['name']}"
                        cv2.putText(processed_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        all_results["cracks"].append({
                            "crack_id": crack_id_counter, "crack_type": crack_type["name"], "area_cm2": area_cm2, "area_range": area_range,
                            "coordinates": coordinates, "confidence": float(conf), "username": username, "role": role
                        })
                        crack_id_counter += 1

        # --- 7. Kerb Detection ---
        if "kerbs" in models:
            with torch.no_grad():
                results = models["kerbs"](cv2.cvtColor(image, cv2.COLOR_BGR2RGB), conf=0.5, device=device)

            KERB_TYPES = {0: {"name": "Damaged", "color": (0, 0, 255)}, 1: {"name": "Faded", "color": (0, 165, 255)}, 2: {"name": "Normal", "color": (0, 255, 0)}}

            for result in results:
                if result.boxes is not None:
                    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                        x1, y1, x2, y2 = map(int, box[:4])
                        kerb_type = KERB_TYPES.get(int(cls), {"name": "Unknown", "color": (128, 128, 128)})
                        color = kerb_type["color"]
                        
                        cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                        text = f"K{kerb_id_counter}: {kerb_type['name']}"
                        cv2.putText(processed_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        all_results["kerbs"].append({
                            "kerb_id": kerb_id_counter, "kerb_type": "Concrete", "condition": kerb_type["name"],
                            "length_m": (x2 - x1) * 0.01, "confidence": float(conf), "coordinates": coordinates,
                            "username": username, "role": role
                        })
                        kerb_id_counter += 1
        
        # --- 8. Save to DB and S3 ---
        db = connect_to_db()
        if db is not None:
            if all_results["potholes"]:
                db.pothole_images.insert_one({"image_id": image_upload_id, "timestamp": timestamp, "coordinates": coordinates, "username": username, "role": role, "potholes": all_results["potholes"], "exif_data": exif_metadata, "media_type": "image"})
            if all_results["cracks"]:
                db.crack_images.insert_one({"image_id": image_upload_id, "timestamp": timestamp, "coordinates": coordinates, "username": username, "role": role, "cracks": all_results["cracks"], "exif_data": exif_metadata, "media_type": "image"})
            if all_results["kerbs"]:
                db.kerb_images.insert_one({"image_id": image_upload_id, "timestamp": timestamp, "coordinates": coordinates, "username": username, "role": role, "kerbs": all_results["kerbs"], "exif_data": exif_metadata, "media_type": "image"})

        # --- 9. Return Response ---
        return {
            "success": True, "processed": True, "classification": classification_result,
            "message": f"Detected {len(all_results['potholes'])} potholes, {len(all_results['cracks'])} cracks, {len(all_results['kerbs'])} kerbs.",
            "processed_image": encode_processed_image(processed_image),
            "potholes": all_results['potholes'], "cracks": all_results['cracks'], "kerbs": all_results['kerbs'],
            "coordinates": coordinates, "username": username, "role": role
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.get('/images')
def list_all_images():
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    images = []
    images.extend(list(db.pothole_images.find({}, {'_id': 0})))
    images.extend(list(db.crack_images.find({}, {'_id': 0})))
    images.extend(list(db.kerb_images.find({}, {'_id': 0})))
    
    return {"success": True, "images": images}

@router.get('/images/{image_id}')
def get_image_details(image_id: str):
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    # Check in pothole_images collection
    image = db.pothole_images.find_one({"image_id": image_id}, {'_id': 0})
    if image:
        return {"success": True, "image": image, "type": "pothole"}
        
    # Check in crack_images collection
    image = db.crack_images.find_one({"image_id": image_id}, {'_id': 0})
    if image:
        return {"success": True, "image": image, "type": "crack"}
        
    # Check in kerb_images collection
    image = db.kerb_images.find_one({"image_id": image_id}, {'_id': 0})
    if image:
        return {"success": True, "image": image, "type": "kerb"}
    
    # If not found in any image collection, check in video_processing as a fallback
    video_doc = db.video_processing.find_one({"video_id": image_id})
    if video_doc:
        image_obj = {
            "image_id": video_doc.get("video_id"),
            "timestamp": video_doc.get("timestamp"),
            "coordinates": video_doc.get("coordinates"),
            "username": video_doc.get("username"),
            "role": video_doc.get("role"),
            "media_type": "video",
            "representative_frame": video_doc.get("representative_frame"),
            "potholes": video_doc.get("model_outputs", {}).get("potholes", []),
            "cracks": video_doc.get("model_outputs", {}).get("cracks", []),
            "kerbs": video_doc.get("model_outputs", {}).get("kerbs", []),
            "exif_data": video_doc.get("metadata"),
            "metadata": video_doc.get("metadata")
        }
        return {"success": True, "image": image_obj, "type": "video"}
        
    raise HTTPException(status_code=404, detail="Image or Video not found")
