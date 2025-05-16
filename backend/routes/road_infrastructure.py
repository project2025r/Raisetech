from flask import Blueprint, request, jsonify, Response, stream_with_context
import cv2
import numpy as np
import base64
import os
import logging
import time
import json
import torch
from utils.models import load_yolo_models
from threading import Lock, Thread
from datetime import datetime
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

road_infrastructure_bp = Blueprint('road_infrastructure', __name__)

# Global variables  
models = None
processing_lock = Lock()
current_processing = None
processing_stop_flag = False
processing_thread = None

# Create processed_videos directory if it doesn't exist
PROCESSED_VIDEOS_DIR = os.path.join(os.path.dirname(__file__), '..', 'processed_videos')
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

# Define the continuous and distinct classes
continuous_classes = [
    'Hot Thermoplastic Paint-edge_line-',
    'Water-Based Kerb Paint',
    'Single W Metal Beam Crash Barrier'
]

distinct_classes = [
    'Hot Thermoplastic Paint-lane_line-',
    'Rubber Speed Breaker',
    'YNM Informatory Sign Boards',
    'Cold Plastic Rumble Marking Paint',
    'Raised Pavement Markers'
]

# Object tracking variables
tracked_objects = {}
object_id_counter = 0
iou_threshold = 0.5
max_missing_frames = 30

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def get_models():
    """Lazy-load models when needed"""
    global models
    
    if models is None:
        logger.info("Loading YOLO models...")
        
        # Log CUDA availability and device info
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA device memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"CUDA device memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        models = load_yolo_models()
        
        # Log model device information
        for model_name, model in models.items():
            if model_name == "road_infra":
                # For YOLO models
                if hasattr(model, 'device'):
                    logger.info(f"Model {model_name} is using device: {model.device}")
                elif hasattr(model, 'model'):
                    logger.info(f"Model {model_name} is using device: {model.model.device}")
            else:
                # For other model types (like SegmentationModel)
                try:
                    device = next(model.parameters()).device
                    logger.info(f"Model {model_name} is using device: {device}")
                except (AttributeError, StopIteration):
                    logger.info(f"Model {model_name} device information not available")
        
        # Get class names from YOLO model
        if 'road_infra' in models:
            yolo_class_names = list(models['road_infra'].names.values())
            models['road_infra_classes'] = yolo_class_names
        else:
            yolo_class_names = []
        
        # # Dynamically assign class names for each detection model
        # for detection_type in models:
        #     if detection_type == "road_infra":
        #         models[f"{detection_type}_classes"] = distinct_classes + continuous_classes
        # logger.info("Models loaded successfully")

    return models

def process_video_frame(frame, frame_count, coordinates, selected_classes, models):
    """Process a single video frame and return detections"""
    global tracked_objects, object_id_counter
    
    logger.debug(f"Processing frame {frame_count}")
    
    # Log GPU memory before processing
    if torch.cuda.is_available():
        logger.debug(f"GPU Memory before frame {frame_count}:")
        logger.debug(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.debug(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    start_time = time.time()
    detection_frame = frame.copy()
    current_frame_detections = []
    continuous_frame_flags = {cls: False for cls in continuous_classes}
    
    # Run detection
    results = models["road_infra"](frame, conf=0.3)
    
    # Log inference time
    inference_time = time.time() - start_time
    logger.debug(f"Frame {frame_count} inference time: {inference_time:.3f} seconds")
        
    # Use model's class names for mapping
    class_names = models["road_infra_classes"]
    
    # Process detections
    for result in results:
        boxes = result.boxes
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coords = boxes.xyxy.cpu().numpy()
        
        for box, cls, conf in zip(coords, classes, confidences):
            detection_class = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            
            # Skip if not in selected classes
            if selected_classes and detection_class not in selected_classes:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            bbox = [x1, y1, x2, y2]
            
            if detection_class in distinct_classes:
                current_frame_detections.append({
                    'class': detection_class,
                    'bbox': bbox,
                    'confidence': float(conf)
                })
            elif detection_class in continuous_classes:
                continuous_frame_flags[detection_class] = True
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(detection_frame, detection_class, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    # Update tracked objects
    updated_tracked = {}
    used_ids = set()
    
    for det in current_frame_detections:
        best_match_id = None
        best_iou = 0
        
        for obj_id, obj in tracked_objects.items():
            if obj['class'] != det['class']:
                continue
            iou = calculate_iou(det['bbox'], obj['bbox'])
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_match_id = obj_id
        
        if best_match_id is not None:
            tracked_objects[best_match_id]['bbox'] = det['bbox']
            tracked_objects[best_match_id]['last_seen'] = frame_count
            updated_tracked[best_match_id] = tracked_objects[best_match_id]
            used_ids.add(best_match_id)
        else:
            object_id_counter += 1
            new_id = object_id_counter
            updated_tracked[new_id] = {
                'class': det['class'],
                'bbox': det['bbox'],
                'last_seen': frame_count
            }
    
    # Update tracked objects that weren't matched
    for obj_id, obj in tracked_objects.items():
        if obj_id not in used_ids:
            if frame_count - obj['last_seen'] <= max_missing_frames:
                updated_tracked[obj_id] = obj
    
    tracked_objects = updated_tracked
    
    # Draw tracked objects
    for obj_id, obj in tracked_objects.items():
        if obj['last_seen'] == frame_count and obj['class'] in distinct_classes:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{obj['class']} ({obj_id})"
            cv2.putText(detection_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert frame to base64 for streaming
    success, buffer = cv2.imencode('.jpg', detection_frame)
    if not success:
        logger.error(f'Frame {frame_count}: JPEG encoding failed!')
        frame_base64 = ''
    else:
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
    logger.debug(f'Frame {frame_count} base64 length: {len(frame_base64)}')
    
    # Log GPU memory after processing
    if torch.cuda.is_available():
        logger.debug(f"GPU Memory after frame {frame_count}:")
        logger.debug(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.debug(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    logger.debug(f"Frame {frame_count} processed successfully")
    return frame_base64, current_frame_detections, continuous_frame_flags

def haversine(coord1, coord2):
    # Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def interpolate_coordinates(coordinates):
    cumulative_distance = 0
    intermediate_data = []
    for i in range(len(coordinates)-1):
        start_time, start_coord = coordinates[i]
        end_time, end_coord = coordinates[i+1]
        time_interval = end_time - start_time
        for t in range(time_interval+1):
            fraction = t/time_interval if time_interval != 0 else 0
            lat = start_coord[0] + (end_coord[0]-start_coord[0])*fraction
            lon = start_coord[1] + (end_coord[1]-start_coord[1])*fraction
            coord = (lat, lon)
            if t==0 and i==0:
                distance = 0
            else:
                distance = haversine(intermediate_data[-1]["Coordinates"], coord)
            cumulative_distance += distance
            intermediate_data.append({
                "Time": start_time+t,
                "Coordinates": coord,
                "Distance (km)": round(distance,3),
                "Cumulative Distance (km)": round(cumulative_distance,3)
            })
    return pd.DataFrame(intermediate_data)

coordinates = [
    (0, (1.3006389, 103.8635833)),
    (53, (1.297318, 103.859268)),
    (103, (1.295415, 103.857382)),
    (145, (1.293250, 103.856030)),
    (181, (1.288964, 103.854396)),
    (205, (1.286459, 103.853885)),
    (226, (1.283927, 103.852936)),
    (239, (1.282433, 103.852168)),
    (265, (1.280199, 103.850698)),
    (281, (1.278865, 103.849899))
]

df_cleaned = interpolate_coordinates(coordinates)


def process_video(video_path, coordinates, selected_classes):
    """Process video and yield frames"""
    global current_processing, processing_stop_flag
    
    try:
        # Reset stop flag at start of processing
        processing_stop_flag = False
        
        # Get models
        models = get_models()
        if not models:
            yield json.dumps({"success": False, "message": "Failed to load models"})
            return

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield json.dumps({"success": False, "message": "Could not open video file"})
            return

        # Get video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video properties - FPS: {frame_rate}, Total frames: {total_frames}, Resolution: {width}x{height}")

        # Create output video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_video_{timestamp}.mp4"
        output_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        logger.info(f"Saving processed video to: {output_path}")

        # Initialize tracking variables
        iou_threshold = 0.1
        max_missing_frames = 5
        tracked_objects = {}
        object_id_counter = 0
        distinct_detections = []
        continuous_lengths = {}
        continuous_last_second_added = {}

        # Process frames
        frame_count = 0
        while cap.isOpened() and not processing_stop_flag:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break

            frame_count += 1
            current_second = frame_count // frame_rate
            matched_row = df_cleaned[df_cleaned["Time"] == current_second]
            coords = matched_row.iloc[0]["Coordinates"] if not matched_row.empty else (None, None)

            detection_frame = frame.copy()
            current_frame_detections = []
            continuous_frame_flags = {cls: False for cls in continuous_classes}

            # Run detection
            results = models["road_infra"](frame, conf=0.3)
            
            # Process detections
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    detection_class = models["road_infra"].names[int(cls)]
                    
                    # Skip if not in selected classes
                    if selected_classes and detection_class not in selected_classes:
                        continue
                
                    x1, y1, x2, y2 = map(int, box)
                    bbox = [x1, y1, x2, y2]
                    
                    if detection_class in distinct_classes:
                        current_frame_detections.append({
                            'class': detection_class,
                            'bbox': bbox
                        })
                    elif detection_class in continuous_classes:
                        continuous_frame_flags[detection_class] = True
                        cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(detection_frame, detection_class, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # Update tracked objects
            updated_tracked = {}
            used_ids = set()
            
            for det in current_frame_detections:
                best_match_id = None
                best_iou = 0
                
                for obj_id, obj in tracked_objects.items():
                    if obj['class'] != det['class']:
                        continue
                    iou = calculate_iou(det['bbox'], obj['bbox'])
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_match_id = obj_id
                
                if best_match_id is not None:
                    tracked_objects[best_match_id]['bbox'] = det['bbox']
                    tracked_objects[best_match_id]['last_seen'] = frame_count
                    updated_tracked[best_match_id] = tracked_objects[best_match_id]
                    distinct_detections.append({
                        'Frame': frame_count,
                        'Second': current_second,
                        'ID': best_match_id,
                        'Class': det['class'],
                        'GPS': coords
                    })
                    used_ids.add(best_match_id)
                else:
                    object_id_counter += 1
                    new_id = object_id_counter
                    updated_tracked[new_id] = {
                        'class': det['class'],
                        'bbox': det['bbox'],
                        'last_seen': frame_count
                    }
                    distinct_detections.append({
                        'Frame': frame_count,
                        'Second': current_second,
                        'ID': new_id,
                        'Class': det['class'],
                        'GPS': coords
                    })
            
            # Update tracked objects that weren't matched
            for obj_id, obj in tracked_objects.items():
                if obj_id not in used_ids:
                    if frame_count - obj['last_seen'] <= max_missing_frames:
                        updated_tracked[obj_id] = obj
            
            tracked_objects = updated_tracked

            # Draw tracked objects
            for obj_id, obj in tracked_objects.items():
                if obj['last_seen'] == frame_count and obj['class'] in distinct_classes:
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{obj['class']} ({obj_id})"
                    cv2.putText(detection_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update continuous lengths
            distance = 0
            if not matched_row.empty:
                distance = matched_row.iloc[0]["Distance (km)"]
            for cls, detected in continuous_frame_flags.items():
                if detected:
                    if continuous_last_second_added.get(cls, -1) != current_second:
                        continuous_lengths[cls] = continuous_lengths.get(cls, 0) + distance
                        continuous_last_second_added[cls] = current_second

            # After updating continuous_lengths, draw cumulative length for each detected continuous class
            for result in results:
                for box, cls_idx, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    detection_class = models["road_infra"].names[int(cls_idx)]
                    if detection_class in continuous_classes and continuous_frame_flags[detection_class]:
                        x1, y1, x2, y2 = map(int, box)
                        cum_length = continuous_lengths.get(detection_class, 0)
                        label = f"{detection_class}: {cum_length:.2f} km"
                        cv2.putText(detection_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # Build live tables
            df_distinct = pd.DataFrame(distinct_detections)
            if not df_distinct.empty:
                df_distinct_sorted = df_distinct.sort_values(by="Frame")
                df_distinct_latest = df_distinct_sorted.groupby("ID", as_index=False).last()
                live_distinct_table = df_distinct_latest[['ID', 'Class', 'GPS', 'Frame', 'Second']].to_dict(orient='records')
            else:
                live_distinct_table = []

            live_continuous_table = [
                {"Class": k, "Cumulative Length (km)": round(v, 3)} for k, v in continuous_lengths.items()
            ]

            # Convert frame to base64 for streaming
            success, buffer = cv2.imencode('.jpg', detection_frame)
            if not success:
                logger.error(f'Frame {frame_count}: JPEG encoding failed!')
                frame_base64 = ''
            else:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Write frame to output video
            out.write(detection_frame)

            # Send frame data
            frame_data = {
                "frame": frame_base64,
                "frame_count": frame_count,
                "total_frames": total_frames,
                "live_distinct_table": live_distinct_table,
                "live_continuous_table": live_continuous_table,
                "output_path": output_path
            }

            # At the start, send class_names to the frontend in the first frame's data
            if frame_count == 1:
                frame_data['class_names'] = list(models["road_infra"].names.values())

            yield f"data: {json.dumps(frame_data)}\n\n"

        # Release video writer
        out.release()
        logger.info(f"Processed video saved successfully to: {output_path}")

        # Send final results
        final_data = {
            "success": True,
            "total_frames": frame_count,
            "distinct_detections": distinct_detections,
            "continuous_lengths": continuous_lengths,
            "output_path": output_path,
            "stopped_early": processing_stop_flag
        }

        logger.info("Processing completed successfully")
        yield f"data: {json.dumps(final_data)}\n\n"
    
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'success': False, 'message': str(e)})}\n\n"
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        current_processing = None
        processing_stop_flag = False  # Reset the stop flag

@road_infrastructure_bp.route('/detect', methods=['GET', 'POST'])
def detect_infrastructure():
    """API endpoint to detect road infrastructure elements"""
    global current_processing
    global processing_thread
    
    if request.method == 'GET':
        logger.info("Received GET request for SSE connection")
        if current_processing is None:
            return jsonify({"success": False, "message": "No video processing in progress"}), 400
        return Response(stream_with_context(current_processing), mimetype='text/event-stream')

    logger.info("Received detection request")
    
    # Check if we have video data
    if 'video' not in request.files:
        logger.error("No video data provided")
        return jsonify({
            "success": False,
            "message": "No video data provided"
        }), 400

    # Get video file and other parameters
    video_file = request.files['video']
    coordinates = request.form.get('coordinates', 'Not Available')
    selected_classes = request.form.get('selectedClasses', '[]')
    selected_classes = json.loads(selected_classes)

    logger.info(f"Processing video with coordinates: {coordinates}")
    logger.info(f"Selected classes: {selected_classes}")

    try:
        # Save video temporarily
        temp_video_path = os.path.join(os.path.dirname(__file__), "temp_video.mp4")
        logger.info(f"Saving video to temporary path: {temp_video_path}")
        video_file.save(temp_video_path)

        # Start processing in a separate thread
        with processing_lock:
            if processing_thread is not None and processing_thread.is_alive():
                return jsonify({"success": False, "message": "Another video is being processed"}), 400
            def target():
                global current_processing
                current_processing = process_video(temp_video_path, coordinates, selected_classes)
                # The generator will run and yield data for SSE
            processing_thread = Thread(target=target)
            processing_thread.start()

        return jsonify({"success": True, "message": "Video processing started"})

    except Exception as e:
        logger.error(f"Error during video setup: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500

@road_infrastructure_bp.route('/data', methods=['GET'])
def get_infrastructure_data():
    """
    API endpoint to retrieve infrastructure data from DB
    """
    # This would normally query a database, but for now return sample data
    sample_data = [
        {
            "type": "Road Sign",
            "location": "Orchard Road",
            "condition": "Good",
            "lastInspection": "2023-12-01",
            "latitude": 1.3040,
            "longitude": 103.8318
        },
        {
            "type": "Speed Bump",
            "location": "Ang Mo Kio Ave 3",
            "condition": "Fair",
            "lastInspection": "2023-11-15",
            "latitude": 1.3700,
            "longitude": 103.8470
        },
        {
            "type": "Lane Marking",
            "location": "Toa Payoh Lorong 6",
            "condition": "Poor",
            "lastInspection": "2023-10-28",
            "latitude": 1.3341,
            "longitude": 103.8491
        },
        {
            "type": "Traffic Light",
            "location": "Clementi Road",
            "condition": "Good",
            "lastInspection": "2023-12-05",
            "latitude": 1.3227,
            "longitude": 103.7768
        },
        {
            "type": "Road Barrier",
            "location": "East Coast Parkway",
            "condition": "Good",
            "lastInspection": "2023-11-22",
            "latitude": 1.3004,
            "longitude": 103.9238
        }
    ]
    
    return jsonify(sample_data)

@road_infrastructure_bp.route('/stop_processing', methods=['POST'])
def stop_processing():
    """API endpoint to stop video processing"""
    global processing_stop_flag
    global processing_thread
    global current_processing
    processing_stop_flag = True
    if processing_thread is not None:
        processing_thread.join(timeout=10)  # Wait up to 10 seconds for thread to finish
        processing_thread = None
    current_processing = None
    return jsonify({"success": True, "message": "Stop signal sent"})

@road_infrastructure_bp.route('/status', methods=['GET'])
def get_processing_status():
    global processing_thread
    status = "processing" if processing_thread is not None and processing_thread.is_alive() else "idle"
    return jsonify({"status": status})