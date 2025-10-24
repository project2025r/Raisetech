from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import base64
import os
import logging
import time
import json
import torch
from utils.models import load_yolo_models
from utils.file_validation import validate_upload_file, get_context_specific_error_message
from threading import Lock, Thread
from datetime import datetime
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

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
    all_detections = []

    # Run detection with proper dtype handling
    # Ensure image is in the correct format for the model
    if frame.shape[2] == 3:
        inference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        inference_frame = frame

    try:
        results = models["road_infra"](inference_frame, conf=0.3)
    except RuntimeError as e:
        if "dtype" in str(e):
            print(f"⚠️ Road infrastructure model dtype error: {e}")
            print("🔄 Attempting road infrastructure inference with CPU fallback...")
            results = models["road_infra"](inference_frame, conf=0.3, device='cpu')
        else:
            raise e

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

            # Add to all detections list for return value
            all_detections.append({
                'class': detection_class,
                'bbox': bbox,
                'confidence': float(conf)
            })

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

    # Log GPU memory after processing
    if torch.cuda.is_available():
        logger.debug(f"GPU Memory after frame {frame_count}:")
        logger.debug(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.debug(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    logger.debug(f"Frame {frame_count} processed successfully")

    # Return both the processed frame and the detections
    return detection_frame, all_detections

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


def process_video(video_path, from_location, to_location, selected_classes):
    """Process video and yield frames"""
    global current_processing, processing_stop_flag

    try:
        # Reset stop flag at start of processing
        processing_stop_flag = False

        # Get models
        models = get_models()
        if not models:
            yield "data: " + json.dumps({"success": False, "message": "Failed to load models"}) + "\n\n"
            return

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield "data: " + json.dumps({"success": False, "message": "Could not open video file"}) + "\n\n"
            return

        # Get video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video properties - FPS: {frame_rate}, Total frames: {total_frames}, Resolution: {width}x{height}")

        # --- Location Interpolation Logic ---
        video_duration = total_frames / frame_rate
        try:
            from_lat, from_lon = map(float, from_location.split(','))
            to_lat, to_lon = map(float, to_location.split(','))
            start_coord = (from_lat, from_lon)
            end_coord = (to_lat, to_lon)
        except (ValueError, AttributeError):
            yield "data: " + json.dumps({"success": False, "message": "Invalid 'From' or 'To' location format. Expected 'lat,lon'."}) + "\n\n"
            return

        def get_interpolated_location(current_time):
            """Calculate interpolated GPS coordinates for a given time."""
            if current_time > video_duration:
                return None, "Range exceeded. Please ensure the video and location remain within the defined From and To coordinates."

            fraction = min(current_time / video_duration, 1.0)
            lat = start_coord[0] + (end_coord[0] - start_coord[0]) * fraction
            lon = start_coord[1] + (end_coord[1] - start_coord[1]) * fraction
            return (lat, lon), None

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
            current_time = frame_count / frame_rate
            current_second = int(current_time)

            # Get interpolated location for the current time
            coords, error_msg = get_interpolated_location(current_time)
            if error_msg:
                logger.warning(error_msg)
                yield f"data: {json.dumps({'success': False, 'message': error_msg})}\n\n"
                processing_stop_flag = True # Stop processing
                continue


            detection_frame = frame.copy()
            current_frame_detections = []
            continuous_frame_flags = {cls: False for cls in continuous_classes}

            # Run detection with proper dtype handling
            # Ensure image is in the correct format for the model
            if frame.shape[2] == 3:
                inference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                inference_frame = frame

            try:
                results = models["road_infra"](inference_frame, conf=0.3)
            except RuntimeError as e:
                if "dtype" in str(e):
                    print(f"⚠️ Road infrastructure model dtype error: {e}")
                    print("🔄 Attempting road infrastructure inference with CPU fallback...")
                    results = models["road_infra"](inference_frame, conf=0.3, device='cpu')
                else:
                    raise e

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
            matched_row = df_cleaned[df_cleaned["Time"] == current_second]
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

@router.post('/detect')
async def detect_infrastructure(
    selectedClasses: str = Form('[]'),
    from_location: str = Form(None),
    to_location: str = Form(None),
    video: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None)
):
    """API endpoint to detect road infrastructure elements"""
    global current_processing
    global processing_thread

    logger.info("Received detection request")

    selected_classes = json.loads(selectedClasses)

    logger.info(f"Processing with From: {from_location}, To: {to_location}")
    logger.info(f"Selected classes: {selected_classes}")

    try:
        # Check if we have video data
        if video:
            # Process video input
            if not from_location or not to_location:
                raise HTTPException(status_code=400, detail="Please provide both 'From' and 'To' locations for video processing.")

            logger.info(f"Processing video input: {video.filename}")

            # Save video temporarily
            temp_video_path = os.path.join(os.path.dirname(__file__), "temp_video.mp4")
            logger.info(f"Saving video to temporary path: {temp_video_path}")
            with open(temp_video_path, "wb") as buffer:
                buffer.write(await video.read())

            # Start processing in a separate thread
            with processing_lock:
                if processing_thread is not None and processing_thread.is_alive():
                    raise HTTPException(status_code=400, detail="Another media is being processed")
                def target():
                    global current_processing
                    current_processing = process_video(temp_video_path, from_location, to_location, selected_classes)
                processing_thread = Thread(target=target)
                processing_thread.start()

            return {"success": True, "message": "Video processing started"}

        # Check if we have image data
        elif image:
            # Process image input
            logger.info(f"Processing image input: {image.filename}")

            # Save image temporarily
            temp_image_path = os.path.join(os.path.dirname(__file__), "temp_image.jpg")
            logger.info(f"Saving image to temporary path: {temp_image_path}")
            with open(temp_image_path, "wb") as buffer:
                buffer.write(await image.read())

            # Read the image
            frame = cv2.imread(temp_image_path)
            if frame is None:
                raise HTTPException(status_code=400, detail="Could not read image file")

            # Process the image
            models = get_models()
            frame_count = 0
            results, detections = process_video_frame(frame, frame_count, from_location, selected_classes, models)

            # Encode the processed image
            _, buffer = cv2.imencode('.jpg', results)
            img_str = base64.b64encode(buffer).decode('utf-8')

            # Return the processed image and detections
            return {
                "success": True,
                "message": "Image processed successfully",
                "frame": f"data:image/jpeg;base64,{img_str}",
                "detections": detections
            }

        else:
            logger.error("No media data provided")
            raise HTTPException(status_code=400, detail="No media data provided")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/detect')
async def get_detect_stream():
    logger.info("Received GET request for SSE connection")
    if current_processing is None:
        raise HTTPException(status_code=400, detail="No video processing in progress")
    return StreamingResponse(current_processing, media_type='text/event-stream')

@router.get('/data')
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

    return sample_data

@router.post('/stop_processing')
def stop_processing():
    """API endpoint to stop video processing"""
    global processing_stop_flag
    global processing_thread
    global current_processing
    processing_stop_flag = True
    if processing_thread is not None:
        # In async context, we can't block with join. The flag should be sufficient.
        processing_thread = None
    current_processing = None
    return {"success": True, "message": "Stop signal sent"}

@router.get('/status')
def get_processing_status():
    global processing_thread
    status = "processing" if processing_thread is not None and processing_thread.is_alive() else "idle"
    return {"status": status}
