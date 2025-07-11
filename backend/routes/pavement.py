from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64
import os
import json
import traceback
from config.db import connect_to_db, get_gridfs
from utils.models import load_yolo_models, load_midas, estimate_depth, calculate_real_depth, calculate_pothole_dimensions, calculate_area
from utils.exif_utils import get_gps_coordinates, format_coordinates
import pandas as pd
import io
from bson import ObjectId
import uuid

pavement_bp = Blueprint('pavement', __name__)

# Global variables for models - lazy loaded or preloaded
models = None
midas = None
midas_transform = None

def preload_models_on_startup():
    """Eagerly preload YOLO and MiDaS models when the Flask server starts"""
    global models, midas, midas_transform

    print("Preloading pavement models on server startup...")
    
    # Load YOLO models
    models = load_yolo_models()
    for model in models.values():
        try:
            model.eval()
        except Exception as e:
            print(f"Warning: Could not set model to eval mode: {e}")
    
    # Load MiDaS model
    midas, midas_transform = load_midas()
    if hasattr(midas, 'eval'):
        midas.eval()
    
    print("✅ Pavement models successfully preloaded on startup")
    return models, midas, midas_transform

def preload_models():
    """Preload YOLO and MiDaS models before the first request if not already loaded"""
    global models, midas, midas_transform

    # Only load if models haven't been loaded yet
    if models is None:
        models = load_yolo_models()
        for model in models.values():
            try:
                model.eval()
            except Exception as e:
                print(f"Warning: Could not set model to eval mode: {e}")

    if midas is None or midas_transform is None:
        midas, midas_transform = load_midas()
        if hasattr(midas, 'eval'):
            midas.eval()


def get_models():
    """Return preloaded models"""
    global models, midas, midas_transform
    return models, midas, midas_transform


def decode_base64_image(base64_string):
    """Decode a base64 image to cv2 format"""
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def encode_processed_image(image):
    """Encode a processed image to base64"""
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_image}"


@pavement_bp.route('/detect-potholes', methods=['POST'])
def detect_potholes():
    """
    API endpoint to detect potholes in an uploaded image
    """
    # Get models
    models, midas, midas_transform = get_models()
    
    if not models or "potholes" not in models:
        return jsonify({
            "success": False,
            "message": "Failed to load pothole detection model"
        }), 500
    
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "message": "No image data provided"
        }), 400
    
    # Extract coordinates if provided
    client_coordinates = request.json.get('coordinates', 'Not Available')
    
    # Get user information
    username = request.json.get('username', 'Unknown')
    role = request.json.get('role', 'Unknown')
    
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
        
        # Run depth estimation if MiDaS is available
        depth_map = None
        if midas and midas_transform:
            depth_map = estimate_depth(processed_image, midas, midas_transform)
        else:
            print("MiDaS model not available, skipping depth estimation")
        
        # Detect potholes
        results = models["potholes"](processed_image, conf=0.2)
        
        # Process results
        pothole_results = []
        pothole_id = 1
        timestamp = pd.Timestamp.now().isoformat()
        image_upload_id = str(uuid.uuid4())  # Create a unique ID for this image upload
        
        # Store the original image once for all potholes
        fs = get_gridfs()
        _, original_buffer = cv2.imencode('.jpg', image)
        original_image_bytes = original_buffer.tobytes()
        original_image_id = fs.put(
            original_image_bytes, 
            filename=f"image_{image_upload_id}_original.jpg",
            content_type="image/jpeg"
        )
        
        for result in results:
            if result.masks is None:
                continue
                
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for mask, box in zip(masks, boxes):
                # Process the segmentation mask
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                binary_mask = cv2.resize(binary_mask, (processed_image.shape[1], processed_image.shape[0]))
                
                # Create colored overlay
                colored_mask = np.zeros_like(processed_image)
                colored_mask[:, :, 2] = binary_mask  # Red channel
                
                # Blend the original image with the mask
                processed_image = cv2.addWeighted(processed_image, 1.0, colored_mask, 0.4, 0)
                
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
                    
                    # Draw bounding box and label
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"ID {pothole_id}, A:{dimensions['area_cm2']}cm², D:{depth_metrics['max_depth_cm']}cm, V:{volume}"
                    cv2.putText(processed_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Collect pothole data for later batch insertion
                    pothole_info = {
                        "pothole_id": pothole_id,
                        "area_cm2": float(dimensions["area_cm2"]),
                        "depth_cm": float(depth_metrics["max_depth_cm"]),
                        "volume": float(volume),
                        "volume_range": volume_range,
                        "coordinates": coordinates,
                        "username": username,
                        "role": role
                    }
                    pothole_results.append(pothole_info)
                    pothole_id += 1
        
        # Store processed image with all potholes marked
        _, processed_buffer = cv2.imencode('.jpg', processed_image)
        processed_image_bytes = processed_buffer.tobytes()
        processed_image_id = fs.put(
            processed_image_bytes, 
            filename=f"image_{image_upload_id}_processed.jpg",
            content_type="image/jpeg"
        )
        
        # Store consolidated entry in the database
        db = connect_to_db()
        if db is not None and pothole_results:
            try:
                db.pothole_images.insert_one({
                    "image_id": image_upload_id,
                    "timestamp": timestamp,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "original_image_id": str(original_image_id),
                    "processed_image_id": str(processed_image_id),
                    "pothole_count": len(pothole_results),
                    "potholes": pothole_results
                })
            except Exception as e:
                print(f"Error saving image data: {e}")
        
        # Encode the processed image
        encoded_image = encode_processed_image(processed_image)
        
        return jsonify({
            "success": True,
            "message": f"Detected {len(pothole_results)} potholes",
            "processed_image": encoded_image,
            "potholes": pothole_results
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }), 500

@pavement_bp.route('/detect-cracks', methods=['POST'])
def detect_cracks():
    """
    API endpoint to detect cracks in an uploaded image using segmentation masks
    """
    models, _, _ = get_models()

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

    client_coordinates = request.json.get('coordinates', 'Not Available')
    username = request.json.get('username', 'Unknown')
    role = request.json.get('role', 'Unknown')

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

        results = models["cracks"](processed_image, conf=0.2)

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

        fs = get_gridfs()
        _, original_buffer = cv2.imencode('.jpg', image)
        original_image_id = fs.put(
            original_buffer.tobytes(),
            filename=f"image_{image_upload_id}_original.jpg",
            content_type="image/jpeg"
        )

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
            area = area_data["area_cm2"] if area_data else 0

            if area < 50:
                area_range = "Small (<50)"
            elif area < 200:
                area_range = "Medium (50-200)"
            else:
                area_range = "Large (>200)"

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
                "area_cm2": round(area, 2),
                "area_range": area_range,
                "coordinates": coordinates,
                "confidence": det["confidence"],
                "username": username,
                "role": role
            })
            condition_counts[det["type"]["name"]] += 1
            crack_id += 1

        _, processed_buffer = cv2.imencode('.jpg', processed_image)
        processed_image_id = fs.put(
            processed_buffer.tobytes(),
            filename=f"image_{image_upload_id}_processed.jpg",
            content_type="image/jpeg"
        )

        db = connect_to_db()
        if db is not None and crack_results:
            try:
                db.crack_images.insert_one({
                    "image_id": image_upload_id,
                    "timestamp": timestamp,
                    "coordinates": coordinates,
                    "username": username,
                    "role": role,
                    "original_image_id": str(original_image_id),
                    "processed_image_id": str(processed_image_id),
                    "crack_count": len(crack_results),
                    "cracks": crack_results,
                    "type_counts": condition_counts
                })
            except Exception as e:
                print(f"Error saving image data: {e}")

        encoded_image = encode_processed_image(processed_image)

        return jsonify({
            "success": True,
            "message": f"Detected {len(crack_results)} cracks",
            "processed_image": encoded_image,
            "cracks": crack_results,
            "type_counts": condition_counts
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
    API endpoint to detect kerbs and assess their condition in an uploaded image
    """
    # Get models
    models, _, _ = get_models()
    
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
    
    # Extract coordinates if provided
    client_coordinates = request.json.get('coordinates', 'Not Available')
    
    # Get user information
    username = request.json.get('username', 'Unknown')
    role = request.json.get('role', 'Unknown')
    
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
        
        # Detect kerbs using YOLOv8 model
        results = models["kerbs"](processed_image, conf=0.5)
        
        # Process results
        kerb_results = []
        kerb_id = 1
        timestamp = pd.Timestamp.now().isoformat()
        image_upload_id = str(uuid.uuid4())  # Create a unique ID for this image upload
        
        # Store the original image once for all kerbs
        fs = get_gridfs()
        _, original_buffer = cv2.imencode('.jpg', image)
        original_image_bytes = original_buffer.tobytes()
        original_image_id = fs.put(
            original_image_bytes, 
            filename=f"image_{image_upload_id}_original.jpg",
            content_type="image/jpeg"
        )
        
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
        
        # Store processed image with all kerbs marked
        _, processed_buffer = cv2.imencode('.jpg', processed_image)
        processed_image_bytes = processed_buffer.tobytes()
        processed_image_id = fs.put(
            processed_image_bytes, 
            filename=f"image_{image_upload_id}_processed.jpg",
            content_type="image/jpeg"
        )
        
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
                    "original_image_id": str(original_image_id),
                    "processed_image_id": str(processed_image_id),
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
            "message": f"Detected {len(kerb_results)} kerbs",
            "processed_image": encoded_image,
            "kerbs": kerb_results,
            "condition_counts": condition_counts
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
            return jsonify({
                "success": True,
                "potholes": latest_image["potholes"],
                "image_data": {
                    "image_id": latest_image["image_id"],
                    "timestamp": latest_image["timestamp"],
                    "original_image_id": latest_image["original_image_id"],
                    "processed_image_id": latest_image["processed_image_id"]
                }
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
    API endpoint to detect all types of defects (potholes, cracks, kerbs) in an uploaded image
    
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
    # Get models
    models, midas, midas_transform = get_models()
    
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
    
    # Extract coordinates if provided
    client_coordinates = request.json.get('coordinates', 'Not Available')
    
    # Get user information
    username = request.json.get('username', 'Unknown')
    role = request.json.get('role', 'Unknown')
    
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
            "coordinates": coordinates,
            "username": username,
            "role": role
        }
        
        timestamp = pd.Timestamp.now().isoformat()
        image_upload_id = str(uuid.uuid4())  # Create a unique ID for this image upload
        
        # Store the original image once for all detections
        fs = get_gridfs()
        _, original_buffer = cv2.imencode('.jpg', image)
        original_image_bytes = original_buffer.tobytes()
        original_image_id = fs.put(
            original_image_bytes, 
            filename=f"image_{image_upload_id}_original.jpg",
            content_type="image/jpeg"
        )
        
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
            
            pothole_results = models["potholes"](pothole_inference_image, conf=0.2)
            pothole_id = 1
            
            for result in pothole_results:
                if result.masks is None:
                    continue
                    
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for mask, box in zip(masks, boxes):
                    # Process the segmentation mask
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255
                    binary_mask = cv2.resize(binary_mask, (display_image.shape[1], display_image.shape[0]))
                    
                    # Create colored overlay for potholes (red) - apply to display image only
                    colored_mask = np.zeros_like(display_image)
                    colored_mask[:, :, 2] = binary_mask  # Red channel
                    
                    # Blend the display image with the mask (not the inference image)
                    display_image = cv2.addWeighted(display_image, 1.0, colored_mask, 0.4, 0)
                    
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
                        
                        # Store pothole data in database
                        pothole_data = {
                            "pothole_id": pothole_id,
                            "area_cm2": dimensions["area_cm2"],
                            "depth_cm": depth_metrics["max_depth_cm"],
                            "volume": volume,
                            "volume_range": volume_range,
                            "coordinates": coordinates,
                            "timestamp": timestamp,
                            "bbox": [x1, y1, x2, y2],
                            "username": username,
                            "role": role,
                            "original_image_id": str(original_image_id),
                            "image_upload_id": image_upload_id
                        }
                        
                        # Create a copy for JSON response (without ObjectId references)
                        pothole_data_for_response = {
                            "pothole_id": pothole_id,
                            "area_cm2": dimensions["area_cm2"],
                            "depth_cm": depth_metrics["max_depth_cm"],
                            "volume": volume,
                            "volume_range": volume_range,
                            "coordinates": coordinates,
                            "bbox": [x1, y1, x2, y2]
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
            
            crack_results = models["cracks"](crack_inference_image, conf=0.2)
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
                    
                    # Store crack data
                    crack_data = {
                        "crack_id": crack_id,
                        "crack_type": crack_type,
                        "area_cm2": area_cm2,
                        "area_range": area_range,
                        "confidence": float(conf),
                        "coordinates": coordinates,
                        "timestamp": timestamp,
                        "bbox": [x1, y1, x2, y2],
                        "username": username,
                        "role": role,
                        "original_image_id": str(original_image_id),
                        "image_upload_id": image_upload_id
                    }
                    
                    # Create a copy for JSON response (without ObjectId references)
                    crack_data_for_response = {
                        "crack_id": crack_id,
                        "crack_type": crack_type,
                        "area_cm2": area_cm2,
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
            
            kerb_results = models["kerbs"](kerb_inference_image, conf=0.5)
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
                            "original_image_id": str(original_image_id),
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
                                "original_image_id": str(original_image_id),
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
        
        # Store processed image
        processed_image_id = None
        try:
            _, processed_buffer = cv2.imencode('.jpg', display_image)
            processed_image_bytes = processed_buffer.tobytes()
            processed_image_id = fs.put(
                processed_image_bytes,
                filename=f"image_{image_upload_id}_processed_all.jpg",
                content_type="image/jpeg"
            )
        except Exception as e:
            print(f"Error storing processed image: {str(e)}")
        
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
                "original_image_id": str(original_image_id),
                "processed_image_id": str(processed_image_id) if processed_image_id else None,
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