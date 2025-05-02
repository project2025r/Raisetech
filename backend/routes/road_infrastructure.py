from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64
import os
from utils.models import load_yolo_models

road_infrastructure_bp = Blueprint('road_infrastructure', __name__)

# Global variable for models - lazy loaded
models = None

# Define the continuous and distinct classes as in the StreamlitApp
continuous_classes = ["HTP-edge_line", "HTP-lane_line", "Water-Based Kerb Paint"]
distinct_classes = [
    "Cold Plastic Rumble Marking Paint",
    "Raised Pavement Markers",
    "Rubber Speed Breaker",
    "SW_Beam_Crash_Barrier",
    "YNM Informatory Sign Boards"
]

def get_models():
    """Lazy-load models when needed"""
    global models
    
    if models is None:
        models = load_yolo_models()
        # Add class names to the models
        models["road_infra_classes"] = distinct_classes + continuous_classes
    
    return models

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

@road_infrastructure_bp.route('/detect', methods=['POST'])
def detect_infrastructure():
    """
    API endpoint to detect road infrastructure elements
    """
    # Get models
    models = get_models()
    
    if not models:
        return jsonify({
            "success": False,
            "message": "Failed to load models"
        }), 500
    
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "message": "No image data provided"
        }), 400
    
    # Extract coordinates if provided
    coordinates = request.json.get('coordinates', 'Not Available')
    
    # Get selected classes for filtering
    selected_classes = request.json.get('selectedClasses', [])
    
    try:
        # Get and decode image data
        image_data = request.json['image']
        image = decode_base64_image(image_data)
        
        if image is None:
            return jsonify({
                "success": False,
                "message": "Invalid image data"
            }), 400
        
        # Process the image
        processed_image = image.copy()
        
        # Get requested detection type
        detection_type = request.json.get('type', 'road_infra')
        
        if detection_type not in models:
            return jsonify({
                "success": False,
                "message": f"Unknown detection type: {detection_type}"
            }), 400
        
        # Detect objects
        results = models[detection_type](processed_image, conf=0.25)
        
        # Process results
        detections = []
        
        # Tracking data (for unique object counting)
        tracked_objects = {}
        object_id_counter = 0
        iou_threshold = 0.5
        
        # Class-based statistics
        continuous_lengths = {}
        continuous_last_second_added = {}
        
        # Colors for different classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0)   # Olive
        ]
        
        for result in results:
            boxes = result.boxes
            classes = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coords = boxes.xyxy.cpu().numpy()
            
            class_names = models[f"{detection_type}_classes"]
            
            for i, (cls, conf, box) in enumerate(zip(classes, confidences, coords)):
                class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
                
                # Skip if this class is not in selected_classes (if filtering is active)
                if selected_classes and class_name not in selected_classes:
                    continue
                
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Draw bounding box
                color = colors[cls % len(colors)]
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                text = f"{class_name} ({conf:.2f})"
                cv2.putText(processed_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Calculate dimensions
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Determine if this is a distinct or continuous element
                element_type = "distinct" if class_name in distinct_classes else "continuous"
                
                # Add to results
                detections.append({
                    "id": i + 1,
                    "class": class_name,
                    "confidence": round(float(conf), 3),
                    "width": width,
                    "height": height,
                    "area": area,
                    "type": element_type,
                    "coordinates": coordinates
                })
        
        # Encode the processed image
        encoded_image = encode_processed_image(processed_image)
        
        # Prepare summary statistics
        class_counts = {}
        for det in detections:
            class_name = det["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Separate distinct and continuous elements
        distinct_elements = [d for d in detections if d['type'] == 'distinct']
        continuous_elements = [d for d in detections if d['type'] == 'continuous']
        
        # Return results
        return jsonify({
            "success": True,
            "message": f"Detected {len(detections)} objects",
            "processed_image": encoded_image,
            "detections": detections,
            "class_counts": class_counts,
            "distinct_elements": distinct_elements,
            "continuous_elements": continuous_elements
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }), 500

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