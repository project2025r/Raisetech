from flask import Blueprint, jsonify, request
import pandas as pd
import os
import json
import logging
from config.db import connect_to_db
import datetime
from utils.rbac import get_allowed_roles, create_role_filter, validate_user_role
from utils.auth_middleware import validate_rbac_access

# Import our comprehensive S3-MongoDB integration
from s3_mongodb_integration import DashboardImageManager, ImageProcessingWorkflow

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)

def generate_s3_url_for_dashboard(s3_key):
    """
    Helper function to generate S3 URLs for dashboard display

    Args:
        s3_key: S3 key path (e.g., 'Supervisor/supervisor1/original/image_abc123.jpg')

    Returns:
        str: Full S3 URL or None if s3_key is None/empty
    """
    if not s3_key:
        return None

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

def parse_filters():
    """Helper function to parse filter parameters including RBAC"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    username = request.args.get('username')
    user_role = request.args.get('user_role')
    
    # Initialize filters dict
    filters = {}
    
    # Handle date filtering
    date_filter = {}
    if start_date:
        try:
            # Convert to datetime and set to start of day
            start_datetime = datetime.datetime.fromisoformat(start_date)
            date_filter['$gte'] = start_datetime.isoformat()
        except ValueError:
            pass
    
    if end_date:
        try:
            # Convert to datetime and set to end of day
            end_datetime = datetime.datetime.fromisoformat(end_date)
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            date_filter['$lte'] = end_datetime.isoformat()
        except ValueError:
            pass
    
    if date_filter:
        filters["timestamp"] = date_filter
    
    # Handle username filtering
    if username:
        filters["username"] = username
    
    # Handle role-based access control
    if user_role and validate_user_role(user_role):
        role_filter = create_role_filter(user_role)
        filters.update(role_filter)
    
    return filters if filters else {}

@dashboard_bp.route('/summary-v2', methods=['GET'])
@validate_rbac_access
def get_dashboard_summary_v2():
    """
    Enhanced dashboard summary using comprehensive S3-MongoDB integration
    Returns latest detections with proper S3 URLs and GridFS fallback
    """
    try:
        # Get filters
        query_filter = parse_filters()

        logger.info(f"Enhanced dashboard summary requested with filters: {query_filter}")

        # Use comprehensive S3-MongoDB integration for dashboard data
        try:
            # Initialize the workflow manager
            workflow = ImageProcessingWorkflow()

            # Get comprehensive dashboard data with S3 URLs and GridFS fallback
            dashboard_success, dashboard_result = workflow.get_dashboard_data(
                defect_types=['pothole', 'crack', 'kerb'],
                limit_per_type=50
            )

            if not dashboard_success:
                logger.error(f"Failed to get dashboard data: {dashboard_result}")
                return jsonify({
                    "success": False,
                    "message": f"Failed to retrieve dashboard data: {dashboard_result}"
                }), 500

            # Apply RBAC filters to the dashboard data
            filtered_dashboard_data = apply_rbac_filters_to_dashboard_data(dashboard_result, query_filter)

            # Add summary statistics
            summary_stats = calculate_dashboard_summary_stats(filtered_dashboard_data)
            filtered_dashboard_data['summary'] = summary_stats

            logger.info(f"✅ Enhanced dashboard data retrieved successfully")

            return jsonify({
                "success": True,
                "data": filtered_dashboard_data,
                "message": "Dashboard data retrieved successfully"
            })

        except Exception as e:
            logger.error(f"Error in comprehensive dashboard workflow: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Error retrieving dashboard data: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"Error in dashboard summary v2: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Internal server error: {str(e)}"
        }), 500

def apply_rbac_filters_to_dashboard_data(dashboard_data, filters):
    """
    Apply RBAC and other filters to dashboard data

    Args:
        dashboard_data (dict): Raw dashboard data from workflow
        filters (dict): Filters to apply

    Returns:
        dict: Filtered dashboard data
    """
    filtered_data = {}

    for defect_type, data in dashboard_data.items():
        if 'latest' in data:
            # Apply filters to the latest data
            filtered_latest = []

            for item in data['latest']:
                # Apply username filter
                if 'username' in filters and item.get('username') != filters['username']:
                    continue

                # Apply role filter (handle both simple string and MongoDB-style $in queries)
                if 'role' in filters:
                    role_filter = filters['role']
                    item_role = item.get('role')

                    # Handle MongoDB-style $in query
                    if isinstance(role_filter, dict) and '$in' in role_filter:
                        allowed_roles = role_filter['$in']
                        if item_role not in allowed_roles:
                            continue
                    # Handle simple string comparison
                    elif isinstance(role_filter, str):
                        if item_role != role_filter:
                            continue

                # Apply timestamp filter (if needed)
                # Add more filters as needed

                filtered_latest.append(item)

            filtered_data[defect_type] = {
                'latest': filtered_latest,
                'count': len(filtered_latest)
            }
        else:
            filtered_data[defect_type] = data

    return filtered_data

def calculate_dashboard_summary_stats(dashboard_data):
    """
    Calculate summary statistics for dashboard

    Args:
        dashboard_data (dict): Dashboard data

    Returns:
        dict: Summary statistics
    """
    total_images = 0
    total_defects = 0

    for defect_type, data in dashboard_data.items():
        if 'latest' in data:
            total_images += len(data['latest'])

            # Count individual defects
            for item in data['latest']:
                if defect_type == 'potholes' and 'potholes' in item:
                    total_defects += len(item['potholes'])
                elif defect_type == 'cracks' and 'cracks' in item:
                    total_defects += len(item['cracks'])
                elif defect_type == 'kerbs' and 'kerbs' in item:
                    total_defects += len(item['kerbs'])

    return {
        'total_images': total_images,
        'total_defects': total_defects,
        'defect_types': len([k for k in dashboard_data.keys() if 'latest' in dashboard_data[k]])
    }

@dashboard_bp.route('/summary', methods=['GET'])
@validate_rbac_access
def get_dashboard_summary():
    """
    Get summary statistics for the dashboard
    """
    db = connect_to_db()
    if db is None:
        return jsonify({
            "success": False,
            "message": "Database connection failed"
        }), 500

    try:
        # Get filters
        query_filter = parse_filters()
        
        # Initialize result container
        dashboard_data = {
            "potholes": {
                "count": 0,
                "by_size": {"Small (<1k)": 0, "Medium (1k - 10k)": 0, "Big (>10k)": 0},
                "avg_volume": 0,
                "latest": []
            },
            "cracks": {
                "count": 0,
                "by_type": {
                    "Alligator Crack": 0, 
                    "Edge Crack": 0, 
                    "Hairline Cracks": 0,
                    "Longitudinal Cracking": 0, 
                    "Transverse Cracking": 0
                },
                "by_size": {"Small (<50)": 0, "Medium (50-200)": 0, "Large (>200)": 0},
                "latest": []
            },
            "kerbs": {
                "count": 0,
                "by_condition": {"Normal Kerbs": 0, "Faded Kerbs": 0, "Damaged Kerbs": 0},
                "latest": []
            }
        }
        
        # --- POTHOLES ---
        
        # Count potholes from pothole_images collection (sum of all pothole_count fields)
        pothole_count = 0
        try:
            pothole_pipeline = [
                {"$match": query_filter} if query_filter else {"$match": {}},
                {"$group": {"_id": None, "total": {"$sum": "$pothole_count"}}}
            ]
            pothole_result = list(db.pothole_images.aggregate(pothole_pipeline))
            if pothole_result and len(pothole_result) > 0:
                pothole_count = pothole_result[0].get("total", 0) or 0
        except Exception as e:
            logger.error(f"Error aggregating potholes: {e}")
            pothole_count = 0
        
        dashboard_data["potholes"]["count"] = pothole_count
        
        # Get data from pothole images
        if pothole_count > 0:
            try:
                # Get pothole images sorted by timestamp
                pothole_images = list(db.pothole_images.find(query_filter).sort([("timestamp", -1)]))
                
                # Calculate total volume and count for average
                total_volume = 0
                total_count = 0
                
                for image in pothole_images:
                    # Add all potholes from this image to the latest list
                    for pothole in image.get("potholes", []):
                        # Add image metadata to each pothole
                        pothole_data = {
                            **pothole,
                            "image_id": image.get("image_id"),
                            "timestamp": image.get("timestamp"),
                            "username": image.get("username"),
                            "original_image_s3_url": image.get("original_image_s3_url"),
                            "processed_image_s3_url": image.get("processed_image_s3_url"),
                            # Full S3 URLs for direct image display
                            "original_image_full_url": generate_s3_url_for_dashboard(image.get("original_image_s3_url")),
                            "processed_image_full_url": generate_s3_url_for_dashboard(image.get("processed_image_s3_url")),
                            # Keep legacy fields for backward compatibility
                            "original_image_id": image.get("original_image_id"),
                            "processed_image_id": image.get("processed_image_id")
                        }
                        dashboard_data["potholes"]["latest"].append(pothole_data)
                        
                        # Update size distribution
                        if "volume_range" in pothole and pothole["volume_range"] in dashboard_data["potholes"]["by_size"]:
                            dashboard_data["potholes"]["by_size"][pothole["volume_range"]] += 1
                        
                        # Add to volume calculations
                        if "volume" in pothole:
                            total_volume += pothole["volume"]
                            total_count += 1
                
                # Calculate average volume
                if total_count > 0:
                    dashboard_data["potholes"]["avg_volume"] = round(total_volume / total_count, 2)
            except Exception as e:
                logger.error(f"Error processing pothole images: {e}")
        
        # Sort all potholes by timestamp, newest first
        try:
            dashboard_data["potholes"]["latest"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.error(f"Error sorting potholes: {e}")
        
        # --- CRACKS ---
        
        # Count cracks in the new structure (sum of all crack_count fields)
        crack_count = 0
        try:
            crack_pipeline = [
                {"$match": query_filter} if query_filter else {"$match": {}},
                {"$group": {"_id": None, "total": {"$sum": "$crack_count"}}}
            ]
            crack_result = list(db.crack_images.aggregate(crack_pipeline))
            if crack_result and len(crack_result) > 0:
                crack_count = crack_result[0].get("total", 0) or 0
        except Exception as e:
            logger.error(f"Error aggregating cracks: {e}")
            crack_count = 0
        
        dashboard_data["cracks"]["count"] = crack_count
        
        # Get data from crack images
        if crack_count > 0:
            try:
                # Get crack images sorted by timestamp
                crack_images = list(db.crack_images.find(query_filter).sort([("timestamp", -1)]))
                
                for image in crack_images:
                    # Add type counts if available
                    if "type_counts" in image:
                        for crack_type, count in image["type_counts"].items():
                            if crack_type in dashboard_data["cracks"]["by_type"]:
                                dashboard_data["cracks"]["by_type"][crack_type] += count
                    
                    # Add all cracks from this image to the latest list
                    for crack in image.get("cracks", []):
                        # Add image metadata to each crack
                        crack_data = {
                            **crack,
                            "image_id": image.get("image_id"),
                            "timestamp": image.get("timestamp"),
                            "username": image.get("username"),
                            "original_image_s3_url": image.get("original_image_s3_url"),
                            "processed_image_s3_url": image.get("processed_image_s3_url"),
                            # Full S3 URLs for direct image display
                            "original_image_full_url": generate_s3_url_for_dashboard(image.get("original_image_s3_url")),
                            "processed_image_full_url": generate_s3_url_for_dashboard(image.get("processed_image_s3_url")),
                            # Keep legacy fields for backward compatibility
                            "original_image_id": image.get("original_image_id"),
                            "processed_image_id": image.get("processed_image_id")
                        }
                        dashboard_data["cracks"]["latest"].append(crack_data)
                        
                        # Update size distribution if not counted in type_counts
                        if "type_counts" not in image and "area_range" in crack:
                            key = crack["area_range"]
                            if "Small" in key:
                                dashboard_data["cracks"]["by_size"]["Small (<50)"] += 1
                            elif "Medium" in key:
                                dashboard_data["cracks"]["by_size"]["Medium (50-200)"] += 1
                            elif "Large" in key:
                                dashboard_data["cracks"]["by_size"]["Large (>200)"] += 1
            except Exception as e:
                logger.error(f"Error processing crack images: {e}")
        
        # Sort all cracks by timestamp, newest first
        try:
            dashboard_data["cracks"]["latest"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.error(f"Error sorting cracks: {e}")
        
        # --- KERBS ---
        
        # Count kerbs in the new structure (sum of all kerb_count fields)
        kerb_count = 0
        try:
            kerb_pipeline = [
                {"$match": query_filter} if query_filter else {"$match": {}},
                {"$group": {"_id": None, "total": {"$sum": "$kerb_count"}}}
            ]
            kerb_result = list(db.kerb_images.aggregate(kerb_pipeline))
            if kerb_result and len(kerb_result) > 0:
                kerb_count = kerb_result[0].get("total", 0) or 0
        except Exception as e:
            logger.error(f"Error aggregating kerbs: {e}")
            kerb_count = 0
        
        dashboard_data["kerbs"]["count"] = kerb_count
        
        # Get data from kerb images
        if kerb_count > 0:
            try:
                # Get kerb images sorted by timestamp
                kerb_images = list(db.kerb_images.find(query_filter).sort([("timestamp", -1)]))
                
                for image in kerb_images:
                    # Add condition counts if available
                    if "condition_counts" in image:
                        for condition, count in image["condition_counts"].items():
                            if condition in dashboard_data["kerbs"]["by_condition"]:
                                dashboard_data["kerbs"]["by_condition"][condition] += count
                    
                    # Add all kerbs from this image to the latest list
                    for kerb in image.get("kerbs", []):
                        # Add image metadata to each kerb
                        kerb_data = {
                            **kerb,
                            "image_id": image.get("image_id"),
                            "timestamp": image.get("timestamp"),
                            "username": image.get("username"),
                            "original_image_s3_url": image.get("original_image_s3_url"),
                            "processed_image_s3_url": image.get("processed_image_s3_url"),
                            # Full S3 URLs for direct image display
                            "original_image_full_url": generate_s3_url_for_dashboard(image.get("original_image_s3_url")),
                            "processed_image_full_url": generate_s3_url_for_dashboard(image.get("processed_image_s3_url")),
                            # Keep legacy fields for backward compatibility
                            "original_image_id": image.get("original_image_id"),
                            "processed_image_id": image.get("processed_image_id")
                        }
                        dashboard_data["kerbs"]["latest"].append(kerb_data)
                        
                        # Update condition distribution if not counted in condition_counts
                        if "condition_counts" not in image and "condition" in kerb:
                            condition = kerb["condition"]
                            if condition in dashboard_data["kerbs"]["by_condition"]:
                                dashboard_data["kerbs"]["by_condition"][condition] += 1
            except Exception as e:
                logger.error(f"Error processing kerb images: {e}")
        
        # Sort all kerbs by timestamp, newest first
        try:
            dashboard_data["kerbs"]["latest"].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.error(f"Error sorting kerbs: {e}")
        
        return jsonify({
            "success": True,
            "data": dashboard_data
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching dashboard data: {str(e)}"
        }), 500

@dashboard_bp.route('/pothole-data', methods=['GET'])
def get_pothole_data():
    """
    Get all pothole data for analysis and visualization
    """
    try:
        # Look for CSV file first (for compatibility with old visualization)
        csv_path = os.path.join('..', 'dashboard', 'potholes_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return jsonify({
                "success": True,
                "data": df.to_dict(orient='records')
            })
        
        # Otherwise, get data from database
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Get filters
        query_filter = parse_filters()
        
        results = []
        
        # Get potholes from old collection
        old_potholes = list(db.potholes.find(query_filter, {'_id': 0}))
        
        for pothole in old_potholes:
            results.append({
                "Pothole ID": pothole.get("pothole_id"),
                "Area (cm²)": pothole.get("area_cm2"),
                "Depth (cm)": pothole.get("depth_cm"),
                "Volume": pothole.get("volume"),
                "Volume Range": pothole.get("volume_range"),
                "Coordinates": pothole.get("coordinates"),
                "Username": pothole.get("username", "Unknown"),
                "Role": pothole.get("role", "Unknown"),
                "Timestamp": pothole.get("timestamp", ""),
                "Original Image": pothole.get("original_image_id"),
                "Processed Image": pothole.get("processed_image_id"),
                "Data Model": "old"
            })
        
        # Get potholes from new collection
        pothole_images = list(db.pothole_images.find(query_filter))
        
        for image in pothole_images:
            image_id = image.get("image_id")
            coordinates = image.get("coordinates")
            timestamp = image.get("timestamp")
            username = image.get("username", "Unknown")
            role = image.get("role", "Unknown")
            original_image_id = image.get("original_image_id")
            processed_image_id = image.get("processed_image_id")
            original_image_s3_url = image.get("original_image_s3_url")
            processed_image_s3_url = image.get("processed_image_s3_url")
            
            for pothole in image.get("potholes", []):
                results.append({
                    "Pothole ID": pothole.get("pothole_id"),
                    "Area (cm²)": pothole.get("area_cm2"),
                    "Depth (cm)": pothole.get("depth_cm"),
                    "Volume": pothole.get("volume"),
                    "Volume Range": pothole.get("volume_range"),
                    "Coordinates": coordinates,
                    "Username": username,
                    "Role": role,
                    "Timestamp": timestamp,
                    "Original Image": original_image_id,
                    "Processed Image": processed_image_id,
                    "Original Image S3 URL": original_image_s3_url,
                    "Processed Image S3 URL": processed_image_s3_url,
                    "Image ID": image_id,
                    "Data Model": "new"
                })
        
        # Sort by timestamp
        results.sort(key=lambda x: x.get("Timestamp", ""), reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(results),
            "data": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching pothole data: {str(e)}"
        }), 500

@dashboard_bp.route('/crack-data', methods=['GET'])
def get_crack_data():
    """
    Get all crack data for analysis and visualization
    """
    try:
        # Look for CSV file first (for compatibility with old visualization)
        csv_path = os.path.join('..', 'dashboard', 'crack_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return jsonify({
                "success": True,
                "data": df.to_dict(orient='records')
            })
        
        # Otherwise, get data from database
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Get filters
        query_filter = parse_filters()
        
        results = []
        
        # Get cracks from old collection
        old_cracks = list(db.cracks.find(query_filter, {'_id': 0}))
        
        for crack in old_cracks:
            results.append({
                "Crack ID": crack.get("crack_id"),
                "Type": crack.get("crack_type"),
                "Area (cm²)": crack.get("area_cm2"),
                "Area Range": crack.get("area_range"),
                "Coordinates": crack.get("coordinates"),
                "Username": crack.get("username", "Unknown"),
                "Role": crack.get("role", "Unknown"),
                "Timestamp": crack.get("timestamp", ""),
                "Original Image": crack.get("original_image_id"),
                "Processed Image": crack.get("processed_image_id"),
                "Data Model": "old"
            })
        
        # Get cracks from new collection
        crack_images = list(db.crack_images.find(query_filter))
        
        for image in crack_images:
            image_id = image.get("image_id")
            coordinates = image.get("coordinates")
            timestamp = image.get("timestamp")
            username = image.get("username", "Unknown")
            role = image.get("role", "Unknown")
            original_image_id = image.get("original_image_id")
            processed_image_id = image.get("processed_image_id")
            original_image_s3_url = image.get("original_image_s3_url")
            processed_image_s3_url = image.get("processed_image_s3_url")

            for crack in image.get("cracks", []):
                results.append({
                    "Crack ID": crack.get("crack_id"),
                    "Type": crack.get("crack_type"),
                    "Area (cm²)": crack.get("area_cm2"),
                    "Area Range": crack.get("area_range"),
                    "Coordinates": coordinates,
                    "Username": username,
                    "Role": role,
                    "Timestamp": timestamp,
                    "Original Image": original_image_id,
                    "Processed Image": processed_image_id,
                    "Original Image S3 URL": original_image_s3_url,
                    "Processed Image S3 URL": processed_image_s3_url,
                    "Image ID": image_id,
                    "Confidence": crack.get("confidence"),
                    "Data Model": "new"
                })
        
        # Sort by timestamp
        results.sort(key=lambda x: x.get("Timestamp", ""), reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(results),
            "data": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching crack data: {str(e)}"
        }), 500

@dashboard_bp.route('/kerb-data', methods=['GET'])
def get_kerb_data():
    """
    Get all kerb data for analysis and visualization
    """
    try:
        # Look for CSV file first (for compatibility with old visualization)
        csv_path = os.path.join('..', 'dashboard', 'kerb_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return jsonify({
                "success": True,
                "data": df.to_dict(orient='records')
            })
        
        # Get data from database
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Get filters
        query_filter = parse_filters()
        
        results = []
        
        # Get kerbs from old collection
        old_kerbs = list(db.kerbs.find(query_filter, {'_id': 0}))
        
        for kerb in old_kerbs:
            results.append({
                "Kerb ID": kerb.get("kerb_id"),
                "Type": kerb.get("kerb_type"),
                "Length (m)": kerb.get("length_m"),
                "Condition": kerb.get("condition"),
                "Coordinates": kerb.get("coordinates"),
                "Username": kerb.get("username", "Unknown"),
                "Role": kerb.get("role", "Unknown"),
                "Timestamp": kerb.get("timestamp", ""),
                "Original Image": kerb.get("original_image_id"),
                "Processed Image": kerb.get("processed_image_id"),
                "Confidence": kerb.get("confidence"),
                "Data Model": "old"
            })
        
        # Get kerbs from new collection
        kerb_images = list(db.kerb_images.find(query_filter))
        
        for image in kerb_images:
            image_id = image.get("image_id")
            coordinates = image.get("coordinates")
            timestamp = image.get("timestamp")
            username = image.get("username", "Unknown")
            role = image.get("role", "Unknown")
            original_image_id = image.get("original_image_id")
            processed_image_id = image.get("processed_image_id")
            original_image_s3_url = image.get("original_image_s3_url")
            processed_image_s3_url = image.get("processed_image_s3_url")

            for kerb in image.get("kerbs", []):
                results.append({
                    "Kerb ID": kerb.get("kerb_id"),
                    "Type": kerb.get("kerb_type"),
                    "Length (m)": kerb.get("length_m"),
                    "Condition": kerb.get("condition"),
                    "Coordinates": coordinates,
                    "Username": username,
                    "Role": role,
                    "Timestamp": timestamp,
                    "Original Image": original_image_id,
                    "Processed Image": processed_image_id,
                    "Original Image S3 URL": original_image_s3_url,
                    "Processed Image S3 URL": processed_image_s3_url,
                    "Image ID": image_id,
                    "Confidence": kerb.get("confidence"),
                    "Data Model": "new"
                })
        
        # Sort by timestamp
        results.sort(key=lambda x: x.get("Timestamp", ""), reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(results),
            "data": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching kerb data: {str(e)}"
        }), 500

@dashboard_bp.route('/image-stats', methods=['GET'])
@validate_rbac_access
def get_image_stats():
    """
    Get statistics about all captured images
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500

        # Get filters
        query_filter = parse_filters()
        
        # Get all images from the database
        pothole_images = list(db.pothole_images.find(query_filter, {
            '_id': 1,
            'image_id': 1,
            'timestamp': 1,
            'coordinates': 1,
            'username': 1,
            'pothole_count': 1,
            'original_image_id': 1,
            'original_image_s3_url': 1
        }))
        
        crack_images = list(db.crack_images.find(query_filter, {
            '_id': 1,
            'image_id': 1,
            'timestamp': 1,
            'coordinates': 1,
            'username': 1,
            'crack_count': 1,
            'type_counts': 1,
            'original_image_id': 1,
            'original_image_s3_url': 1
        }))

        kerb_images = list(db.kerb_images.find(query_filter, {
            '_id': 1,
            'image_id': 1,
            'timestamp': 1,
            'coordinates': 1,
            'username': 1,
            'kerb_count': 1,
            'condition_counts': 1,
            'original_image_id': 1,
            'original_image_s3_url': 1
        }))
        
        # Process images for response
        all_images = []
        
        # Process pothole images
        for img in pothole_images:
            all_images.append({
                "id": str(img.get('_id')),
                "image_id": img.get('image_id'),
                "timestamp": img.get('timestamp'),
                "coordinates": img.get('coordinates'),
                "username": img.get('username', 'Unknown'),
                "type": "pothole",
                "defect_count": img.get('pothole_count', 0),
                "original_image_id": img.get('original_image_id'),
                "original_image_s3_url": img.get('original_image_s3_url'),
                "original_image_full_url": generate_s3_url_for_dashboard(img.get('original_image_s3_url'))
            })
        
        # Process crack images
        for img in crack_images:
            all_images.append({
                "id": str(img.get('_id')),
                "image_id": img.get('image_id'),
                "timestamp": img.get('timestamp'),
                "coordinates": img.get('coordinates'),
                "username": img.get('username', 'Unknown'),
                "type": "crack",
                "defect_count": img.get('crack_count', 0),
                "type_counts": img.get('type_counts', {}),
                "original_image_id": img.get('original_image_id'),
                "original_image_s3_url": img.get('original_image_s3_url'),
                "original_image_full_url": generate_s3_url_for_dashboard(img.get('original_image_s3_url'))
            })

        # Process kerb images
        for img in kerb_images:
            all_images.append({
                "id": str(img.get('_id')),
                "image_id": img.get('image_id'),
                "timestamp": img.get('timestamp'),
                "coordinates": img.get('coordinates'),
                "username": img.get('username', 'Unknown'),
                "type": "kerb",
                "defect_count": img.get('kerb_count', 0),
                "condition_counts": img.get('condition_counts', {}),
                "original_image_id": img.get('original_image_id'),
                "original_image_s3_url": img.get('original_image_s3_url'),
                "original_image_full_url": generate_s3_url_for_dashboard(img.get('original_image_s3_url'))
            })
        
        # Sort all images by timestamp
        all_images.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Calculate statistics
        total_images = len(all_images)
        pothole_image_count = len(pothole_images)
        crack_image_count = len(crack_images)
        kerb_image_count = len(kerb_images)
        
        # Calculate total defects by type
        total_potholes = sum(img.get('pothole_count', 0) for img in pothole_images)
        total_cracks = sum(img.get('crack_count', 0) for img in crack_images)
        total_kerbs = sum(img.get('kerb_count', 0) for img in kerb_images)
        
        # Get unique users who submitted images
        unique_users = set()
        for img in all_images:
            if 'username' in img and img['username'] != 'Unknown':
                unique_users.add(img['username'])
        
        # Group by date (just the date part, not time)
        date_counts = {}
        for img in all_images:
            if 'timestamp' in img:
                try:
                    # Extract just the date part (YYYY-MM-DD)
                    date_str = img['timestamp'].split('T')[0]
                    date_counts[date_str] = date_counts.get(date_str, 0) + 1
                except:
                    # Skip if timestamp format is invalid
                    continue
        
        # Convert to list of (date, count) pairs and sort by date
        date_distribution = [{"date": date, "count": count} for date, count in date_counts.items()]
        date_distribution.sort(key=lambda x: x["date"], reverse=True)
        
        # Gather user statistics
        user_stats = {}
        for img in all_images:
            username = img.get('username', 'Unknown')
            user_stats[username] = user_stats.get(username, {
                'username': username,
                'image_count': 0,
                'pothole_count': 0,
                'crack_count': 0,
                'kerb_count': 0
            })
            
            user_stats[username]['image_count'] += 1
            
            if img.get('type') == 'pothole':
                user_stats[username]['pothole_count'] += img.get('defect_count', 0)
            elif img.get('type') == 'crack':
                user_stats[username]['crack_count'] += img.get('defect_count', 0)
            elif img.get('type') == 'kerb':
                user_stats[username]['kerb_count'] += img.get('defect_count', 0)
        
        # Convert user_stats dict to list and sort by total image count
        user_stats_list = list(user_stats.values())
        user_stats_list.sort(key=lambda x: x['image_count'], reverse=True)
        
        result = {
            "success": True,
            "total_images": total_images,
            "image_counts": {
                "pothole_images": pothole_image_count,
                "crack_images": crack_image_count,
                "kerb_images": kerb_image_count
            },
            "defect_counts": {
                "potholes": total_potholes,
                "cracks": total_cracks,
                "kerbs": total_kerbs,
                "total": total_potholes + total_cracks + total_kerbs
            },
            "unique_users": len(unique_users),
            "date_distribution": date_distribution,
            "user_stats": user_stats_list,
            "images": all_images[:100]  # Limit to first 100 to avoid massive responses
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error fetching image statistics: {str(e)}"
        }), 500

@dashboard_bp.route('/statistics', methods=['GET'])
@validate_rbac_access
def get_statistics():
    """
    Get statistics for different types of issues
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Parse filters
        query_filter = parse_filters()
        
        response_data = {
            "issues_by_type": {
                "potholes": 0,
                "cracks": 0,
                "kerbs": 0
            },
            "total_issues": 0,
            "total_images": 0,
            "pothole_stats": {
                "count": 0,
                "by_size": {"Small (<1k)": 0, "Medium (1k - 10k)": 0, "Big (>10k)": 0},
                "avg_volume": 0
            },
            "crack_stats": {
                "count": 0,
                "by_type": {
                    "Alligator Crack": 0, 
                    "Edge Crack": 0, 
                    "Hairline Cracks": 0,
                    "Longitudinal Cracking": 0, 
                    "Transverse Cracking": 0
                }
            },
            "kerb_stats": {
                "count": 0,
                "by_condition": {"Normal Kerbs": 0, "Faded Kerbs": 0, "Damaged Kerbs": 0}
            }
        }
        
        # --- POTHOLES ---
        
        # Count potholes from the new collection
        pothole_count = 0
        pothole_pipeline = [
            {"$match": query_filter} if query_filter else {"$match": {}},
            {"$group": {"_id": None, "total": {"$sum": "$pothole_count"}}}
        ]
        pothole_result = list(db.pothole_images.aggregate(pothole_pipeline))
        if pothole_result:
            pothole_count = pothole_result[0]["total"]
        
        response_data["issues_by_type"]["potholes"] = pothole_count
        response_data["total_images"] = pothole_count
        
        # Get pothole size distribution and volume
        if pothole_count > 0:
            # Calculate total volume for average calculation
            total_volume = 0
            volume_count = 0
            
            # Get all pothole images to process distribution
            pothole_images = list(db.pothole_images.find(query_filter))
            
            for image in pothole_images:
                for pothole in image.get("potholes", []):
                    # Update size distribution
                    if "volume_range" in pothole:
                        if pothole["volume_range"] in response_data["pothole_stats"]["by_size"]:
                            response_data["pothole_stats"]["by_size"][pothole["volume_range"]] += 1
                    
                    # Add to volume calculation
                    if "volume" in pothole:
                        total_volume += pothole["volume"]
                        volume_count += 1
            
            # Calculate average volume
            if volume_count > 0:
                response_data["pothole_stats"]["avg_volume"] = round(total_volume / volume_count, 2)
        
        # --- CRACKS ---
        
        # Count cracks from the new collection
        crack_count = 0
        crack_pipeline = [
            {"$match": query_filter} if query_filter else {"$match": {}},
            {"$group": {"_id": None, "total": {"$sum": "$crack_count"}}}
        ]
        crack_result = list(db.crack_images.aggregate(crack_pipeline))
        if crack_result:
            crack_count = crack_result[0]["total"]
        
        response_data["issues_by_type"]["cracks"] = crack_count
        response_data["total_images"] += crack_count
        
        # Get crack type distribution
        if crack_count > 0:
            # Get all crack images to process distribution
            crack_images = list(db.crack_images.find(query_filter))
            
            for image in crack_images:
                # If image has type_counts, use those
                if "type_counts" in image:
                    for crack_type, count in image["type_counts"].items():
                        if crack_type in response_data["crack_stats"]["by_type"]:
                            response_data["crack_stats"]["by_type"][crack_type] += count
                else:
                    # Otherwise count each crack individually
                    for crack in image.get("cracks", []):
                        if "crack_type" in crack and crack["crack_type"] in response_data["crack_stats"]["by_type"]:
                            response_data["crack_stats"]["by_type"][crack["crack_type"]] += 1
        
        # --- KERBS ---
        
        # Count kerbs from the new collection
        kerb_count = 0
        kerb_pipeline = [
            {"$match": query_filter} if query_filter else {"$match": {}},
            {"$group": {"_id": None, "total": {"$sum": "$kerb_count"}}}
        ]
        kerb_result = list(db.kerb_images.aggregate(kerb_pipeline))
        if kerb_result:
            kerb_count = kerb_result[0]["total"]
        
        response_data["issues_by_type"]["kerbs"] = kerb_count
        response_data["total_images"] += kerb_count
        
        # Get kerb condition distribution
        if kerb_count > 0:
            # Get all kerb images to process distribution
            kerb_images = list(db.kerb_images.find(query_filter))
            
            for image in kerb_images:
                # If image has condition_counts, use those
                if "condition_counts" in image:
                    for condition, count in image["condition_counts"].items():
                        if condition in response_data["kerb_stats"]["by_condition"]:
                            response_data["kerb_stats"]["by_condition"][condition] += count
                else:
                    # Otherwise count each kerb individually
                    for kerb in image.get("kerbs", []):
                        if "condition" in kerb and kerb["condition"] in response_data["kerb_stats"]["by_condition"]:
                            response_data["kerb_stats"]["by_condition"][kerb["condition"]] += 1
        
        # Calculate total issues
        response_data["total_issues"] = pothole_count + crack_count + kerb_count
        
        return jsonify({
            "success": True,
            "data": response_data
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching statistics: {str(e)}"
        }), 500

@dashboard_bp.route('/issues-by-type', methods=['GET'])
@validate_rbac_access
def get_issues_by_type():
    """
    Get issue counts by type for charts
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Parse filters
        query_filter = parse_filters()
        
        # Initialize type counts dictionary
        type_counts = {
            # Pothole types
            "Pothole - Small (<1k)": 0,
            "Pothole - Medium (1k - 10k)": 0,
            "Pothole - Big (>10k)": 0,
            
            # Crack types
            "Crack - Alligator Crack": 0,
            "Crack - Edge Crack": 0,
            "Crack - Hairline Cracks": 0,
            "Crack - Longitudinal Cracking": 0,
            "Crack - Transverse Cracking": 0,
            
            # Kerb types
            "Kerb - Normal Kerbs": 0,
            "Kerb - Faded Kerbs": 0,
            "Kerb - Damaged Kerbs": 0
        }
        
        # Get pothole counts from collection
        pothole_images = list(db.pothole_images.find(query_filter))
        for image in pothole_images:
            for pothole in image.get("potholes", []):
                if "volume_range" in pothole:
                    type_key = f"Pothole - {pothole['volume_range']}"
                    if type_key in type_counts:
                        type_counts[type_key] += 1
        
        # Get crack counts from collection
        crack_images = list(db.crack_images.find(query_filter))
        for image in crack_images:
            # If type_counts is available in the image document, use it
            if "type_counts" in image:
                for crack_type, count in image["type_counts"].items():
                    type_key = f"Crack - {crack_type}"
                    if type_key in type_counts:
                        type_counts[type_key] += count
            else:
                # Otherwise, count individual cracks
                for crack in image.get("cracks", []):
                    if "crack_type" in crack:
                        type_key = f"Crack - {crack['crack_type']}"
                        if type_key in type_counts:
                            type_counts[type_key] += 1
        
        # Get kerb counts from collection
        kerb_images = list(db.kerb_images.find(query_filter))
        for image in kerb_images:
            # If condition_counts is available in the image document, use it
            if "condition_counts" in image:
                for condition, count in image["condition_counts"].items():
                    type_key = f"Kerb - {condition}"
                    if type_key in type_counts:
                        type_counts[type_key] += count
            else:
                # Otherwise, count individual kerbs
                for kerb in image.get("kerbs", []):
                    if "condition" in kerb:
                        type_key = f"Kerb - {kerb['condition']}"
                        if type_key in type_counts:
                            type_counts[type_key] += 1
        
        # Convert the dictionary to lists for response
        types = []
        counts = []
        for type_name, count in type_counts.items():
            if count > 0:  # Only include types with non-zero counts
                types.append(type_name)
                counts.append(count)
        
        return jsonify({
            "types": types,
            "counts": counts
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching issue types: {str(e)}"
        }), 500

@dashboard_bp.route('/weekly-trend', methods=['GET'])
@validate_rbac_access
def get_weekly_trend():
    """
    Get weekly trend data for chart
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Parse filters
        query_filter = parse_filters()
        base_query = {}
        
        # Extract username filter if present
        username_filter = None
        if 'username' in query_filter:
            username_filter = query_filter['username']
            base_query['username'] = username_filter
        
        # Extract role filter if present (for RBAC)
        if 'role' in query_filter:
            base_query['role'] = query_filter['role']
        
        # Extract date filter if present
        date_filter = None
        if 'timestamp' in query_filter:
            date_filter = query_filter['timestamp']
        
        # Get the last 7 days for the date range or use default if no filter provided
        if not date_filter:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=7)
        else:
            # Try to extract dates from the filter
            if '$lte' in date_filter:
                end_date = datetime.datetime.fromisoformat(date_filter['$lte'])
            else:
                end_date = datetime.datetime.now()
                
            if '$gte' in date_filter:
                start_date = datetime.datetime.fromisoformat(date_filter['$gte'])
            else:
                start_date = end_date - datetime.timedelta(days=7)
            
            # Limit to 7 days if the range is too large
            date_diff = (end_date - start_date).days
            if date_diff > 7:
                start_date = end_date - datetime.timedelta(days=7)
        
        # Generate day labels
        days = []
        issues_by_day = []
        
        # Loop through each day and get counts
        current_date = start_date
        while current_date <= end_date:
            day_str = current_date.strftime('%Y-%m-%d')
            days.append(current_date.strftime('%a'))  # Day abbreviation
            
            # Create query for this day
            day_query = base_query.copy()
            day_start = datetime.datetime.combine(current_date.date(), datetime.time.min)
            day_end = datetime.datetime.combine(current_date.date(), datetime.time.max)
            
            day_query["timestamp"] = {
                "$gte": day_start.isoformat(),
                "$lte": day_end.isoformat()
            }
            
            # Count issues from new collections for this day
            
            # Potholes
            pothole_count = 0
            pothole_images = list(db.pothole_images.find(day_query))
            for image in pothole_images:
                pothole_count += image.get("pothole_count", 0)
            
            # Cracks
            crack_count = 0
            crack_images = list(db.crack_images.find(day_query))
            for image in crack_images:
                crack_count += image.get("crack_count", 0)
            
            # Kerbs
            kerb_count = 0
            kerb_images = list(db.kerb_images.find(day_query))
            for image in kerb_images:
                kerb_count += image.get("kerb_count", 0)
            
            # Sum all counts
            total_count = pothole_count + crack_count + kerb_count
            
            issues_by_day.append(total_count)
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        return jsonify({
            "days": days,
            "issues": issues_by_day
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching weekly trend: {str(e)}"
        }), 500 