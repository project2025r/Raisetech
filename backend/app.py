from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import application modules
from routes.auth import auth_bp
from routes.pavement import pavement_bp, list_potholes, get_recent_potholes, list_cracks, get_recent_cracks, list_kerbs, get_recent_kerbs, list_all_images, get_image_details
from routes.road_infrastructure import road_infrastructure_bp
from routes.recommendation import recommendation_bp
from routes.dashboard import dashboard_bp, get_image_stats
from routes.users import users_bp

# Create Flask application
app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(pavement_bp, url_prefix='/api/pavement')
app.register_blueprint(road_infrastructure_bp, url_prefix='/api/road-infrastructure')
app.register_blueprint(recommendation_bp, url_prefix='/api/recommendation')
app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
app.register_blueprint(users_bp, url_prefix='/api/users')

# Eagerly load pavement models on application startup
from routes.pavement import preload_models_on_startup
preload_models_on_startup()
logger.info("Pavement models preloaded successfully")

# Register the direct route for potholes list
@app.route('/api/potholes/list', methods=['GET'])
def get_potholes_list():
    return list_potholes()

# Register the direct route for recent potholes
@app.route('/api/potholes/recent', methods=['GET'])
def get_recent_potholes_route():
    return get_recent_potholes()

# Register the direct route for cracks list
@app.route('/api/cracks/list', methods=['GET'])
def get_cracks_list():
    return list_cracks()

# Register the direct route for recent cracks
@app.route('/api/cracks/recent', methods=['GET'])
def get_recent_cracks_route():
    return get_recent_cracks()

# Register the direct route for kerbs list
@app.route('/api/kerbs/list', methods=['GET'])
def get_kerbs_list():
    return list_kerbs()

# Register the direct route for recent kerbs
@app.route('/api/kerbs/recent', methods=['GET'])
def get_recent_kerbs_route():
    return get_recent_kerbs()

# Register the direct route for all images
@app.route('/api/images/list', methods=['GET'])
def get_all_images_route():
    return list_all_images()

# Register the route for specific image details
@app.route('/api/images/<image_id>', methods=['GET'])
def get_image_details_route(image_id):
    return get_image_details(image_id)

# Register the route for image statistics
@app.route('/api/dashboard/image-stats', methods=['GET'])
def get_image_stats_route():
    return get_image_stats()

# Register the route for video processing data
@app.route('/api/dashboard/video-processing-data', methods=['GET'])
def get_video_processing_data_route():
    from routes.dashboard import get_video_processing_data
    return get_video_processing_data()

# Register the route for video processing export
@app.route('/api/dashboard/video-processing-export', methods=['GET'])
def export_video_processing_data_route():
    from routes.dashboard import export_video_processing_data
    return export_video_processing_data()

# Register the route for S3 video proxy
@app.route('/api/pavement/get-s3-video/<video_id>/<video_type>', methods=['GET'])
def get_s3_video_route(video_id, video_type):
    from routes.pavement import get_s3_video
    return get_s3_video(video_id, video_type)

# Register debug endpoint for videos
@app.route('/api/pavement/debug-videos', methods=['GET'])
def debug_videos_route():
    from routes.pavement import debug_videos
    return debug_videos()

# Basic route for testing
@app.route('/')
def index():
    return jsonify({
        "message": "Road AI Safety Enhancement API",
        "status": "online",
        "version": "1.0.0"
    })

# Error handling
@app.errorhandler(404)
def not_found(error):
    logger.error(f"404 Error: {error}")
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"500 Error: {error}")
    return jsonify({"error": "Server error"}), 500

def initialize_database():
    """Initialize database with retry mechanism"""
    from config.db import connect_to_db, create_collections
    
    # Try to establish database connection with retries
    max_retries = 8  # Increased from 5
    retry_count = 0
    db = None
    backoff_time = 1.0  # Start with 1 second
    
    while retry_count < max_retries and db is None:
        logger.info(f"Attempting database connection (attempt {retry_count+1}/{max_retries})...")
        db = connect_to_db()
        if db is None:
            retry_count += 1
            logger.warning(f"Database connection attempt {retry_count} failed, retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            # Exponential backoff with a max of 8 seconds
            backoff_time = min(backoff_time * 1.5, 8.0)
    
    if db is None:
        logger.warning("Could not connect to database after multiple attempts, continuing without database...")
        return False
    
    # Create collections once we have a connection
    logger.info("Creating database collections...")
    try:
        create_collections()
        logger.info("Database initialization complete!")
        return True
    except Exception as e:
        logger.error(f"Error creating collections: {e}")
        return False

# Run the application
if __name__ == "__main__":
    logger.info("Starting Road AI Safety Enhancement API server")
    try:
        # Initialize the database
        database_ready = initialize_database()
        if not database_ready:
            logger.warning("Application starting without confirmed database connection. Some features may not work.")
        
        # Check if model files exist
        from utils.models import MODEL_PATHS
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                logger.info(f"Model file found: {model_name} at {model_path}")
            else:
                logger.warning(f"Model file NOT found: {model_name} at {model_path}")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True) 