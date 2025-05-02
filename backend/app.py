from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging

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
CORS(app, resources={r"/*": {"origins": "http://localhost:5000"}})

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(pavement_bp, url_prefix='/api/pavement')
app.register_blueprint(road_infrastructure_bp, url_prefix='/api/road-infrastructure')
app.register_blueprint(recommendation_bp, url_prefix='/api/recommendation')
app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
app.register_blueprint(users_bp, url_prefix='/api/users')

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

# Run the application
if __name__ == "__main__":
    logger.info("Starting Road AI Safety Enhancement API server")
    try:
        # Create MongoDB collections if they don't exist
        from config.db import create_collections
        create_collections()
        
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