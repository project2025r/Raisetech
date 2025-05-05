from flask import Blueprint, request, jsonify
from config.db import connect_to_db
import datetime
import time
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)

# Valid users (in production, this would be in a database)
VALID_USERS = {
    "Supervisor": {"username": "supervisor1", "password": "1234"},
    "Field Officer": {"username": "fieldofficer1", "password": "1234"},
    "Admin": {"username": "admin1", "password": "1234"}
}

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Authenticate a user based on role, username, and password
    """
    data = request.json
    
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400
    
    role = data.get('role')
    username = data.get('username')
    password = data.get('password')
    
    if not all([role, username, password]):
        return jsonify({"success": False, "message": "Missing credentials"}), 400
    
    # Try database authentication first with improved retry mechanism
    max_retries = 5  # Increased from 3
    retry_count = 0
    db = None
    backoff_time = 0.2  # Start with 200ms
    
    while retry_count < max_retries and db is None:
        db = connect_to_db()
        if db is None:
            retry_count += 1
            print(f"Database connection attempt {retry_count} failed, retrying in {backoff_time}s...")
            time.sleep(backoff_time)
            # Exponential backoff with max of 1.6 seconds
            backoff_time = min(backoff_time * 2, 1.6)
    
    # If we have a database connection, proceed with authentication
    if db is not None:
        try:
            # Find user in database
            user = db.users.find_one({"username": username})
            
            if user and user.get("role") == role and check_password_hash(user.get("password"), password):
                # User exists and credentials match
                # Log the successful login to the database
                try:
                    # Insert into user_login_logs collection with timestamp
                    db.user_login_logs.insert_one({
                        "role": role,
                        "username": username,
                        "login_time": datetime.datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Error logging user login: {e}")
                
                return jsonify({
                    "success": True,
                    "message": "Login successful",
                    "user": {
                        "role": role,
                        "username": username
                    }
                })
        except Exception as e:
            print(f"Database error during authentication: {e}")
            # Continue to hardcoded authentication as fallback
    
    # Fall back to hardcoded users for development
    if role in VALID_USERS:
        valid_user = VALID_USERS[role]
        if username == valid_user["username"] and password == valid_user["password"]:
            # Log the successful login to the database if it's available
            if db is not None:
                try:
                    # Insert into user_login_logs collection
                    db.user_login_logs.insert_one({
                        "role": role,
                        "username": username,
                        "login_time": datetime.datetime.now().isoformat()
                    })
                    
                    # Check if this hardcoded user exists in the database
                    # If not, add them for future reference
                    if not db.users.find_one({"username": username}):
                        db.users.insert_one({
                            "role": role,
                            "username": username,
                            "password": generate_password_hash(password),
                            "created_at": datetime.datetime.now().isoformat()
                        })
                    
                except Exception as e:
                    print(f"Error logging user login: {e}")
            
            return jsonify({
                "success": True,
                "message": "Login successful",
                "user": {
                    "role": role,
                    "username": username
                }
            })
    
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    Handle user logout (in a real app, this would invalidate their session/token)
    """
    return jsonify({
        "success": True,
        "message": "Logout successful"
    })

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user
    """
    data = request.json
    
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400
    
    role = data.get('role')
    username = data.get('username')
    password = data.get('password')
    
    if not all([role, username, password]):
        return jsonify({"success": False, "message": "Missing user information"}), 400
    
    # Check if username already exists
    db = connect_to_db()
    if db is None:
        return jsonify({"success": False, "message": "Database connection failed"}), 500
    
    existing_user = db.users.find_one({"username": username})
    if existing_user:
        return jsonify({"success": False, "message": "Username already exists"}), 409
    
    # Create new user
    try:
        db.users.insert_one({
            "role": role,
            "username": username,
            "password": generate_password_hash(password),
            "created_at": datetime.datetime.now().isoformat()
        })
        
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "user": {
                "role": role,
                "username": username
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error registering user: {str(e)}"
        }), 500 