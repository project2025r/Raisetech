from flask import request, jsonify
from functools import wraps
from utils.rbac import validate_user_role, get_allowed_roles

def get_user_role_from_request():
    """
    Helper to extract user_role from query parameters or JSON body.
    """
    user_role = request.args.get('user_role')
    if not user_role:
        if request.is_json:
            try:
                json_body = request.get_json()
                user_role = json_body.get('user_role')
            except Exception:
                pass
    return user_role

def validate_rbac_access(f):
    """
    Decorator to validate that the user has a valid RBAC role.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_role = get_user_role_from_request()
        if user_role and not validate_user_role(user_role):
            return jsonify({"detail": "Invalid user role"}), 400
        return f(*args, **kwargs)
    return decorated_function

def require_role(allowed_roles):
    """
    Decorator factory to protect endpoints based on user roles.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_role = get_user_role_from_request()
            if not user_role:
                return jsonify({"detail": "User role is required"}), 401
            
            if not validate_user_role(user_role):
                return jsonify({"detail": "Invalid user role"}), 400
                
            if user_role not in allowed_roles:
                return jsonify({"detail": "Insufficient permissions"}), 403
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def check_data_access_permission(user_role, data_owner_role):
    """
    Check if the current user can access data from another user.
    
    Args:
        user_role (str): The requesting user's role
        data_owner_role (str): The role of the user who owns the data
        
    Returns:
        bool: True if access is allowed, False otherwise
    """
    if not user_role:
        return False
    
    allowed = get_allowed_roles(user_role)
    return data_owner_role in allowed
