from flask import request, jsonify
from functools import wraps
from utils.rbac import validate_user_role, get_allowed_roles

def require_role(allowed_roles):
    """
    Decorator to require specific roles for API endpoints.
    
    Args:
        allowed_roles (list): List of roles that can access the endpoint
    
    Usage:
        @require_role(['Admin', 'Supervisor'])
        def some_endpoint():
            pass
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get user role from query params or JSON body (safely)
            user_role = request.args.get('user_role')
            
            # Only try to get from JSON if it's a POST/PUT request and JSON exists
            if not user_role and request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    if request.json:
                        user_role = request.json.get('user_role')
                except Exception:
                    # If there's any issue accessing JSON, just continue without it
                    pass
            
            if not user_role:
                return jsonify({
                    "success": False,
                    "message": "User role is required"
                }), 401
            
            if not validate_user_role(user_role):
                return jsonify({
                    "success": False,
                    "message": "Invalid user role"
                }), 400
            
            if user_role not in allowed_roles:
                return jsonify({
                    "success": False,
                    "message": "Insufficient permissions"
                }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_rbac_access(f):
    """
    Decorator to validate that the user has valid RBAC permissions.
    This is a general purpose decorator that ensures the user role is valid.
    
    Usage:
        @validate_rbac_access
        def some_endpoint():
            pass
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get user role from query params or JSON body (safely)
        user_role = request.args.get('user_role')
        
        # Only try to get from JSON if it's a POST/PUT request and JSON exists
        if not user_role and request.method in ['POST', 'PUT', 'PATCH']:
            try:
                if request.json:
                    user_role = request.json.get('user_role')
            except Exception:
                # If there's any issue accessing JSON, just continue without it
                pass
        
        if user_role and not validate_user_role(user_role):
            return jsonify({
                "success": False,
                "message": "Invalid user role"
            }), 400
        
        return f(*args, **kwargs)
    return decorated_function

def get_current_user_role():
    """
    Get the current user's role from the request.
    
    Returns:
        str: The user's role, or None if not provided
    """
    # Get user role from query params first
    user_role = request.args.get('user_role')
    
    # Only try to get from JSON if it's a POST/PUT request and JSON exists
    if not user_role and request.method in ['POST', 'PUT', 'PATCH']:
        try:
            if request.json:
                user_role = request.json.get('user_role')
        except Exception:
            # If there's any issue accessing JSON, just continue without it
            pass
    
    return user_role

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
    
    allowed_roles = get_allowed_roles(user_role)
    return data_owner_role in allowed_roles 