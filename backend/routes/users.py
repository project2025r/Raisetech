from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from config.db import connect_to_db
import datetime
from utils.rbac import get_allowed_roles, validate_user_role
from utils.auth_middleware import validate_rbac_access

users_bp = Blueprint('users', __name__)

# ... existing code ...

@users_bp.route('/summary', methods=['GET'])
@validate_rbac_access
def get_user_summary():
    """
    API endpoint to get user summary for dashboard with RBAC
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Get user role from request
        user_role = request.args.get('user_role')
        
        # Get allowed roles for RBAC filtering
        allowed_roles = get_allowed_roles(user_role) if user_role and validate_user_role(user_role) else None
        
        # Get total users count (filtered by RBAC)
        if allowed_roles:
            total_users = db.users.count_documents({"role": {"$in": allowed_roles}})
        else:
            total_users = db.users.count_documents({})
        
        # Get roles distribution (filtered by RBAC)
        if allowed_roles:
            role_pipeline = [
                {"$match": {"role": {"$in": allowed_roles}}},
                {"$group": {"_id": "$role", "count": {"$sum": 1}}}
            ]
        else:
            role_pipeline = [
                {"$group": {"_id": "$role", "count": {"$sum": 1}}}
            ]
        role_data = list(db.users.aggregate(role_pipeline))
        
        roles_distribution = {}
        for result in role_data:
            role = result["_id"] if result["_id"] else "unknown"
            roles_distribution[role] = result["count"]
        
        # Get recent users (last 10 logins) - filtered by RBAC
        recent_logins = list(db.user_login_logs.find().sort([("login_time", -1)]).limit(50))
        
        recent_users = []
        for login in recent_logins:
            user_info = db.users.find_one({"username": login.get("username")})
            if user_info:
                user_role_from_db = user_info.get("role", "user")
                
                # Apply RBAC filtering - only include users the requesting user can see
                if allowed_roles is None or user_role_from_db in allowed_roles:
                    recent_users.append({
                        "username": login.get("username"),
                        "role": user_role_from_db,
                        "last_login": login.get("login_time")
                    })
                    
                    # Stop when we have 10 filtered results
                    if len(recent_users) >= 10:
                        break
        
        return jsonify({
            "success": True,
            "total_users": total_users,
            "roles_distribution": roles_distribution,
            "recent_users": recent_users
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching user summary: {str(e)}"
        }), 500

@users_bp.route('/all', methods=['GET'])
@validate_rbac_access
def get_all_users():
    """
    API endpoint to get all users for dashboard filter with RBAC
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Get user role from request
        user_role = request.args.get('user_role')
        
        # Get all users from database with limited fields
        all_users = list(db.users.find({}, {
            "_id": 0,
            "username": 1,
            "role": 1
        }))
        
        # Filter users based on requesting user's role permissions
        if user_role and validate_user_role(user_role):
            allowed_roles = get_allowed_roles(user_role)
            filtered_users = [user for user in all_users if user.get('role') in allowed_roles]
        else:
            # If no role provided or invalid role, return all users (backwards compatibility)
            filtered_users = all_users
        
        return jsonify({
            "success": True,
            "users": filtered_users
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching users: {str(e)}"
        }), 500