from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from config.db import connect_to_db
import datetime

users_bp = Blueprint('users', __name__)

# ... existing code ...

@users_bp.route('/summary', methods=['GET'])
def get_user_summary():
    """
    API endpoint to get user summary for dashboard
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Database connection failed"
            }), 500
        
        # Get total users count
        total_users = db.users.count_documents({})
        
        # Get roles distribution
        role_pipeline = [
            {"$group": {"_id": "$role", "count": {"$sum": 1}}}
        ]
        role_data = list(db.users.aggregate(role_pipeline))
        
        roles_distribution = {}
        for result in role_data:
            role = result["_id"] if result["_id"] else "unknown"
            roles_distribution[role] = result["count"]
        
        # Get recent users (last 10 logins)
        recent_logins = list(db.user_login_logs.find().sort([("login_time", -1)]).limit(10))
        
        recent_users = []
        for login in recent_logins:
            user_info = db.users.find_one({"username": login.get("username")})
            if user_info:
                recent_users.append({
                    "username": login.get("username"),
                    "role": user_info.get("role", "user"),
                    "last_login": login.get("login_time")
                })
        
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