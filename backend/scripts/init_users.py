#!/usr/bin/env python3
"""
Initialize default users in the database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.db import connect_to_db, safe_insert_one, safe_find_one
from werkzeug.security import generate_password_hash
import datetime

def init_default_users():
    """Initialize default users in the database"""
    
    # Default users
    default_users = [
        {"role": "Admin", "username": "admin1", "password": "123456"},
        {"role": "Supervisor", "username": "supervisor1", "password": "123456"},
        {"role": "Field Officer", "username": "fieldofficer1", "password": "123456"}
    ]
    
    try:
        db = connect_to_db()
        if db is None:
            print("❌ Could not connect to database")
            return False
        
        print("🔄 Initializing default users...")
        
        for user_data in default_users:
            # Check if user already exists
            existing_user = safe_find_one(
                db.users, 
                {"username": user_data["username"]}, 
                operation_name="check_existing_user"
            )
            
            if existing_user:
                print(f"✅ User {user_data['username']} already exists")
                continue
            
            # Create new user with hashed password
            new_user = {
                "role": user_data["role"],
                "username": user_data["username"],
                "password": generate_password_hash(user_data["password"]),
                "created_at": datetime.datetime.now().isoformat(),
                "active": True
            }
            
            # Insert user
            safe_insert_one(db.users, new_user, operation_name="create_default_user")
            print(f"✅ Created user: {user_data['username']} with role: {user_data['role']}")
        
        print("🎉 Default users initialization completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing users: {str(e)}")
        return False

if __name__ == "__main__":
    success = init_default_users()
    sys.exit(0 if success else 1)