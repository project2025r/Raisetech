from fastapi import APIRouter, HTTPException, Body
from config.db import connect_to_db
import datetime
import time
from werkzeug.security import generate_password_hash, check_password_hash
from pydantic import BaseModel

class User(BaseModel):
    role: str
    username: str
    password: str

router = APIRouter()

# Valid users (in production, this would be in a database)
VALID_USERS = {
    "Supervisor": {"username": "supervisor1", "password": "1234"},
    "Field Officer": {"username": "fieldofficer1", "password": "1234"},
    "Admin": {"username": "admin1", "password": "1234"}
}

@router.post('/login')
def login(user: User):
    """
    Authenticate a user based on role, username, and password
    """
    role = user.role
    username = user.username
    password = user.password

    # Try database authentication first with improved retry mechanism
    max_retries = 5
    retry_count = 0
    db = None
    backoff_time = 0.2

    while retry_count < max_retries and db is None:
        db = connect_to_db()
        if db is None:
            retry_count += 1
            print(f"Database connection attempt {retry_count} failed, retrying in {backoff_time}s...")
            time.sleep(backoff_time)
            backoff_time = min(backoff_time * 2, 1.6)

    # If we have a database connection, proceed with authentication
    if db is not None:
        try:
            db_user = db.users.find_one({"username": username})
            if db_user and db_user.get("role") == role and check_password_hash(db_user.get("password"), password):
                try:
                    db.user_login_logs.insert_one({
                        "role": role,
                        "username": username,
                        "login_time": datetime.datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Error logging user login: {e}")
                
                return {
                    "success": True,
                    "message": "Login successful",
                    "user": {
                        "role": role,
                        "username": username
                    }
                }
        except Exception as e:
            print(f"Database error during authentication: {e}")

    # Fall back to hardcoded users for development
    if role in VALID_USERS:
        valid_user = VALID_USERS[role]
        if username == valid_user["username"] and password == valid_user["password"]:
            if db is not None:
                try:
                    db.user_login_logs.insert_one({
                        "role": role,
                        "username": username,
                        "login_time": datetime.datetime.now().isoformat()
                    })
                    if not db.users.find_one({"username": username}):
                        db.users.insert_one({
                            "role": role,
                            "username": username,
                            "password": generate_password_hash(password),
                            "created_at": datetime.datetime.now().isoformat()
                        })
                except Exception as e:
                    print(f"Error logging user login: {e}")
            
            return {
                "success": True,
                "message": "Login successful",
                "user": {
                    "role": role,
                    "username": username
                }
            }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@router.post('/logout')
def logout():
    """
    Handle user logout (in a real app, this would invalidate their session/token)
    """
    return {
        "success": True,
        "message": "Logout successful"
    }

@router.post('/register')
def register(user: User):
    """
    Register a new user
    """
    role = user.role
    username = user.username
    password = user.password

    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    existing_user = db.users.find_one({"username": username})
    if existing_user:
        raise HTTPException(status_code=409, detail="Username already exists")

    try:
        db.users.insert_one({
            "role": role,
            "username": username,
            "password": generate_password_hash(password),
            "created_at": datetime.datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "role": role,
                "username": username
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering user: {str(e)}")
