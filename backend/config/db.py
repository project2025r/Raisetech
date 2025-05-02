import pymongo
import os
from dotenv import load_dotenv
import gridfs

# Load environment variables from .env file
load_dotenv()

def get_db_connection():
    """
    Create a database connection to MongoDB
    """
    try:
        # Connect to MongoDB (Replace with your MongoDB URI if using a remote server)
        client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db = client.get_database(os.getenv("DB_NAME", "LTA"))
        return db
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return None

def connect_to_db():
    """Wrapper for get_db_connection for backward compatibility"""
    return get_db_connection()

def get_gridfs():
    """
    Get a GridFS instance for the database
    """
    db = connect_to_db()
    if db is not None:
        return gridfs.GridFS(db)
    return None

def create_collections():
    """
    Create all required collections if they don't exist
    """
    db = connect_to_db()
    if db is not None:
        try:
            # Create collections for user logs, potholes, and cracks
            db.create_collection("user_login_logs", ignore_existing=True)
            #db.create_collection("potholes", ignore_existing=True)
            #db.create_collection("cracks", ignore_existing=True)
            #db.create_collection("kerbs", ignore_existing=True)
            db.create_collection("recommendations", ignore_existing=True)
            
            # Create new collections for consolidated image-based storage
            db.create_collection("pothole_images", ignore_existing=True)
            db.create_collection("crack_images", ignore_existing=True)
            db.create_collection("kerb_images", ignore_existing=True)

            # Optionally, add indexes for faster queries
            db.user_login_logs.create_index([("username", pymongo.ASCENDING)])
            # db.potholes.create_index([("pothole_id", pymongo.ASCENDING)])
            # db.cracks.create_index([("crack_id", pymongo.ASCENDING)])
            # db.kerbs.create_index([("kerb_id", pymongo.ASCENDING)])
            db.recommendations.create_index([("location", pymongo.ASCENDING), ("issueType", pymongo.ASCENDING)])
            
            # Indexes for new collections
            db.pothole_images.create_index([("timestamp", pymongo.DESCENDING)])
            db.crack_images.create_index([("timestamp", pymongo.DESCENDING)])
            db.kerb_images.create_index([("timestamp", pymongo.DESCENDING)])

            print("✅ Collections created successfully")
        except Exception as e:
            print(f"❌ Error creating collections: {e}")
    else:
        print("❌ Could not create collections due to connection error")

# def insert_sample_data():
#     """
#     Insert some sample data into MongoDB collections with 'role' and 'username' as the first columns
#     """
#     db = connect_to_db()
#     if db is not None:
#         try:
#             # Sample data for user_login_logs
#             db.user_login_logs.insert_one({
#                 "role": "admin",
#                 "username": "john_doe",
#                 "login_time": "2025-04-18T12:00:00"
#             })

#             # Sample data for potholes
#             db.potholes.insert_one({
#                 "role": "admin",       # Added role (first column)
#                 "username": "admin1",  # Added username (second column)
#                 "pothole_id": 101,
#                 "area_cm2": 500.5,
#                 "depth_cm": 10.2,
#                 "volume": 5000.3,
#                 "volume_range": "Big (>10k)",
#                 "coordinates": "40.7128,-74.0060"
#             })

#             # Sample data for cracks
#             db.cracks.insert_one({
#                 "role": "admin",       # Added role (first column)
#                 "username": "admin1",  # Added username (second column)
#                 "crack_id": 201,
#                 "crack_type": "Alligator Crack",
#                 "area_cm2": 100.2,
#                 "area_range": "Medium (50-200)",
#                 "coordinates": "40.7128,-74.0060"
#             })
            
#             # Sample data for kerbs
#             db.kerbs.insert_one({
#                 "role": "admin",
#                 "username": "admin1",
#                 "kerb_id": 301,
#                 "kerb_type": "Concrete Kerb",
#                 "length_m": 15.7,
#                 "condition": "Damaged Kerbs",
#                 "confidence": 0.92,
#                 "coordinates": "40.7128,-74.0060"
#             })
            
#             # Sample data for recommendations
#             db.recommendations.insert_one({
#                 "location": "40.7128,-74.0060",
#                 "issueType": "Pothole (ID: 101)",
#                 "priority": "High",
#                 "estimatedCost": 2500.15,
#                 "status": "Pending",
#                 "details": "Volume: 5000.30 cm³, Category: Big (>10k)"
#             })

#             print("✅ Sample data inserted successfully")
#         except Exception as e:
#             print(f"❌ Error inserting data: {e}")
#     else:
#         print("❌ Could not insert data due to connection error")

# Initialize collections and insert sample data when this module is imported
if __name__ == "__main__":
    create_collections()  # Create collections
    insert_sample_data()  # Insert sample data