
import os
import sys
import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import logging
from datetime import datetime
import uuid
from config.db import connect_to_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3ImageManager:
    """
    Comprehensive S3 Image Management Class
    Handles all S3 operations for image storage and retrieval
    """
    
    def __init__(self):
        """Initialize S3 client and configuration"""
        self.aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        self.aws_folder = os.environ.get('AWS_FOLDER', 'aispry-project/2024_Oct_YNMSafety_RoadSafetyAudit/audit/raisetech')
        
        # Extract bucket and prefix from aws_folder
        self.aws_folder = self.aws_folder.strip('/')
        parts = self.aws_folder.split('/', 1)
        self.bucket = parts[0]
        self.prefix = parts[1] if len(parts) > 1 else ''
        
        # Initialize S3 client
        self.s3_client = None
        self._initialize_s3_client()
    
    def _initialize_s3_client(self):
        """Initialize S3 client with error handling"""
        try:
            if not self.aws_access_key or not self.aws_secret_key:
                raise ValueError("AWS credentials not found in environment variables")
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.aws_region
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"✅ S3 client initialized successfully for bucket: {self.bucket}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {str(e)}")
            raise
    
    def upload_image_to_s3(self, image_buffer, s3_key, content_type='image/jpeg'):
        """
        Upload an image buffer to S3
        
        Args:
            image_buffer (bytes): Image data as bytes
            s3_key (str): S3 key path for the image
            content_type (str): MIME type for the image
            
        Returns:
            tuple: (success: bool, s3_url_or_error: str)
        """
        try:
            # Compose full S3 key with prefix
            full_s3_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key
            
            # Upload image to S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=full_s3_key,
                Body=image_buffer,
                ContentType=content_type
            )
            
            # Return relative S3 path (not full URL)
            logger.info(f"✅ Image uploaded successfully to S3: {s3_key}")
            return True, s3_key
            
        except ClientError as e:
            error_msg = f"S3 upload error: {e}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during S3 upload: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def upload_images_to_s3(self, original_image, processed_image, image_id, role, username):
        """
        Upload both original and processed images to S3 with organized folder structure
        
        Args:
            original_image (np.ndarray): Original image as numpy array
            processed_image (np.ndarray): Processed image as numpy array
            image_id (str): Unique identifier for this image upload
            role (str): User role for folder structure
            username (str): Username for folder structure
            
        Returns:
            tuple: (original_s3_url: str, processed_s3_url: str, success: bool, error_msg: str)
        """
        try:
            # Encode images to JPEG format
            _, original_buffer = cv2.imencode('.jpg', original_image)
            _, processed_buffer = cv2.imencode('.jpg', processed_image)
            
            # Create S3 keys with organized folder structure
            # Structure: {role}/{username}/original/image_{id}.jpg
            #           {role}/{username}/processed/image_{id}.jpg
            original_s3_key = f"{role}/{username}/original/image_{image_id}.jpg"
            processed_s3_key = f"{role}/{username}/processed/image_{image_id}.jpg"
            
            # Upload original image
            original_success, original_result = self.upload_image_to_s3(
                original_buffer.tobytes(), original_s3_key
            )
            
            if not original_success:
                return None, None, False, f"Failed to upload original image: {original_result}"
            
            # Upload processed image
            processed_success, processed_result = self.upload_image_to_s3(
                processed_buffer.tobytes(), processed_s3_key
            )
            
            if not processed_success:
                return None, None, False, f"Failed to upload processed image: {processed_result}"
            
            logger.info(f"✅ Successfully uploaded both images to S3: {original_s3_key}, {processed_s3_key}")
            return original_s3_key, processed_s3_key, True, None
            
        except Exception as e:
            error_msg = f"Error uploading images to S3: {e}"
            logger.error(error_msg)
            return None, None, False, error_msg
    
    def generate_s3_url(self, s3_key):
        """
        Generate a public S3 URL from an S3 key
        
        Args:
            s3_key (str): S3 key path
            
        Returns:
            str: Full S3 URL
        """
        if not s3_key:
            return None
        
        # Compose full S3 key with prefix
        full_s3_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key
        
        # Generate public URL
        return f"https://{self.bucket}.s3.{self.aws_region}.amazonaws.com/{full_s3_key}"
    
    def download_image_from_s3(self, s3_key):
        """
        Download an image from S3

        Args:
            s3_key (str): S3 key path

        Returns:
            bytes: Image data or None if not found
        """
        try:
            full_s3_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key
            
            response = self.s3_client.get_object(Bucket=self.bucket, Key=full_s3_key)
            
            return response['Body'].read()
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"Image not found at S3 key: {full_s3_key}")
                return None
            else:
                logger.error(f"Error downloading image from S3: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error downloading image from S3: {e}")
            return None


class MongoDBImageManager:
    """
    MongoDB Image Management Class
    Handles all MongoDB operations for image metadata storage and retrieval
    """
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.db = None
        self._initialize_db_connection()
    
    def _initialize_db_connection(self):
        """Initialize MongoDB connection with error handling"""
        try:
            self.db = connect_to_db()
            if self.db is None:
                raise ConnectionError("Failed to connect to MongoDB")
            
            logger.info("✅ MongoDB connection established successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MongoDB connection: {str(e)}")
            raise
    
    def insert_image_metadata(self, collection_name, image_data):
        """
        Insert image metadata with S3 URLs into MongoDB
        
        Args:
            collection_name (str): MongoDB collection name ('pothole_images', 'crack_images', 'kerb_images')
            image_data (dict): Image metadata including S3 URLs
            
        Returns:
            tuple: (success: bool, document_id_or_error: str)
        """
        try:
            # Validate required fields
            required_fields = ['image_id', 'original_image_s3_url', 'processed_image_s3_url']
            missing_fields = [field for field in required_fields if field not in image_data]
            
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}"
            
            # Get collection
            collection = self.db[collection_name]
            
            # Insert document
            result = collection.insert_one(image_data)
            
            logger.info(f"✅ Image metadata inserted successfully into {collection_name}: {result.inserted_id}")
            return True, str(result.inserted_id)
            
        except Exception as e:
            error_msg = f"Error inserting image metadata: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def fetch_image_metadata(self, collection_name, query=None, limit=None, sort_field='_id', sort_order=-1):
        """
        Fetch image metadata from MongoDB
        
        Args:
            collection_name (str): MongoDB collection name
            query (dict): MongoDB query filter (optional)
            limit (int): Maximum number of documents to return (optional)
            sort_field (str): Field to sort by (default: '_id')
            sort_order (int): Sort order (1 for ascending, -1 for descending)
            
        Returns:
            tuple: (success: bool, documents_or_error: list/str)
        """
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Build query
            if query is None:
                query = {}
            
            # Execute query
            cursor = collection.find(query).sort(sort_field, sort_order)
            
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert cursor to list
            documents = list(cursor)
            
            logger.info(f"✅ Fetched {len(documents)} documents from {collection_name}")
            return True, documents
            
        except Exception as e:
            error_msg = f"Error fetching image metadata: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_image_by_id(self, collection_name, image_id):
        """
        Get a specific image by its ID
        
        Args:
            collection_name (str): MongoDB collection name
            image_id (str): Image ID to search for
            
        Returns:
            tuple: (success: bool, document_or_error: dict/str)
        """
        try:
            # Get collection
            collection = self.db[collection_name]
            
            # Find document
            document = collection.find_one({"image_id": image_id})
            
            if document:
                logger.info(f"✅ Found image with ID {image_id} in {collection_name}")
                return True, document
            else:
                return False, f"Image with ID {image_id} not found in {collection_name}"
                
        except Exception as e:
            error_msg = f"Error getting image by ID: {e}"
            logger.error(error_msg)
            return False, error_msg


class DashboardImageManager:
    """
    Dashboard Image Management Class
    Handles image display logic for the dashboard
    """

    def __init__(self):
        """Initialize S3 and MongoDB managers"""
        self.s3_manager = S3ImageManager()
        self.mongodb_manager = MongoDBImageManager()

    def prepare_image_data_for_dashboard(self, collection_name, limit=50):
        """
        Prepare image data for dashboard display with both S3 URLs and GridFS fallback

        Args:
            collection_name (str): MongoDB collection name
            limit (int): Maximum number of images to fetch

        Returns:
            tuple: (success: bool, prepared_data_or_error: list/str)
        """
        try:
            # Fetch image metadata from MongoDB
            success, documents = self.mongodb_manager.fetch_image_metadata(
                collection_name, limit=limit
            )

            if not success:
                return False, documents  # documents contains error message

            prepared_data = []

            for doc in documents:
                # Prepare image data with both S3 and GridFS support
                image_data = {
                    'image_id': doc.get('image_id'),
                    'timestamp': doc.get('timestamp'),
                    'username': doc.get('username'),
                    'role': doc.get('role'),
                    'coordinates': doc.get('coordinates'),
                }

                # Add S3 URLs if available (new data)
                original_s3_key = doc.get('original_image_s3_url')
                processed_s3_key = doc.get('processed_image_s3_url')

                if original_s3_key:
                    image_data['original_image_s3_url'] = original_s3_key
                    image_data['original_image_full_url'] = self.s3_manager.generate_s3_url(original_s3_key)

                if processed_s3_key:
                    image_data['processed_image_s3_url'] = processed_s3_key
                    image_data['processed_image_full_url'] = self.s3_manager.generate_s3_url(processed_s3_key)

                # Add GridFS IDs for backward compatibility (old data)
                image_data['original_image_id'] = doc.get('original_image_id')
                image_data['processed_image_id'] = doc.get('processed_image_id')

                # Add defect-specific data and flatten structure for dashboard
                if collection_name == 'pothole_images':
                    potholes = doc.get('potholes', [])
                    if potholes:
                        # Create separate entries for each pothole
                        for pothole in potholes:
                            pothole_data = image_data.copy()
                            pothole_data.update(pothole)
                            pothole_data['pothole_count'] = doc.get('pothole_count', 0)
                            prepared_data.append(pothole_data)
                    else:
                        # No potholes, add the image data as is
                        image_data['pothole_count'] = doc.get('pothole_count', 0)
                        image_data['potholes'] = []
                        prepared_data.append(image_data)

                elif collection_name == 'crack_images':
                    cracks = doc.get('cracks', [])
                    if cracks:
                        # Create separate entries for each crack
                        for crack in cracks:
                            crack_data = image_data.copy()
                            crack_data.update(crack)
                            crack_data['crack_count'] = doc.get('crack_count', 0)
                            prepared_data.append(crack_data)
                    else:
                        # No cracks, add the image data as is
                        image_data['crack_count'] = doc.get('crack_count', 0)
                        image_data['cracks'] = []
                        prepared_data.append(image_data)

                elif collection_name == 'kerb_images':
                    kerbs = doc.get('kerbs', [])
                    if kerbs:
                        # Create separate entries for each kerb
                        for kerb in kerbs:
                            kerb_data = image_data.copy()
                            kerb_data.update(kerb)
                            kerb_data['kerb_count'] = doc.get('kerb_count', 0)
                            prepared_data.append(kerb_data)
                    else:
                        # No kerbs, add the image data as is
                        image_data['kerb_count'] = doc.get('kerb_count', 0)
                        image_data['kerbs'] = []
                        prepared_data.append(image_data)

            logger.info(f"✅ Prepared {len(prepared_data)} image records for dashboard from {collection_name}")
            return True, prepared_data

        except Exception as e:
            error_msg = f"Error preparing image data for dashboard: {e}"
            logger.error(error_msg)
            return False, error_msg

    def get_image_url_for_display(self, image_data, image_type='original'):
        """
        Get the appropriate image URL for display (S3 or GridFS fallback)

        Args:
            image_data (dict): Image metadata from MongoDB
            image_type (str): 'original' or 'processed'

        Returns:
            str: Image URL for display or None if not available
        """
        try:
            # Try S3 full URL first (new data)
            full_url_field = f"{image_type}_image_full_url"
            if image_data.get(full_url_field):
                return image_data[full_url_field]

            # Try S3 key with URL generation (new data without full URL)
            s3_key_field = f"{image_type}_image_s3_url"
            if image_data.get(s3_key_field):
                return self.s3_manager.generate_s3_url(image_data[s3_key_field])

            # Fall back to GridFS endpoint (old data)
            gridfs_id_field = f"{image_type}_image_id"
            if image_data.get(gridfs_id_field):
                return f"/api/pavement/get-image/{image_data[gridfs_id_field]}"

            # No image URL available
            return None

        except Exception as e:
            logger.error(f"Error getting image URL for display: {e}")
            return None


class ImageProcessingWorkflow:
    """
    Complete Image Processing Workflow Class
    Handles the entire workflow from upload to dashboard display
    """

    def __init__(self):
        """Initialize all managers"""
        self.s3_manager = S3ImageManager()
        self.mongodb_manager = MongoDBImageManager()
        self.dashboard_manager = DashboardImageManager()

    def process_and_store_images(self, original_image, processed_image, metadata, defect_results, defect_type):
        """
        Complete workflow: Process images, upload to S3, and store metadata in MongoDB

        Args:
            original_image (np.ndarray): Original image as numpy array
            processed_image (np.ndarray): Processed image with detections
            metadata (dict): Image metadata (username, role, coordinates, timestamp)
            defect_results (list): Detection results (potholes, cracks, or kerbs)
            defect_type (str): Type of defect ('pothole', 'crack', 'kerb')

        Returns:
            tuple: (success: bool, result_data_or_error: dict/str)
        """
        try:
            # Generate unique image ID
            image_id = str(uuid.uuid4())

            # Step 1: Upload images to S3
            logger.info(f"🔄 Step 1: Uploading images to S3 for {defect_type} detection...")
            original_s3_url, processed_s3_url, upload_success, upload_error = self.s3_manager.upload_images_to_s3(
                original_image, processed_image, image_id, metadata['role'], metadata['username']
            )

            if not upload_success:
                return False, f"Failed to upload images to S3: {upload_error}"

            # Step 2: Prepare MongoDB document
            logger.info(f"🔄 Step 2: Preparing MongoDB document...")
            mongo_document = {
                'image_id': image_id,
                'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                'coordinates': metadata.get('coordinates'),
                'username': metadata['username'],
                'role': metadata['role'],
                'original_image_s3_url': original_s3_url,
                'processed_image_s3_url': processed_s3_url,
                # Include EXIF data and metadata
                'exif_data': metadata.get('exif_data', {}),
                'metadata': metadata.get('metadata', {}),
                'media_type': metadata.get('media_type', 'image'),
            }

            # Add defect-specific data
            if defect_type == 'pothole':
                mongo_document['pothole_count'] = len(defect_results)
                mongo_document['potholes'] = defect_results
                collection_name = 'pothole_images'
            elif defect_type == 'crack':
                mongo_document['crack_count'] = len(defect_results)
                mongo_document['cracks'] = defect_results
                collection_name = 'crack_images'
            elif defect_type == 'kerb':
                mongo_document['kerb_count'] = len(defect_results)
                mongo_document['kerbs'] = defect_results
                collection_name = 'kerb_images'
            else:
                return False, f"Unknown defect type: {defect_type}"

            # Step 3: Store metadata in MongoDB
            logger.info(f"🔄 Step 3: Storing metadata in MongoDB...")
            mongo_success, mongo_result = self.mongodb_manager.insert_image_metadata(
                collection_name, mongo_document
            )

            if not mongo_success:
                return False, f"Failed to store metadata in MongoDB: {mongo_result}"

            # Step 4: Prepare result data
            result_data = {
                'image_id': image_id,
                'original_s3_url': original_s3_url,
                'processed_s3_url': processed_s3_url,
                'original_full_url': self.s3_manager.generate_s3_url(original_s3_url),
                'processed_full_url': self.s3_manager.generate_s3_url(processed_s3_url),
                'mongodb_id': mongo_result,
                'defect_count': len(defect_results),
                'defects': defect_results
            }

            logger.info(f"✅ Complete workflow successful for {defect_type} detection: {image_id}")
            return True, result_data

        except Exception as e:
            error_msg = f"Error in image processing workflow: {e}"
            logger.error(error_msg)
            return False, error_msg

    def get_dashboard_data(self, defect_types=['pothole', 'crack', 'kerb'], limit_per_type=50):
        """
        Get comprehensive dashboard data for all defect types

        Args:
            defect_types (list): List of defect types to fetch
            limit_per_type (int): Maximum number of records per defect type

        Returns:
            tuple: (success: bool, dashboard_data_or_error: dict/str)
        """
        try:
            dashboard_data = {}

            for defect_type in defect_types:
                collection_name = f"{defect_type}_images"

                logger.info(f"🔄 Fetching dashboard data for {defect_type}s...")
                success, data = self.dashboard_manager.prepare_image_data_for_dashboard(
                    collection_name, limit=limit_per_type
                )

                if success:
                    dashboard_data[f"{defect_type}s"] = {
                        'latest': data,
                        'count': len(data)
                    }
                else:
                    logger.warning(f"⚠️  Failed to fetch {defect_type} data: {data}")
                    dashboard_data[f"{defect_type}s"] = {
                        'latest': [],
                        'count': 0,
                        'error': data
                    }

            logger.info(f"✅ Dashboard data prepared successfully")
            return True, dashboard_data

        except Exception as e:
            error_msg = f"Error getting dashboard data: {e}"
            logger.error(error_msg)
            return False, error_msg
