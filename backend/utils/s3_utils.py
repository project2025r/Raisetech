import boto3
import os
import logging
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from botocore.config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class S3Handler:
    """
    Handle S3 operations for work_status images
    """
    
    def __init__(self):
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket_name = os.getenv('S3_BUCKET_NAME', 'aispry-project')
        self.s3_base_folder = os.getenv('S3_BASE_FOLDER', '2024_Oct_YNMSafety_RoadSafetyAudit/Civion/Live_Project_Data')
        self.s3_work_status_folder = os.getenv('S3_WORK_STATUS_FOLDER', 'work_status')
        self.s3_full_work_status_path = f"{self.s3_base_folder}/{self.s3_work_status_folder}"
        
        # Initialize S3 client
        self.s3_client = None
        self._init_s3_client()
    
    def _init_s3_client(self):
        """Initialize S3 client with credentials and s3v4 signature"""
        try:
            if self.aws_access_key_id and self.aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.aws_region,
                    config=Config(signature_version='s3v4')  # Required for newer regions
                )
                logger.info("✅ S3 client initialized successfully with s3v4 signature")
                logger.info(f"✅ S3 Configuration: Bucket={self.s3_bucket_name}, Region={self.aws_region}")
                logger.info(f"✅ Work Status Path: {self.s3_full_work_status_path}")
            else:
                logger.warning("⚠️ S3 credentials not found in environment variables")
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {str(e)}")
    
    def generate_presigned_url(self, object_key, expiration=3600):
        """
        Generate a presigned URL for S3 object
        
        Args:
            object_key (str): S3 object key/path (relative to work_status folder)
            expiration (int): URL expiration time in seconds (default: 1 hour)
            
        Returns:
            str: Presigned URL or None if error
        """
        if not self.s3_client or not self.s3_bucket_name:
            logger.error("❌ S3 client not initialized or bucket name not configured")
            logger.error(f"   S3 client: {self.s3_client is not None}")
            logger.error(f"   Bucket name: {self.s3_bucket_name}")
            return None
        
        try:
            # If object_key already contains the full path, use it as-is
            # Otherwise, prepend the base folder structure
            if object_key.startswith(self.s3_base_folder):
                full_object_key = object_key
                logger.info(f"🔗 Using full path as-is: {full_object_key}")
            else:
                # Assume object_key is relative to work_status folder
                full_object_key = f"{self.s3_full_work_status_path}/{object_key}"
                logger.info(f"🔗 Constructed full path: {full_object_key}")
            
            logger.info(f"🚀 Generating presigned URL for: {full_object_key}")
            logger.info(f"   Bucket: {self.s3_bucket_name}")
            logger.info(f"   Expiration: {expiration} seconds")
            
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket_name, 'Key': full_object_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"✅ Generated presigned URL successfully")
            logger.info(f"   URL length: {len(presigned_url)} characters")
            logger.info(f"   URL preview: {presigned_url[:100]}...")
            
            return presigned_url
        except ClientError as e:
            logger.error(f"❌ AWS ClientError generating presigned URL for {object_key}: {str(e)}")
            logger.error(f"   Error code: {e.response.get('Error', {}).get('Code', 'Unknown')}")
            logger.error(f"   Error message: {e.response.get('Error', {}).get('Message', 'Unknown')}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error generating presigned URL for {object_key}: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")
            return None
    
    def get_multiple_presigned_urls(self, object_keys, expiration=3600):
        """
        Generate presigned URLs for multiple objects
        
        Args:
            object_keys (list): List of S3 object keys/paths
            expiration (int): URL expiration time in seconds
            
        Returns:
            dict: Dictionary with object_key as key and presigned_url as value
        """
        urls = {}
        for key in object_keys:
            if key:  # Only process non-empty keys
                url = self.generate_presigned_url(key, expiration)
                if url:
                    urls[key] = url
        return urls
    
    def check_object_exists(self, object_key):
        """
        Check if an object exists in S3
        
        Args:
            object_key (str): S3 object key/path (relative to work_status folder)
            
        Returns:
            bool: True if object exists, False otherwise
        """
        if not self.s3_client or not self.s3_bucket_name:
            logger.warning(f"⚠️ Cannot check object existence - S3 not configured")
            return False
        
        try:
            # If object_key already contains the full path, use it as-is
            # Otherwise, prepend the base folder structure
            if object_key.startswith(self.s3_base_folder):
                full_object_key = object_key
            else:
                # Assume object_key is relative to work_status folder
                full_object_key = f"{self.s3_full_work_status_path}/{object_key}"
            
            logger.info(f"🔍 Checking if object exists: {full_object_key}")
            self.s3_client.head_object(Bucket=self.s3_bucket_name, Key=full_object_key)
            logger.info(f"✅ Object exists: {full_object_key}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                logger.warning(f"❌ Object not found: {full_object_key}")
            else:
                logger.error(f"❌ Error checking object existence: {error_code} - {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error checking object existence: {str(e)}")
            return False
    
    def is_configured(self):
        """
        Check if S3 is properly configured
        
        Returns:
            bool: True if S3 is configured, False otherwise
        """
        return (self.s3_client is not None and 
                self.s3_bucket_name is not None and 
                self.aws_access_key_id is not None and 
                self.aws_secret_access_key is not None and
                self.s3_base_folder is not None and
                self.s3_work_status_folder is not None)
    
    def get_full_path(self, relative_path):
        """
        Get the full S3 path for a given relative path
        
        Args:
            relative_path (str): Relative path from work_status folder
            
        Returns:
            str: Full S3 path
        """
        if relative_path.startswith(self.s3_base_folder):
            return relative_path
        else:
            return f"{self.s3_full_work_status_path}/{relative_path}"
    
    def get_config_info(self):
        """
        Get S3 configuration information for debugging
        
        Returns:
            dict: Configuration information
        """
        return {
            "bucket_name": self.s3_bucket_name,
            "region": self.aws_region,
            "base_folder": self.s3_base_folder,
            "work_status_folder": self.s3_work_status_folder,
            "full_work_status_path": self.s3_full_work_status_path,
            "credentials_set": {
                "aws_access_key_id": "SET" if self.aws_access_key_id else "NOT SET",
                "aws_secret_access_key": "SET" if self.aws_secret_access_key else "NOT SET"
            },
            "s3_client_initialized": self.s3_client is not None
        }

# Global S3 handler instance
s3_handler = S3Handler()

def get_s3_handler():
    """Get the global S3 handler instance"""
    return s3_handler

def get_work_status_image_url(image_path):
    """
    Get presigned URL for work_status image
    
    Args:
        image_path (str): Path to the image in S3
        
    Returns:
        str: Presigned URL or None if error
    """
    if not image_path:
        return None
    
    handler = get_s3_handler()
    if not handler.is_configured():
        logger.warning("S3 not configured, cannot generate image URL")
        return None
    
    return handler.generate_presigned_url(image_path)

def get_work_status_image_urls(image_paths):
    """
    Get presigned URLs for multiple work_status images
    
    Args:
        image_paths (list): List of image paths in S3
        
    Returns:
        dict: Dictionary with image_path as key and presigned_url as value
    """
    if not image_paths:
        return {}
    
    handler = get_s3_handler()
    if not handler.is_configured():
        logger.warning("S3 not configured, cannot generate image URLs")
        return {}
    
    return handler.get_multiple_presigned_urls(image_paths) 