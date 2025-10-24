"""
File Upload Validation Utility

This module provides comprehensive file validation for image and video uploads
with proper exception handling and user-friendly error messages.

Features:
- MIME type validation
- File extension validation
- File size validation
- Content validation (basic file header checks)
- Graceful exception handling
- User-friendly error messages
"""

import os
import mimetypes
import logging
import base64
import io
from typing import Tuple, Optional, List, Union
from PIL import Image
from fastapi import UploadFile

# Configure logging
logger = logging.getLogger(__name__)

# Allowed file types configuration
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.avif'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}

ALLOWED_IMAGE_MIMES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 
    'image/tiff', 'image/tif', 'image/webp', 'image/avif'
}

ALLOWED_VIDEO_MIMES = {
    'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 
    'video/x-matroska', 'video/x-ms-wmv', 'video/x-flv', 'video/webm',
    'video/x-m4v', 'video/3gpp', 'video/ogg'
}

# File size limits (in bytes)
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB

# File signature checks for common formats
FILE_SIGNATURES = {
    # Images
    b'\xFF\xD8\xFF': 'image/jpeg',
    b'\x89PNG\r\n\x1a\n': 'image/png',
    b'GIF87a': 'image/gif',
    b'GIF89a': 'image/gif',
    b'BM': 'image/bmp',
    b'RIFF': 'image/webp',  # WebP files start with RIFF
    # Videos
    b'\x00\x00\x00\x18ftypmp4': 'video/mp4',
    b'\x00\x00\x00\x20ftypmp4': 'video/mp4',
    b'RIFF': 'video/avi',  # AVI files also start with RIFF
}


class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    def __init__(self, message: str, error_type: str = "validation_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename in lowercase
    
    Args:
        filename (str): The filename
        
    Returns:
        str: File extension in lowercase (including the dot)
    """
    if not filename:
        return ""
    return os.path.splitext(filename.lower())[1]


def get_mime_type(file_obj: UploadFile) -> str:
    """
    Get MIME type from file object
    
    Args:
        file_obj (FileStorage): The uploaded file object
        
    Returns:
        str: MIME type
    """
    # First try to get MIME type from the file object
    if hasattr(file_obj, 'content_type') and file_obj.content_type:
        return file_obj.content_type.lower()
    
    # Fallback to guessing from filename
    if file_obj.filename:
        mime_type, _ = mimetypes.guess_type(file_obj.filename)
        if mime_type:
            return mime_type.lower()
    
    return 'application/octet-stream'


async def check_file_signature(file_obj: UploadFile) -> Optional[str]:
    """
    Check file signature (magic bytes) to verify file type
    
    Args:
        file_obj (FileStorage): The uploaded file object
        
    Returns:
        Optional[str]: Detected MIME type based on signature, or None if not detected
    """
    try:
        # Save current position
        current_pos = file_obj.file.tell()
        
        # Read first 32 bytes for signature checking
        await file_obj.seek(0)
        header = await file_obj.read(32)
        
        # Restore position
        await file_obj.seek(current_pos)
        
        # Check against known signatures
        for signature, mime_type in FILE_SIGNATURES.items():
            if header.startswith(signature):
                return mime_type
                
        # Special case for WebP (needs more specific check)
        if header.startswith(b'RIFF') and b'WEBP' in header[:16]:
            return 'image/webp'
            
        # Special case for AVI (needs more specific check)
        if header.startswith(b'RIFF') and b'AVI ' in header[:16]:
            return 'video/avi'
            
        return None
        
    except Exception as e:
        logger.warning(f"Error checking file signature: {str(e)}")
        return None


async def validate_file_size(file_obj: UploadFile, file_type: str) -> None:
    """
    Validate file size based on file type
    
    Args:
        file_obj (FileStorage): The uploaded file object
        file_type (str): 'image' or 'video'
        
    Raises:
        FileValidationError: If file size exceeds limits
    """
    try:
        # Get file size
        current_pos = file_obj.file.tell()
        file_obj.file.seek(0, 2)  # Seek to end
        file_size = file_obj.file.tell()
        file_obj.file.seek(current_pos)  # Restore position
        
        # Check size limits
        if file_type == 'image' and file_size > MAX_IMAGE_SIZE:
            raise FileValidationError(
                f"Image file size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size ({MAX_IMAGE_SIZE / (1024*1024)}MB). Please compress your image or choose a smaller file.",
                "file_size_error"
            )
        elif file_type == 'video' and file_size > MAX_VIDEO_SIZE:
            raise FileValidationError(
                f"Video file size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size ({MAX_VIDEO_SIZE / (1024*1024)}MB). Please compress your video or choose a smaller file.",
                "file_size_error"
            )
            
    except FileValidationError:
        raise
    except Exception as e:
        logger.warning(f"Error checking file size: {str(e)}")
        # Don't fail validation just because we can't check size


async def validate_image_file(file_obj: UploadFile) -> Tuple[bool, str]:
    """
    Validate uploaded image file
    
    Args:
        file_obj (FileStorage): The uploaded file object
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        if not file_obj or not file_obj.filename:
            return False, "No image file provided."
        
        # Get file extension and MIME type
        file_extension = get_file_extension(file_obj.filename)
        mime_type = get_mime_type(file_obj)
        
        # Check file extension
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            return False, f"Invalid file format. Please upload only image files (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif). Received: {file_extension or 'unknown'}"
        
        # Check MIME type
        if mime_type not in ALLOWED_IMAGE_MIMES:
            return False, f"Invalid file format. Please upload only image files. The file appears to be: {mime_type}"
        
        # Validate file size
        await validate_file_size(file_obj, 'image')
        
        # Optional: Check file signature for additional security
        detected_mime = await check_file_signature(file_obj)
        if detected_mime and not detected_mime.startswith('image/'):
            return False, "Invalid file format. The file content does not match an image format. Please upload only image files."
        
        return True, ""
        
    except FileValidationError as e:
        return False, e.message
    except Exception as e:
        logger.error(f"Unexpected error during image validation: {str(e)}")
        return False, "An error occurred while validating the image file. Please try again with a different file."


async def validate_video_file(file_obj: UploadFile) -> Tuple[bool, str]:
    """
    Validate uploaded video file
    
    Args:
        file_obj (FileStorage): The uploaded file object
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        if not file_obj or not file_obj.filename:
            return False, "No video file provided."
        
        # Get file extension and MIME type
        file_extension = get_file_extension(file_obj.filename)
        mime_type = get_mime_type(file_obj)
        
        # Check file extension
        if file_extension not in ALLOWED_VIDEO_EXTENSIONS:
            return False, f"Invalid file format. Please upload only video files (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp, .ogv). Received: {file_extension or 'unknown'}"
        
        # Check MIME type
        if mime_type not in ALLOWED_VIDEO_MIMES:
            return False, f"Invalid file format. Please upload only video files. The file appears to be: {mime_type}"
        
        # Validate file size
        await validate_file_size(file_obj, 'video')
        
        # Optional: Check file signature for additional security
        detected_mime = await check_file_signature(file_obj)
        if detected_mime and not detected_mime.startswith('video/'):
            return False, "Invalid file format. The file content does not match a video format. Please upload only video files."
        
        return True, ""
        
    except FileValidationError as e:
        return False, e.message
    except Exception as e:
        logger.error(f"Unexpected error during video validation: {str(e)}")
        return False, "An error occurred while validating the video file. Please try again with a different file."


async def validate_upload_file(file_obj: UploadFile, expected_type: str) -> Tuple[bool, str]:
    """
    Main validation function for uploaded files
    
    Args:
        file_obj (FileStorage): The uploaded file object
        expected_type (str): Expected file type ('image' or 'video')
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        if expected_type == 'image':
            return await validate_image_file(file_obj)
        elif expected_type == 'video':
            return await validate_video_file(file_obj)
        else:
            return False, f"Invalid expected file type: {expected_type}. Must be 'image' or 'video'."
            
    except Exception as e:
        logger.error(f"Unexpected error during file validation: {str(e)}")
        return False, "An error occurred while validating the file. Please try again."


def validate_base64_image(image_data: str, context: str = "") -> Tuple[bool, str]:
    """
    Validate base64 encoded image data

    Args:
        image_data (str): Base64 encoded image data (with or without data URL prefix)
        context (str): Additional context for error messages

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        if not image_data:
            return False, "No image data provided."

        # Handle data URL format (data:image/jpeg;base64,...)
        if image_data.startswith('data:'):
            try:
                # Extract MIME type and base64 data
                header, base64_data = image_data.split(',', 1)
                mime_type = header.split(':')[1].split(';')[0].lower()

                # Check if MIME type is allowed for images
                if mime_type not in ALLOWED_IMAGE_MIMES:
                    return False, get_context_specific_error_message('image', context)

            except (ValueError, IndexError):
                return False, "Invalid image data format. Please provide a valid base64 encoded image."
        else:
            # Plain base64 data without data URL prefix
            base64_data = image_data

        # Decode base64 data
        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception as e:
            return False, "Invalid base64 image data. Please provide a valid base64 encoded image."

        # Check file size
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return False, f"Image size ({len(image_bytes) / (1024*1024):.1f}MB) exceeds maximum allowed size ({MAX_IMAGE_SIZE / (1024*1024)}MB). Please compress your image or choose a smaller file."

        # Try to open and validate the image using PIL
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Verify it's a valid image
                img.verify()

                # Check image format
                if hasattr(img, 'format') and img.format:
                    format_lower = img.format.lower()
                    # Map PIL formats to our allowed extensions
                    format_mapping = {
                        'jpeg': '.jpg',
                        'png': '.png',
                        'gif': '.gif',
                        'bmp': '.bmp',
                        'tiff': '.tiff',
                        'webp': '.webp'
                    }

                    if format_lower in format_mapping:
                        file_ext = format_mapping[format_lower]
                        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                            return False, get_context_specific_error_message('image', context)
                    else:
                        # Unknown format
                        return False, get_context_specific_error_message('image', context)

                # Check minimum image dimensions
                if hasattr(img, 'size'):
                    width, height = img.size
                    if width < 10 or height < 10:
                        return False, "Image is too small. Please provide an image with dimensions at least 10x10 pixels."

        except Exception as e:
            return False, "Invalid image data. The provided data does not represent a valid image file."

        return True, ""

    except Exception as e:
        logger.error(f"Unexpected error during base64 image validation: {str(e)}")
        return False, "An error occurred while validating the image data. Please try again with a different image."


def get_context_specific_error_message(expected_type: str, context: str = "") -> str:
    """
    Get context-specific error message for file validation

    Args:
        expected_type (str): Expected file type ('image' or 'video')
        context (str): Additional context (e.g., 'pothole_detection', 'crack_detection')

    Returns:
        str: Context-specific error message
    """
    if expected_type == 'image':
        if 'pothole' in context.lower():
            return "Invalid file format. Please upload only image files for pothole detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif)."
        elif 'crack' in context.lower():
            return "Invalid file format. Please upload only image files for crack detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif)."
        elif 'kerb' in context.lower():
            return "Invalid file format. Please upload only image files for kerb detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif)."
        else:
            return "Invalid file format. Please upload only image files (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif)."
    elif expected_type == 'video':
        return "Invalid file format. Please upload only video files (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp, .ogv)."
    else:
        return "Invalid file format. Please upload only image or video files."