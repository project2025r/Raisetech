#!/usr/bin/env python3
"""
Image format conversion utilities
Handles conversion of various image formats to YOLO-supported formats
"""

import io
import base64
import imageio
from PIL import Image
import numpy as np
import logging
import os
import subprocess
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_avif_to_jpg_base64(avif_data_url):
    """
    Convert AVIF image from base64 data URL to JPG base64 data URL
    
    Args:
        avif_data_url (str): Base64 data URL of AVIF image
        
    Returns:
        str: Base64 data URL of converted JPG image
    """
    try:
        # Extract base64 data from data URL
        if avif_data_url.startswith('data:image/avif;base64,'):
            base64_data = avif_data_url.split(',')[1]
        elif avif_data_url.startswith('data:image/;base64,'):
            base64_data = avif_data_url.split(',')[1]
        else:
            # Assume it's already base64 data
            base64_data = avif_data_url
            
        # Decode base64 to bytes
        avif_bytes = base64.b64decode(base64_data)
        
        # Try multiple methods to read AVIF image
        img_array = None
        
        # Method 1: Try ffmpeg subprocess (most reliable on EC2)
        try:
            logger.info("Attempting to read AVIF with ffmpeg subprocess...")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.avif', delete=False) as avif_temp:
                avif_temp.write(avif_bytes)
                avif_temp_path = avif_temp.name
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as jpg_temp:
                jpg_temp_path = jpg_temp.name
            
            # Use ffmpeg to convert AVIF to JPG
            cmd = ['ffmpeg', '-i', avif_temp_path, '-y', jpg_temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Read the converted JPG
                with open(jpg_temp_path, 'rb') as f:
                    jpg_bytes = f.read()
                
                # Convert to numpy array
                pil_img = Image.open(io.BytesIO(jpg_bytes))
                img_array = np.array(pil_img)
                logger.info(f"Successfully read AVIF image with ffmpeg, shape: {img_array.shape}")
                
                # Clean up temp files
                os.unlink(avif_temp_path)
                os.unlink(jpg_temp_path)
            else:
                logger.warning(f"ffmpeg failed: {result.stderr}")
                # Clean up temp files
                os.unlink(avif_temp_path)
                os.unlink(jpg_temp_path)
                raise ValueError(f"ffmpeg conversion failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"ffmpeg subprocess failed: {str(e)}")
            # Clean up temp files if they exist
            try:
                if 'avif_temp_path' in locals():
                    os.unlink(avif_temp_path)
                if 'jpg_temp_path' in locals():
                    os.unlink(jpg_temp_path)
            except:
                pass
            
            # Fallback to other methods
            img_array = None
        
        # Method 2: Try PIL with AVIF plugin (fallback)
        if img_array is None:
            try:
                logger.info("Attempting to read AVIF with PIL...")
                pil_img = Image.open(io.BytesIO(avif_bytes))
                if pil_img.mode in ('RGBA', 'LA', 'P'):
                    pil_img = pil_img.convert('RGB')
                img_array = np.array(pil_img)
                logger.info(f"Successfully read AVIF image with PIL, shape: {img_array.shape}")
            except Exception as e:
                logger.warning(f"PIL failed: {str(e)}")
                 
        # Method 3: Try imageio (fallback)
        if img_array is None:
            try:
                logger.info("Attempting to read AVIF with imageio...")
                img_array = imageio.imread(io.BytesIO(avif_bytes))
                logger.info(f"Successfully read AVIF image with imageio, shape: {img_array.shape}")
            except Exception as e:
                logger.warning(f"imageio failed: {str(e)}")
        
        if img_array is None:
            raise ValueError("All methods failed to read AVIF image")
        
        # Convert to PIL Image for easier manipulation
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA image, convert to RGB
            logger.info("Converting RGBA to RGB")
            pil_img = Image.fromarray(img_array, 'RGBA')
            pil_img = pil_img.convert('RGB')
        else:
            pil_img = Image.fromarray(img_array)
        
        # Convert to JPG bytes in memory
        jpg_buffer = io.BytesIO()
        pil_img.save(jpg_buffer, format='JPEG', quality=95, optimize=True)
        jpg_bytes = jpg_buffer.getvalue()
        
        # Convert to base64
        jpg_base64 = base64.b64encode(jpg_bytes).decode('utf-8')
        jpg_data_url = f"data:image/jpeg;base64,{jpg_base64}"
        
        logger.info(f"Successfully converted AVIF to JPG. JPG size: {len(jpg_bytes)} bytes")
        return jpg_data_url
        
    except Exception as e:
        logger.error(f"Error converting AVIF to JPG: {str(e)}")
        raise ValueError(f"Failed to convert AVIF image: {str(e)}")

def convert_avif_to_jpg_bytes(avif_bytes):
    """
    Convert AVIF image bytes to JPG bytes
    
    Args:
        avif_bytes (bytes): Raw AVIF image bytes
        
    Returns:
        bytes: JPG image bytes
    """
    try:
        # Try multiple methods to read AVIF image
        img_array = None
        
        # Method 1: Try ffmpeg subprocess (most reliable on EC2)
        try:
            logger.info("Attempting to read AVIF with ffmpeg subprocess...")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.avif', delete=False) as avif_temp:
                avif_temp.write(avif_bytes)
                avif_temp_path = avif_temp.name
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as jpg_temp:
                jpg_temp_path = jpg_temp.name
            
            # Use ffmpeg to convert AVIF to JPG
            cmd = ['ffmpeg', '-i', avif_temp_path, '-y', jpg_temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Read the converted JPG
                with open(jpg_temp_path, 'rb') as f:
                    jpg_bytes = f.read()
                
                # Convert to numpy array
                pil_img = Image.open(io.BytesIO(jpg_bytes))
                img_array = np.array(pil_img)
                logger.info(f"Successfully read AVIF image with ffmpeg, shape: {img_array.shape}")
                
                # Clean up temp files
                os.unlink(avif_temp_path)
                os.unlink(jpg_temp_path)
            else:
                logger.warning(f"ffmpeg failed: {result.stderr}")
                # Clean up temp files
                os.unlink(avif_temp_path)
                os.unlink(jpg_temp_path)
                raise ValueError(f"ffmpeg conversion failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"ffmpeg subprocess failed: {str(e)}")
            # Clean up temp files if they exist
            try:
                if 'avif_temp_path' in locals():
                    os.unlink(avif_temp_path)
                if 'jpg_temp_path' in locals():
                    os.unlink(jpg_temp_path)
            except:
                pass
            
            # Fallback to other methods
            img_array = None
        
        # Method 2: Try PIL with AVIF plugin (fallback)
        if img_array is None:
            try:
                logger.info("Attempting to read AVIF with PIL...")
                pil_img = Image.open(io.BytesIO(avif_bytes))
                if pil_img.mode in ('RGBA', 'LA', 'P'):
                    pil_img = pil_img.convert('RGB')
                img_array = np.array(pil_img)
                logger.info(f"Successfully read AVIF image with PIL, shape: {img_array.shape}")
            except Exception as e:
                logger.warning(f"PIL failed: {str(e)}")
                 
        # Method 3: Try imageio (fallback)
        if img_array is None:
            try:
                logger.info("Attempting to read AVIF with imageio...")
                img_array = imageio.imread(io.BytesIO(avif_bytes))
                logger.info(f"Successfully read AVIF image with imageio, shape: {img_array.shape}")
            except Exception as e:
                logger.warning(f"imageio failed: {str(e)}")
        
        if img_array is None:
            raise ValueError("All methods failed to read AVIF image")
        
        # Convert to PIL Image
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA image, convert to RGB
            logger.info("Converting RGBA to RGB")
            pil_img = Image.fromarray(img_array, 'RGBA')
            pil_img = pil_img.convert('RGB')
        else:
            pil_img = Image.fromarray(img_array)
        
        # Convert to JPG bytes
        jpg_buffer = io.BytesIO()
        pil_img.save(jpg_buffer, format='JPEG', quality=95, optimize=True)
        jpg_bytes = jpg_buffer.getvalue()
        
        logger.info(f"Successfully converted AVIF to JPG. JPG size: {len(jpg_bytes)} bytes")
        return jpg_bytes
        
    except Exception as e:
        logger.error(f"Error converting AVIF to JPG: {str(e)}")
        raise ValueError(f"Failed to convert AVIF image: {str(e)}")

def is_avif_image(image_data):
    """
    Check if the image data is in AVIF format
    
    Args:
        image_data (str or bytes): Image data (base64 data URL or bytes)
        
    Returns:
        bool: True if image is AVIF format
    """
    try:
        if isinstance(image_data, str):
            # Check if it's a base64 data URL
            if image_data.startswith('data:image/avif;base64,'):
                return True
            elif image_data.startswith('data:image/;base64,'):
                # Could be AVIF, need to check the actual data
                base64_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
            else:
                return False
        else:
            image_bytes = image_data
        
        # Check if it's AVIF by trying to read it
        try:
            # Try PIL first
            pil_img = Image.open(io.BytesIO(image_bytes))
            return True
        except:
            pass
        
        # Try imageio
        try:
            img_array = imageio.imread(io.BytesIO(image_bytes))
            return True
        except:
            pass
        
        return False
        
    except Exception as e:
        logger.warning(f"Error checking AVIF format: {str(e)}")
        return False

def get_image_info(image_data):
    """
    Get information about an image
    
    Args:
        image_data (str or bytes): Image data
        
    Returns:
        dict: Image information
    """
    try:
        if isinstance(image_data, str) and image_data.startswith('data:'):
            # Extract base64 data
            base64_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
        else:
            image_bytes = image_data
        
        # Try to read image
        try:
            # Try PIL first
            pil_img = Image.open(io.BytesIO(image_bytes))
            info = {
                'size': pil_img.size,
                'mode': pil_img.mode,
                'format': pil_img.format,
                'size_bytes': len(image_bytes),
                'width': pil_img.size[0],
                'height': pil_img.size[1]
            }
            return info
        except:
            pass
        
        # Try imageio
        try:
            img_array = imageio.imread(io.BytesIO(image_bytes))
            info = {
                'shape': img_array.shape,
                'dtype': str(img_array.dtype),
                'size_bytes': len(image_bytes),
                'channels': img_array.shape[2] if len(img_array.shape) == 3 else 1,
                'width': img_array.shape[1],
                'height': img_array.shape[0]
            }
            return info
        except:
            pass
        
        return {'error': 'Could not read image'}
        
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return {'error': str(e)}

def convert_image_to_yolo_supported(image_data):
    """
    Convert any image format to YOLO-supported format (JPG)
    
    Args:
        image_data (str or bytes): Image data (base64 data URL or bytes)
        
    Returns:
        str: Base64 data URL of converted JPG image
    """
    try:
        # Check if it's already a supported format
        if isinstance(image_data, str) and image_data.startswith('data:image/jpeg;base64,'):
            return image_data
        
        # Check if it's AVIF
        if is_avif_image(image_data):
            logger.info("Converting AVIF image to JPG")
            return convert_avif_to_jpg_base64(image_data)
        
        # For other formats, try to convert to JPG
        if isinstance(image_data, str) and image_data.startswith('data:'):
            # Extract base64 data
            base64_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
        else:
            image_bytes = image_data
        
        # Try to read with PIL
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.mode in ('RGBA', 'LA', 'P'):
                pil_img = pil_img.convert('RGB')
            
            # Convert to JPG
            jpg_buffer = io.BytesIO()
            pil_img.save(jpg_buffer, format='JPEG', quality=95, optimize=True)
            jpg_bytes = jpg_buffer.getvalue()
            
            # Convert to base64
            jpg_base64 = base64.b64encode(jpg_bytes).decode('utf-8')
            jpg_data_url = f"data:image/jpeg;base64,{jpg_base64}"
            
            logger.info(f"Successfully converted image to JPG. JPG size: {len(jpg_bytes)} bytes")
            return jpg_data_url
            
        except Exception as e:
            logger.error(f"Error converting image to JPG: {str(e)}")
            raise ValueError(f"Failed to convert image: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error converting image to YOLO-supported format: {str(e)}")
        raise ValueError(f"Failed to convert image: {str(e)}")

# Test function
def test_avif_conversion():
    """Test AVIF conversion functionality"""
    try:
        # Test with a sample AVIF file
        test_file = "/home/ubuntu/LTA_REFL/4.avif"
        if os.path.exists(test_file):
            with open(test_file, 'rb') as f:
                avif_bytes = f.read()
            
            # Test conversion
            jpg_data_url = convert_avif_to_jpg_base64(avif_bytes)
            print(f"✅ AVIF conversion successful! JPG data URL length: {len(jpg_data_url)}")
            return True
        else:
            print(f"❌ Test file not found: {test_file}")
            return False
            
    except Exception as e:
        print(f"❌ AVIF conversion test failed: {e}")
        return False

if __name__ == "__main__":
    test_avif_conversion()
