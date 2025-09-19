import PIL.Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import io
import base64
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_exif_data(image_data):
    """Extract EXIF data from an image.
    
    Args:
        image_data: Either a file path, file object, or base64 encoded image
    """
    exif_data = {}
    try:
        # Handle base64 encoded images
        if isinstance(image_data, str) and "base64," in image_data:
            # Extract the base64 part
            base64_data = image_data.split("base64,")[1]
            # Convert to image
            img_data = base64.b64decode(base64_data)
            img = PIL.Image.open(io.BytesIO(img_data))
        else:
            # Handle regular file paths or file objects
            img = PIL.Image.open(image_data)
        
        # Get EXIF data
        info = img._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for gps_tag in value:
                        sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
                        gps_data[sub_decoded] = value[gps_tag]
                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
    return exif_data

def convert_to_degrees(value):
    """Convert GPS coordinates from DMS (degree, minutes, seconds) to DD (decimal degrees)."""
    if not value:
        return None

    try:
        def safe_float_conversion(val):
            """Safely convert a value to float, handling rational numbers and zero denominators."""
            if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                # Handle rational numbers (fractions)
                if val.denominator == 0:
                    logger.warning(f"GPS coordinate has zero denominator: {val}")
                    return 0.0
                return float(val.numerator) / float(val.denominator)
            else:
                return float(val)

        d = safe_float_conversion(value[0])
        m = safe_float_conversion(value[1])
        s = safe_float_conversion(value[2])

        return d + (m / 60.0) + (s / 3600.0)
    except (ValueError, TypeError, ZeroDivisionError, IndexError) as e:
        logger.warning(f"Error converting GPS coordinates to degrees: {e}")
        return None

def get_gps_coordinates(image_data):
    """Extract GPS coordinates from image EXIF data and return them as decimal degrees."""
    try:
        exif_data = get_exif_data(image_data)
        if not exif_data or "GPSInfo" not in exif_data:
            logger.debug("No EXIF data or GPS info found in image")
            return None, None

        gps_info = exif_data["GPSInfo"]
        logger.debug(f"GPS info found: {list(gps_info.keys())}")

        lat = None
        lon = None

        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            logger.debug(f"GPS Latitude data: {gps_info['GPSLatitude']}")
            lat = convert_to_degrees(gps_info["GPSLatitude"])
            if lat is not None and gps_info["GPSLatitudeRef"] == "S":
                lat = -lat
            logger.debug(f"Converted latitude: {lat}")

        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            logger.debug(f"GPS Longitude data: {gps_info['GPSLongitude']}")
            lon = convert_to_degrees(gps_info["GPSLongitude"])
            if lon is not None and gps_info["GPSLongitudeRef"] == "W":
                lon = -lon
            logger.debug(f"Converted longitude: {lon}")

        return lat, lon

    except Exception as e:
        logger.error(f"Error extracting GPS coordinates: {e}")
        return None, None

def get_comprehensive_exif_data(image_data):
    """Extract comprehensive EXIF data including GPS, camera info, and timestamp."""
    try:
        # Handle base64 encoded images
        if isinstance(image_data, str) and "base64," in image_data:
            # Extract the base64 part
            base64_data = image_data.split("base64,")[1]
            # Convert to image
            img_data = base64.b64decode(base64_data)
            img = PIL.Image.open(io.BytesIO(img_data))
        else:
            # Handle regular file paths or file objects
            img = PIL.Image.open(image_data)

        # Get basic image info
        basic_info = {
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode
        }

        # Get EXIF data
        exif_data = {}
        try:
            exif_dict = img._getexif()
            if exif_dict:
                for tag, value in exif_dict.items():
                    decoded = TAGS.get(tag, tag)
                    if decoded == "GPSInfo":
                        gps_data = {}
                        for gps_tag in value:
                            sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
                            gps_data[sub_decoded] = value[gps_tag]
                        exif_data[decoded] = gps_data
                    else:
                        exif_data[decoded] = value
        except Exception as e:
            logger.warning(f"Error extracting EXIF data: {e}")

        # Extract specific metadata
        metadata = {
            'basic_info': basic_info,
            'exif_data': exif_data,
            'gps_coordinates': None,
            'timestamp': None,
            'camera_info': {},
            'technical_info': {}
        }

        # Extract GPS coordinates
        if 'GPSInfo' in exif_data:
            lat, lon = get_gps_coordinates(image_data)
            if lat is not None and lon is not None:
                metadata['gps_coordinates'] = {
                    'latitude': lat,
                    'longitude': lon,
                    'coordinates_string': f"{lat},{lon}"
                }

        # Extract timestamp
        timestamp_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
        for field in timestamp_fields:
            if field in exif_data:
                try:
                    timestamp = datetime.strptime(exif_data[field], '%Y:%m:%d %H:%M:%S')
                    metadata['timestamp'] = timestamp.isoformat()
                    break
                except ValueError:
                    continue

        # Extract camera information
        camera_fields = {
            'Make': 'camera_make',
            'Model': 'camera_model',
            'Software': 'software',
            'LensModel': 'lens_model',
            'LensMake': 'lens_make'
        }

        for exif_field, meta_field in camera_fields.items():
            if exif_field in exif_data:
                metadata['camera_info'][meta_field] = str(exif_data[exif_field])

        # Extract technical information
        technical_fields = {
            'ExposureTime': 'exposure_time',
            'FNumber': 'f_number',
            'ISO': 'iso',
            'ISOSpeedRatings': 'iso_speed',
            'FocalLength': 'focal_length',
            'Flash': 'flash',
            'WhiteBalance': 'white_balance',
            'ExposureMode': 'exposure_mode',
            'SceneCaptureType': 'scene_type'
        }

        for exif_field, meta_field in technical_fields.items():
            if exif_field in exif_data:
                metadata['technical_info'][meta_field] = str(exif_data[exif_field])

        return metadata

    except Exception as e:
        logger.error(f"Error extracting comprehensive EXIF data: {e}")
        return {
            'basic_info': {},
            'exif_data': {},
            'gps_coordinates': None,
            'timestamp': None,
            'camera_info': {},
            'technical_info': {}
        }

def format_coordinates(lat, lon):
    """Format coordinates as a string."""
    if lat is None or lon is None:
        return "Not Available"
    
    return f"{lat:.6f}, {lon:.6f}" 