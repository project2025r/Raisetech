import PIL.Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import io
import base64

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
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except (ValueError, TypeError):
        return None

def get_gps_coordinates(image_data):
    """Extract GPS coordinates from image EXIF data and return them as decimal degrees."""
    exif_data = get_exif_data(image_data)
    if not exif_data or "GPSInfo" not in exif_data:
        return None, None
    
    gps_info = exif_data["GPSInfo"]
    
    lat = None
    lon = None
    
    if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
        lat = convert_to_degrees(gps_info["GPSLatitude"])
        if gps_info["GPSLatitudeRef"] == "S":
            lat = -lat if lat is not None else None
    
    if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info["GPSLongitudeRef"] == "W":
            lon = -lon if lon is not None else None
    
    return lat, lon

def format_coordinates(lat, lon):
    """Format coordinates as a string."""
    if lat is None or lon is None:
        return "Not Available"
    
    return f"{lat:.6f}, {lon:.6f}" 