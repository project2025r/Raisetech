"""
Real Coordinate Extraction Utilities

This module provides comprehensive coordinate extraction functionality for the 
Road AI Safety Enhancement system, consolidating GPS coordinate extraction 
from various media types and formats.

Author: Road AI Safety Enhancement System
Date: 2025-09-19
"""

import logging
import re
from typing import Tuple, Optional, Dict, Any, Union
import base64
import io
from datetime import datetime

# Import existing utilities
try:
    from utils.exif_utils import get_gps_coordinates, format_coordinates
    from utils.video_metadata_utils import get_video_gps_coordinates, extract_video_metadata
except ModuleNotFoundError as e:
    import sys
    print(f"Error: {e}. Please ensure 'utils/exif_utils.py' and 'utils/video_metadata_utils.py' exist and are in the correct path.")
    sys.exit(1)

logger = logging.getLogger(__name__)


def extract_coordinates_from_media(media_data: Union[str, bytes], media_type: str = 'image') -> Tuple[Optional[float], Optional[float]]:
    """
    Extract GPS coordinates from media data (image or video).
    
    Args:
        media_data: Base64 encoded media data or file path
        media_type: Type of media ('image' or 'video')
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if not found
    """
    try:
        if media_type.lower() == 'video':
            return get_video_gps_coordinates(media_data)
        else:
            return get_gps_coordinates(media_data)
    except Exception as e:
        logger.error(f"Error extracting coordinates from {media_type}: {e}")
        return None, None


def extract_coordinates_from_string(coord_string: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract coordinates from various string formats.
    
    Supported formats:
    - "lat,lon" (e.g., "18.520601,73.849890")
    - "lat, lon" (with spaces)
    - ISO 6709 format (e.g., "+18.520601-073.849890/")
    - DMS format parsing
    
    Args:
        coord_string: String containing coordinates
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if parsing fails
    """
    if not coord_string or not isinstance(coord_string, str):
        return None, None
    
    try:
        # Clean the string
        coord_string = coord_string.strip()
        
        # Format 1: Simple "lat,lon" format
        if ',' in coord_string and not coord_string.startswith(('+', '-')):
            parts = coord_string.split(',')
            if len(parts) == 2:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return lat, lon
        
        # Format 2: ISO 6709 format (+lat-lon or +lat+lon)
        if coord_string.startswith(('+', '-')):
            # Match pattern like +18.520601-073.849890/ or +18.520601+073.849890/
            match = re.match(r'([+-]\d+\.?\d*)([+-]\d+\.?\d*)', coord_string)
            if match:
                lat = float(match.group(1))
                lon = float(match.group(2))
                return lat, lon
        
        # Format 3: Space-separated coordinates
        if ' ' in coord_string:
            parts = coord_string.split()
            if len(parts) >= 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    return lat, lon
                except ValueError:
                    pass
        
        logger.warning(f"Could not parse coordinate string: {coord_string}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error parsing coordinate string '{coord_string}': {e}")
        return None, None


def validate_coordinates(lat: Optional[float], lon: Optional[float]) -> bool:
    """
    Validate that coordinates are within valid ranges.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    if lat is None or lon is None:
        return False
    
    # Check latitude range (-90 to 90)
    if not (-90 <= lat <= 90):
        return False
    
    # Check longitude range (-180 to 180)
    if not (-180 <= lon <= 180):
        return False
    
    return True


def format_coordinates_comprehensive(lat: Optional[float], lon: Optional[float]) -> Dict[str, Any]:
    """
    Format coordinates in multiple formats for comprehensive display.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        
    Returns:
        Dictionary containing various coordinate formats
    """
    if not validate_coordinates(lat, lon):
        return {
            'decimal': 'Not Available',
            'string': 'Not Available',
            'dms': 'Not Available',
            'valid': False,
            'latitude': None,
            'longitude': None
        }
    
    try:
        # Decimal format
        decimal_format = f"{lat:.6f}, {lon:.6f}"
        
        # String format (for database storage)
        string_format = f"{lat:.6f},{lon:.6f}"
        
        # DMS (Degrees, Minutes, Seconds) format
        def decimal_to_dms(decimal_deg: float, is_latitude: bool = True) -> str:
            """Convert decimal degrees to DMS format."""
            abs_deg = abs(decimal_deg)
            degrees = int(abs_deg)
            minutes_float = (abs_deg - degrees) * 60
            minutes = int(minutes_float)
            seconds = (minutes_float - minutes) * 60
            
            if is_latitude:
                direction = 'N' if decimal_deg >= 0 else 'S'
            else:
                direction = 'E' if decimal_deg >= 0 else 'W'
            
            return f"{degrees}°{minutes}'{seconds:.2f}\"{direction}"
        
        lat_dms = decimal_to_dms(lat, True)
        lon_dms = decimal_to_dms(lon, False)
        dms_format = f"{lat_dms}, {lon_dms}"
        
        return {
            'decimal': decimal_format,
            'string': string_format,
            'dms': dms_format,
            'valid': True,
            'latitude': lat,
            'longitude': lon
        }
        
    except Exception as e:
        logger.error(f"Error formatting coordinates {lat}, {lon}: {e}")
        return {
            'decimal': 'Error',
            'string': 'Error',
            'dms': 'Error',
            'valid': False,
            'latitude': lat,
            'longitude': lon
        }


def extract_real_coordinates(data: Union[str, bytes, Dict[str, Any]], 
                           data_type: str = 'auto',
                           fallback_coords: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to extract real coordinates from various data sources.
    
    Args:
        data: Input data (media data, coordinate string, or metadata dict)
        data_type: Type of data ('image', 'video', 'string', 'metadata', 'auto')
        fallback_coords: Fallback coordinate string if extraction fails
        
    Returns:
        Dictionary containing extracted and formatted coordinates
    """
    lat, lon = None, None
    extraction_source = 'none'
    
    try:
        # Auto-detect data type if not specified
        if data_type == 'auto':
            if isinstance(data, dict):
                data_type = 'metadata'
            elif isinstance(data, str):
                if data.startswith('data:') or 'base64' in data:
                    data_type = 'image'  # Assume image for base64 data
                else:
                    data_type = 'string'
            else:
                data_type = 'image'  # Default fallback
        
        # Extract coordinates based on data type
        if data_type == 'metadata' and isinstance(data, dict):
            # Extract from metadata dictionary
            if 'gps_coordinates' in data and data['gps_coordinates']:
                gps_data = data['gps_coordinates']
                if isinstance(gps_data, dict):
                    lat = gps_data.get('latitude')
                    lon = gps_data.get('longitude')
                    extraction_source = 'metadata_gps'
            elif 'coordinates' in data:
                coord_str = data['coordinates']
                lat, lon = extract_coordinates_from_string(coord_str)
                extraction_source = 'metadata_coordinates'
                
        elif data_type == 'string':
            # Extract from coordinate string
            lat, lon = extract_coordinates_from_string(data)
            extraction_source = 'string_parsing'
            
        elif data_type in ['image', 'video']:
            # Extract from media data
            lat, lon = extract_coordinates_from_media(data, data_type)
            extraction_source = f'{data_type}_exif'
        
        # Use fallback coordinates if extraction failed
        if not validate_coordinates(lat, lon) and fallback_coords:
            lat, lon = extract_coordinates_from_string(fallback_coords)
            extraction_source = 'fallback'
        
        # Format the results
        formatted_coords = format_coordinates_comprehensive(lat, lon)
        formatted_coords['extraction_source'] = extraction_source
        formatted_coords['timestamp'] = datetime.now().isoformat()
        
        return formatted_coords
        
    except Exception as e:
        logger.error(f"Error in extract_real_coordinates: {e}")
        return {
            'decimal': 'Error',
            'string': 'Error',
            'dms': 'Error',
            'valid': False,
            'latitude': None,
            'longitude': None,
            'extraction_source': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }


# Convenience functions for backward compatibility
def get_coordinates_from_image(image_data: Union[str, bytes]) -> Tuple[Optional[float], Optional[float]]:
    """Extract coordinates from image data."""
    return extract_coordinates_from_media(image_data, 'image')


def get_coordinates_from_video(video_data: Union[str, bytes]) -> Tuple[Optional[float], Optional[float]]:
    """Extract coordinates from video data."""
    return extract_coordinates_from_media(video_data, 'video')


def parse_coordinate_string(coord_string: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse coordinates from string format."""
    return extract_coordinates_from_string(coord_string)


if __name__ == "__main__":
    # Test the coordinate extraction functionality
    test_coords = [
        "18.520601,73.849890",
        "+18.520601-073.849890/",
        "18.520601 73.849890",
        "invalid coordinates",
        None
    ]
    
    print("Testing coordinate extraction:")
    for coord in test_coords:
        result = extract_real_coordinates(coord, 'string')
        print(f"Input: {coord}")
        print(f"Result: {result}")
        print("-" * 50)
