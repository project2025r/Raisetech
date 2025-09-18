import logging
import subprocess
import json
import os
from datetime import datetime
import tempfile
import base64

logger = logging.getLogger(__name__)

def extract_video_metadata(video_data):
    """
    Extract comprehensive metadata from video files including GPS, timestamp, and technical info.
    
    Args:
        video_data: Either a file path, file object, or base64 encoded video
        
    Returns:
        dict: Comprehensive metadata including GPS coordinates, timestamp, camera info, etc.
    """
    try:
        # Handle different input types
        temp_file_path = None
        
        if isinstance(video_data, str):
            if "base64," in video_data:
                # Handle base64 encoded video
                base64_data = video_data.split("base64,")[1]
                video_bytes = base64.b64decode(base64_data)
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    temp_file.write(video_bytes)
                    temp_file_path = temp_file.name
                    video_path = temp_file_path
            else:
                # Assume it's a file path
                video_path = video_data
        else:
            # Handle file objects or other types
            logger.warning("Unsupported video data type for metadata extraction")
            return _get_empty_metadata()
        
        # Use ffprobe to extract metadata
        metadata = _extract_with_ffprobe(video_path)
        
        # Clean up temporary file if created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting video metadata: {e}")
        return _get_empty_metadata()

def _extract_with_ffprobe(video_path):
    """Extract metadata using ffprobe."""
    try:
        # Run ffprobe to get comprehensive metadata
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.warning(f"ffprobe failed: {result.stderr}")
            return _get_empty_metadata()
        
        probe_data = json.loads(result.stdout)
        
        # Process the metadata
        metadata = {
            'basic_info': {},
            'video_streams': [],
            'audio_streams': [],
            'format_info': {},
            'gps_coordinates': None,
            'timestamp': None,
            'camera_info': {},
            'technical_info': {}
        }
        
        # Extract format information
        if 'format' in probe_data:
            format_info = probe_data['format']
            metadata['format_info'] = {
                'filename': format_info.get('filename', ''),
                'format_name': format_info.get('format_name', ''),
                'format_long_name': format_info.get('format_long_name', ''),
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'bit_rate': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None
            }
            
            # Extract creation time and other metadata from format tags
            if 'tags' in format_info:
                tags = format_info['tags']
                metadata['timestamp'] = _extract_timestamp_from_tags(tags)
                metadata['camera_info'] = _extract_camera_info_from_tags(tags)
                metadata['gps_coordinates'] = _extract_gps_from_tags(tags)
        
        # Extract stream information
        if 'streams' in probe_data:
            for stream in probe_data['streams']:
                if stream.get('codec_type') == 'video':
                    video_stream = {
                        'codec_name': stream.get('codec_name', ''),
                        'codec_long_name': stream.get('codec_long_name', ''),
                        'width': stream.get('width', 0),
                        'height': stream.get('height', 0),
                        'duration': float(stream.get('duration', 0)) if stream.get('duration') else None,
                        'bit_rate': int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None,
                        'frame_rate': stream.get('r_frame_rate', ''),
                        'pixel_format': stream.get('pix_fmt', '')
                    }
                    metadata['video_streams'].append(video_stream)
                    
                    # Update basic info with first video stream
                    if not metadata['basic_info']:
                        metadata['basic_info'] = {
                            'width': video_stream['width'],
                            'height': video_stream['height'],
                            'duration': video_stream['duration'],
                            'codec': video_stream['codec_name']
                        }
                
                elif stream.get('codec_type') == 'audio':
                    audio_stream = {
                        'codec_name': stream.get('codec_name', ''),
                        'codec_long_name': stream.get('codec_long_name', ''),
                        'sample_rate': int(stream.get('sample_rate', 0)) if stream.get('sample_rate') else None,
                        'channels': stream.get('channels', 0),
                        'bit_rate': int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None
                    }
                    metadata['audio_streams'].append(audio_stream)
        
        return metadata
        
    except subprocess.TimeoutExpired:
        logger.error("ffprobe timed out")
        return _get_empty_metadata()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output: {e}")
        return _get_empty_metadata()
    except Exception as e:
        logger.error(f"Error running ffprobe: {e}")
        return _get_empty_metadata()

def _extract_timestamp_from_tags(tags):
    """Extract timestamp from video metadata tags."""
    timestamp_fields = ['creation_time', 'date', 'DATE', 'CREATION_TIME']
    
    for field in timestamp_fields:
        if field in tags:
            try:
                # Try different timestamp formats
                timestamp_str = tags[field]
                
                # ISO format with timezone
                if 'T' in timestamp_str and 'Z' in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    return timestamp.isoformat()
                
                # Other common formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y:%m:%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        return timestamp.isoformat()
                    except ValueError:
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to parse timestamp {tags[field]}: {e}")
                continue
    
    return None

def _extract_camera_info_from_tags(tags):
    """Extract camera information from video metadata tags."""
    camera_info = {}
    
    camera_fields = {
        'make': 'camera_make',
        'MAKE': 'camera_make',
        'model': 'camera_model',
        'MODEL': 'camera_model',
        'software': 'software',
        'SOFTWARE': 'software',
        'encoder': 'encoder',
        'ENCODER': 'encoder'
    }
    
    for tag_field, meta_field in camera_fields.items():
        if tag_field in tags:
            camera_info[meta_field] = str(tags[tag_field])
    
    return camera_info

def _extract_gps_from_tags(tags):
    """Extract GPS coordinates from video metadata tags."""
    # Look for GPS information in various tag formats
    gps_fields = ['location', 'LOCATION', 'gps', 'GPS', 'com.apple.quicktime.location.ISO6709']
    
    for field in gps_fields:
        if field in tags:
            try:
                location_str = tags[field]
                
                # Parse ISO 6709 format: +37.5090-122.2594+000.000/
                if location_str.startswith('+') or location_str.startswith('-'):
                    # Extract latitude and longitude
                    import re
                    match = re.match(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', location_str)
                    if match:
                        lat = float(match.group(1))
                        lon = float(match.group(2))
                        return {
                            'latitude': lat,
                            'longitude': lon,
                            'coordinates_string': f"{lat},{lon}"
                        }
                
                # Try other formats as needed
                logger.info(f"Found GPS field {field} with value {location_str}, but couldn't parse")
                
            except Exception as e:
                logger.warning(f"Failed to parse GPS data from {field}: {e}")
                continue
    
    return None

def _get_empty_metadata():
    """Return empty metadata structure."""
    return {
        'basic_info': {},
        'video_streams': [],
        'audio_streams': [],
        'format_info': {},
        'gps_coordinates': None,
        'timestamp': None,
        'camera_info': {},
        'technical_info': {}
    }

def get_video_gps_coordinates(video_data):
    """Extract GPS coordinates from video metadata."""
    metadata = extract_video_metadata(video_data)
    gps_coords = metadata.get('gps_coordinates')
    
    if gps_coords:
        return gps_coords['latitude'], gps_coords['longitude']
    
    return None, None
