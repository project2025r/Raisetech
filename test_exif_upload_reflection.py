#!/usr/bin/env python3
"""
Test script to verify EXIF data upload and reflection in map view
This script tests the complete workflow from upload to map display
"""

import requests
import json
import base64
import time
from datetime import datetime
import os

# Configuration
BASE_URL = "http://localhost:5000"  # Adjust as needed
TEST_IMAGE_PATH = "test_image_with_exif.jpg"  # You'll need to provide this

def create_test_image_with_exif():
    """Create a test image with EXIF GPS data"""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        import io

        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')

        # Add EXIF GPS data (this is simplified - real EXIF GPS is more complex)
        # For testing, we'll create a basic image and add coordinates manually in the upload

        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        return img_bytes.getvalue()
    except ImportError:
        print("⚠️  PIL not available, using placeholder image data")
        # Return a minimal JPEG header for testing
        return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00d\x00d\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'

def encode_image_to_base64(image_data):
    """Encode image data to base64 string"""
    return f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

def test_pothole_upload():
    """Test pothole detection upload with EXIF data"""
    print("🔄 Testing pothole upload with EXIF data...")
    
    # Create test image
    image_data = create_test_image_with_exif()
    base64_image = encode_image_to_base64(image_data)
    
    # Upload data
    upload_data = {
        "image": base64_image,
        "username": "test_user",
        "role": "audit",
        "coordinates": "12.9716,77.5946",  # Bangalore coordinates
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/pavement/detect-potholes", 
                               json=upload_data, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✅ Upload successful! Image ID: {result.get('image_id', 'N/A')}")
                return result.get('image_id')
            else:
                print(f"❌ Upload failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
    
    return None

def test_map_data_retrieval():
    """Test retrieving data from the map API"""
    print("🔄 Testing map data retrieval...")
    
    try:
        # Add cache-busting parameter
        params = {"_t": int(time.time() * 1000)}
        response = requests.get(f"{BASE_URL}/api/dashboard/image-stats", 
                              params=params, 
                              timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                images = result.get("images", [])
                print(f"✅ Retrieved {len(images)} images from map API")
                
                # Check for recent uploads (last 5 minutes)
                recent_threshold = datetime.now().timestamp() - 300  # 5 minutes ago
                recent_images = []
                
                for img in images:
                    try:
                        img_timestamp = datetime.fromisoformat(img.get("timestamp", "")).timestamp()
                        if img_timestamp > recent_threshold:
                            recent_images.append(img)
                    except:
                        continue
                
                print(f"📊 Found {len(recent_images)} recent images (last 5 minutes)")
                
                # Check EXIF data presence
                exif_count = sum(1 for img in images if img.get("exif_data"))
                print(f"📊 Images with EXIF data: {exif_count}/{len(images)}")
                
                # Show sample of latest images
                if images:
                    print("\n📋 Latest 3 images:")
                    for i, img in enumerate(images[:3]):
                        print(f"  {i+1}. ID: {img.get('image_id', 'N/A')}")
                        print(f"     Timestamp: {img.get('timestamp', 'N/A')}")
                        print(f"     Coordinates: {img.get('coordinates', 'N/A')}")
                        print(f"     EXIF GPS: {img.get('exif_data', {}).get('gps_coordinates', 'N/A')}")
                        print(f"     Media Type: {img.get('media_type', 'N/A')}")
                        print()
                
                return images
            else:
                print(f"❌ API returned error: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
    
    return []

def test_database_direct_query():
    """Test direct database query to verify data storage"""
    print("🔄 Testing direct database query...")
    
    try:
        import sys
        sys.path.append('backend')
        from config.db import connect_to_db
        db = connect_to_db()
        
        if db is None:
            print("❌ Could not connect to database")
            return
        
        # Check latest pothole images
        latest_potholes = list(db.pothole_images.find({}, {
            'image_id': 1,
            'timestamp': 1,
            'coordinates': 1,
            'exif_data': 1,
            'metadata': 1,
            'media_type': 1
        }).sort("timestamp", -1).limit(5))
        
        print(f"✅ Found {len(latest_potholes)} pothole images in database")
        
        for i, img in enumerate(latest_potholes):
            print(f"  {i+1}. ID: {img.get('image_id', 'N/A')}")
            print(f"     Timestamp: {img.get('timestamp', 'N/A')}")
            print(f"     Has EXIF: {bool(img.get('exif_data'))}")
            print(f"     Media Type: {img.get('media_type', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"❌ Database query error: {e}")

def main():
    """Main test function"""
    print("🚀 Starting EXIF Upload and Map Reflection Test")
    print("=" * 50)
    
    # Test 1: Upload image with EXIF data
    image_id = test_pothole_upload()
    
    if image_id:
        print(f"\n⏳ Waiting 5 seconds for processing...")
        time.sleep(5)
        
        # Test 2: Check if it appears in map API
        images = test_map_data_retrieval()
        
        # Test 3: Check database directly
        test_database_direct_query()
        
        # Test 4: Check if our uploaded image appears in the results
        if images:
            uploaded_image = next((img for img in images if img.get('image_id') == image_id), None)
            if uploaded_image:
                print(f"✅ SUCCESS: Uploaded image found in map API!")
                print(f"   Image ID: {uploaded_image.get('image_id')}")
                print(f"   EXIF Data: {bool(uploaded_image.get('exif_data'))}")
                print(f"   GPS Coordinates: {uploaded_image.get('exif_data', {}).get('gps_coordinates', 'N/A')}")
            else:
                print(f"❌ ISSUE: Uploaded image (ID: {image_id}) NOT found in map API")
        
    print("\n" + "=" * 50)
    print("🏁 Test completed")

if __name__ == "__main__":
    main()
