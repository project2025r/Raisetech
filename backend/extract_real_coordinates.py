#!/usr/bin/env python3
"""
Extract real GPS coordinates from EXIF data of images
"""

import sys
import os
from datetime import datetime
import base64
import io

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def extract_real_gps_from_images():
    """Extract real GPS coordinates from image EXIF data"""
    try:
        from config.db import connect_to_db
        from utils.exif_utils import get_gps_coordinates, get_comprehensive_exif_data
        from utils.s3_utils import download_from_s3
        
        print("🔄 Connecting to database...")
        db = connect_to_db()
        
        if db is None:
            print("❌ Database connection failed")
            return False
        
        print("✅ Database connection successful")
        
        collections = [
            ("pothole_images", "pothole"),
            ("crack_images", "crack"),
            ("kerb_images", "kerb")
        ]
        
        total_updated = 0
        india_coordinates = []
        
        for collection_name, defect_type in collections:
            collection = getattr(db, collection_name)
            
            # Find images with S3 URLs to extract real EXIF data
            images = list(collection.find({
                "original_image_s3_url": {"$exists": True, "$ne": None}
            }).limit(20))  # Limit to 20 for testing
            
            print(f"📊 Found {len(images)} {defect_type} images with S3 URLs")
            
            updated_count = 0
            for image in images:
                try:
                    s3_url = image.get("original_image_s3_url")
                    if not s3_url:
                        continue
                    
                    print(f"🔄 Processing {defect_type} image: {image.get('image_id', 'unknown')}")
                    
                    # Try to download and extract EXIF data
                    try:
                        # Download image from S3
                        image_data = download_from_s3(s3_url)
                        if not image_data:
                            print(f"⚠️  Could not download image from S3: {s3_url}")
                            continue
                        
                        # Extract GPS coordinates
                        lat, lng = get_gps_coordinates(image_data)
                        
                        if lat is not None and lng is not None:
                            # Check if coordinates are in India (approximate bounds)
                            if 6.0 <= lat <= 37.0 and 68.0 <= lng <= 97.0:
                                coordinates_str = f"{lat},{lng}"
                                india_coordinates.append((lat, lng, defect_type))
                                
                                print(f"✅ Found India coordinates: {coordinates_str}")
                                
                                # Extract comprehensive EXIF data
                                comprehensive_exif = get_comprehensive_exif_data(image_data)
                                
                                # Update the database
                                result = collection.update_one(
                                    {"_id": image["_id"]},
                                    {
                                        "$set": {
                                            "coordinates": coordinates_str,
                                            "exif_data": comprehensive_exif.get('exif_data', {}),
                                            "metadata": comprehensive_exif,
                                            "gps_source": "real_exif",
                                            "country": "India"
                                        }
                                    }
                                )
                                
                                if result.modified_count > 0:
                                    updated_count += 1
                                    print(f"✅ Updated {defect_type} with real coordinates: {coordinates_str}")
                            else:
                                print(f"⚠️  Coordinates not in India: {lat}, {lng}")
                        else:
                            print(f"⚠️  No GPS coordinates found in EXIF data")
                            
                    except Exception as e:
                        print(f"❌ Error processing image {image.get('image_id')}: {e}")
                        continue
                        
                except Exception as e:
                    print(f"❌ Error processing {defect_type} image: {e}")
                    continue
            
            print(f"✅ Updated {updated_count} {defect_type} images with real coordinates")
            total_updated += updated_count
        
        print(f"\n📊 SUMMARY:")
        print(f"✅ Total images updated with real coordinates: {total_updated}")
        print(f"✅ India coordinates found: {len(india_coordinates)}")
        
        if india_coordinates:
            print(f"\n📍 Sample India coordinates:")
            for i, (lat, lng, defect_type) in enumerate(india_coordinates[:5]):
                print(f"   {i+1}. {defect_type}: {lat:.6f}, {lng:.6f}")
            
            # Calculate center point for India
            avg_lat = sum(coord[0] for coord in india_coordinates) / len(india_coordinates)
            avg_lng = sum(coord[1] for coord in india_coordinates) / len(india_coordinates)
            print(f"\n🎯 Suggested map center for India: [{avg_lat:.6f}, {avg_lng:.6f}]")
        
        return total_updated > 0
        
    except Exception as e:
        print(f"❌ Error extracting real coordinates: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_map_center_for_india():
    """Update the frontend map center to focus on India"""
    try:
        # Read the DefectMap.js file
        defect_map_path = "../frontend/src/components/DefectMap.js"
        
        if not os.path.exists(defect_map_path):
            print(f"❌ DefectMap.js not found at {defect_map_path}")
            return False
        
        with open(defect_map_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Singapore coordinates with India coordinates
        # India center: approximately [20.5937, 78.9629]
        old_center = "const [center] = useState([1.3521, 103.8198]); // Singapore center"
        new_center = "const [center] = useState([20.5937, 78.9629]); // India center"
        
        if old_center in content:
            content = content.replace(old_center, new_center)
            
            with open(defect_map_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Updated DefectMap.js to center on India")
            return True
        else:
            print("⚠️  Could not find Singapore center coordinates to replace")
            return False
            
    except Exception as e:
        print(f"❌ Error updating map center: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting real GPS coordinate extraction...\n")
    
    # Extract real coordinates from EXIF data
    success = extract_real_gps_from_images()
    
    if success:
        # Update map center to India
        map_updated = update_map_center_for_india()
        
        print(f"\n{'='*60}")
        print("✅ REAL COORDINATE EXTRACTION COMPLETED!")
        print("✅ Images now have actual GPS coordinates from EXIF data")
        if map_updated:
            print("✅ Map center updated to focus on India")
        print("✅ The DefectMap should now show markers at real locations in India")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ COORDINATE EXTRACTION FAILED!")
        print("❌ Could not extract real GPS coordinates from images")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
