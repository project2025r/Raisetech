#!/usr/bin/env python3
"""
Simple AVIF to JPG conversion test for EC2
This script will try to convert your 4.avif file to JPG
"""

import os
import sys

def simple_avif_test():
    """Simple test to convert AVIF to JPG"""
    
    avif_path = "/home/ubuntu/LTA_REFL/4.avif"
    output_dir = "/home/ubuntu/LTA_REFL"
    
    print(f"🔍 Looking for AVIF file: {avif_path}")
    
    if not os.path.exists(avif_path):
        print(f"❌ File not found: {avif_path}")
        return False
    
    print(f"✅ Found AVIF file")
    file_size = os.path.getsize(avif_path)
    print(f"   Size: {file_size} bytes")
    
    # Try PIL first (most reliable)
    print("\n🔄 Trying PIL method...")
    try:
        from PIL import Image
        import io
        
        # Open the AVIF file
        with open(avif_path, 'rb') as f:
            avif_data = f.read()
        
        # Try to open with PIL
        img = Image.open(io.BytesIO(avif_data))
        print(f"✅ PIL successfully opened AVIF")
        print(f"   Mode: {img.mode}")
        print(f"   Size: {img.size}")
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
            print(f"   Converted to RGB")
        
        # Save as JPG
        jpg_path = os.path.join(output_dir, "4_converted_pil.jpg")
        img.save(jpg_path, 'JPEG', quality=95)
        
        jpg_size = os.path.getsize(jpg_path)
        print(f"✅ Successfully saved as JPG")
        print(f"   Path: {jpg_path}")
        print(f"   Size: {jpg_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ PIL method failed: {e}")
    
    # Try imageio
    print("\n🔄 Trying imageio method...")
    try:
        import imageio
        import numpy as np
        from PIL import Image
        
        # Read with imageio
        img_array = imageio.imread(avif_path)
        print(f"✅ imageio successfully read AVIF")
        print(f"   Shape: {img_array.shape}")
        print(f"   Dtype: {img_array.dtype}")
        
        # Convert to PIL
        img = Image.fromarray(img_array)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Save as JPG
        jpg_path = os.path.join(output_dir, "4_converted_imageio.jpg")
        img.save(jpg_path, 'JPEG', quality=95)
        
        jpg_size = os.path.getsize(jpg_path)
        print(f"✅ Successfully saved as JPG")
        print(f"   Path: {jpg_path}")
        print(f"   Size: {jpg_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ imageio method failed: {e}")
    
    # Try ffmpeg
    print("\n🔄 Trying ffmpeg method...")
    try:
        import subprocess
        
        jpg_path = os.path.join(output_dir, "4_converted_ffmpeg.jpg")
        
        # Use ffmpeg to convert
        cmd = ['ffmpeg', '-i', avif_path, '-y', jpg_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            jpg_size = os.path.getsize(jpg_path)
            print(f"✅ ffmpeg successfully converted AVIF to JPG")
            print(f"   Path: {jpg_path}")
            print(f"   Size: {jpg_size} bytes")
            return True
        else:
            print(f"❌ ffmpeg failed")
            print(f"   Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ ffmpeg method failed: {e}")
    
    return False

def install_dependencies():
    """Install missing dependencies"""
    print("\n🔧 Installing missing dependencies...")
    
    try:
        import subprocess
        
        print("Updating package list...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        
        print("Installing ffmpeg...")
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True)
        
        print("Installing libavif...")
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libavif-dev', 'libavif-bin'], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Simple AVIF to JPG Test")
    print("=" * 40)
    
    # Try conversion
    success = simple_avif_test()
    
    if not success:
        print("\n❌ All methods failed!")
        print("\n🔧 Installing missing dependencies...")
        
        if install_dependencies():
            print("\n🔄 Retrying conversion...")
            success = simple_avif_test()
            
            if success:
                print("\n🎉 Success! AVIF conversion is now working.")
            else:
                print("\n❌ Still failing after dependency installation.")
        else:
            print("\n❌ Failed to install dependencies.")
    
    if success:
        print("\n📁 Check the converted JPG files:")
        print("   - 4_converted_pil.jpg")
        print("   - 4_converted_imageio.jpg")
        print("   - 4_converted_ffmpeg.jpg")
    
    print("\n💡 If still failing, try:")
    print("   sudo apt-get update")
    print("   sudo apt-get install -y ffmpeg libavif-dev libavif-bin")
    print("   pip install --upgrade pillow-avif-plugin")

if __name__ == "__main__":
    main()
