#!/usr/bin/env python3
"""
Comprehensive AVIF to JPG conversion test for EC2
This script will test multiple methods and provide solutions
"""

import os
import sys
import subprocess
import tempfile
import base64
import io

def check_system_dependencies():
    """Check what system dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ ffmpeg is available")
            ffmpeg_version = result.stdout.split('\n')[0]
            print(f"   Version: {ffmpeg_version}")
        else:
            print("❌ ffmpeg is not working properly")
    except FileNotFoundError:
        print("❌ ffmpeg is not installed")
    except Exception as e:
        print(f"❌ ffmpeg test failed: {e}")
    
    # Check libavif
    try:
        result = subprocess.run(['avifdec', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ libavif is available")
            print(f"   Version: {result.stdout.strip()}")
        else:
            print("❌ libavif is not working properly")
    except FileNotFoundError:
        print("❌ libavif is not installed")
    except Exception as e:
        print(f"❌ libavif test failed: {e}")
    
    # Check Python packages
    print("\n🔍 Checking Python packages...")
    
    packages = [
        'imageio', 'PIL', 'numpy', 'pillow_avif_plugin'
    ]
    
    for package in packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✅ {package} imported successfully: {PIL.__version__}")
            elif package == 'pillow_avif_plugin':
                import pillow_avif_plugin
                print(f"✅ {package} imported successfully")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package} imported successfully: {version}")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")

def test_avif_conversion_methods():
    """Test different AVIF conversion methods"""
    print("\n🔍 Testing AVIF conversion methods...")
    
    avif_path = "/home/ubuntu/LTA_REFL/4.avif"
    
    if not os.path.exists(avif_path):
        print(f"❌ AVIF file not found: {avif_path}")
        return False
    
    print(f"✅ Found AVIF file: {avif_path}")
    file_size = os.path.getsize(avif_path)
    print(f"   File size: {file_size} bytes")
    
    # Read the AVIF file
    with open(avif_path, 'rb') as f:
        avif_bytes = f.read()
    
    print(f"✅ Read {len(avif_bytes)} bytes from AVIF file")
    
    # Method 1: Try PIL with AVIF plugin
    print("\n🔄 Method 1: PIL with AVIF plugin")
    try:
        from PIL import Image
        pil_img = Image.open(io.BytesIO(avif_bytes))
        print(f"✅ PIL successfully opened AVIF image")
        print(f"   Mode: {pil_img.mode}")
        print(f"   Size: {pil_img.size}")
        
        # Convert to RGB if necessary
        if pil_img.mode in ('RGBA', 'LA', 'P'):
            pil_img = pil_img.convert('RGB')
            print(f"   Converted to RGB mode")
        
        # Save as JPG
        jpg_path = "/home/ubuntu/LTA_REFL/converted_pil.jpg"
        pil_img.save(jpg_path, 'JPEG', quality=95)
        print(f"✅ Successfully saved as JPG: {jpg_path}")
        print(f"   JPG size: {os.path.getsize(jpg_path)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ PIL method failed: {e}")
    
    # Method 2: Try imageio
    print("\n🔄 Method 2: imageio")
    try:
        import imageio
        img_array = imageio.imread(avif_path)
        print(f"✅ imageio successfully read AVIF image")
        print(f"   Shape: {img_array.shape}")
        print(f"   Dtype: {img_array.dtype}")
        
        # Convert to PIL and save
        from PIL import Image
        import numpy as np
        
        pil_img = Image.fromarray(img_array)
        if pil_img.mode in ('RGBA', 'LA', 'P'):
            pil_img = pil_img.convert('RGB')
        
        jpg_path = "/home/ubuntu/LTA_REFL/converted_imageio.jpg"
        pil_img.save(jpg_path, 'JPEG', quality=95)
        print(f"✅ Successfully saved as JPG: {jpg_path}")
        print(f"   JPG size: {os.path.getsize(jpg_path)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ imageio method failed: {e}")
    
    # Method 3: Try imageio.v2
    print("\n🔄 Method 3: imageio.v2")
    try:
        import imageio.v2 as imageio_v2
        img_array = imageio_v2.imread(avif_path)
        print(f"✅ imageio.v2 successfully read AVIF image")
        print(f"   Shape: {img_array.shape}")
        print(f"   Dtype: {img_array.dtype}")
        
        # Convert to PIL and save
        from PIL import Image
        import numpy as np
        
        pil_img = Image.fromarray(img_array)
        if pil_img.mode in ('RGBA', 'LA', 'P'):
            pil_img = pil_img.convert('RGB')
        
        jpg_path = "/home/ubuntu/LTA_REFL/converted_imageio_v2.jpg"
        pil_img.save(jpg_path, 'JPEG', quality=95)
        print(f"✅ Successfully saved as JPG: {jpg_path}")
        print(f"   JPG size: {os.path.getsize(jpg_path)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ imageio.v2 method failed: {e}")
    
    # Method 4: Try ffmpeg subprocess
    print("\n🔄 Method 4: ffmpeg subprocess")
    try:
        jpg_path = "/home/ubuntu/LTA_REFL/converted_ffmpeg.jpg"
        
        # Use ffmpeg to convert AVIF to JPG
        cmd = ['ffmpeg', '-i', avif_path, '-y', jpg_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ ffmpeg successfully converted AVIF to JPG")
            print(f"   JPG path: {jpg_path}")
            print(f"   JPG size: {os.path.getsize(jpg_path)} bytes")
            return True
        else:
            print(f"❌ ffmpeg failed with return code: {result.returncode}")
            print(f"   Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ ffmpeg method failed: {e}")
    
    # Method 5: Try libavif tools
    print("\n🔄 Method 5: libavif tools")
    try:
        jpg_path = "/home/ubuntu/LTA_REFL/converted_libavif.jpg"
        
        # Use avifdec to convert AVIF to PNG first, then convert to JPG
        png_path = "/home/ubuntu/LTA_REFL/temp.png"
        
        # Convert AVIF to PNG
        cmd1 = ['avifdec', avif_path, png_path]
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
        
        if result1.returncode == 0:
            print(f"✅ libavif successfully converted AVIF to PNG")
            
            # Convert PNG to JPG using PIL
            try:
                from PIL import Image
                pil_img = Image.open(png_path)
                if pil_img.mode in ('RGBA', 'LA', 'P'):
                    pil_img = pil_img.convert('RGB')
                
                pil_img.save(jpg_path, 'JPEG', quality=95)
                print(f"✅ Successfully converted PNG to JPG: {jpg_path}")
                print(f"   JPG size: {os.path.getsize(jpg_path)} bytes")
                
                # Clean up temp PNG
                os.remove(png_path)
                return True
                
            except Exception as e:
                print(f"❌ PNG to JPG conversion failed: {e}")
                # Clean up temp PNG
                if os.path.exists(png_path):
                    os.remove(png_path)
        else:
            print(f"❌ libavif failed with return code: {result1.returncode}")
            print(f"   Error: {result1.stderr}")
            
    except Exception as e:
        print(f"❌ libavif method failed: {e}")
    
    return False

def install_missing_dependencies():
    """Install missing system dependencies"""
    print("\n🔧 Installing missing dependencies...")
    
    # Install ffmpeg
    try:
        print("Installing ffmpeg...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True)
        print("✅ ffmpeg installed successfully")
    except Exception as e:
        print(f"❌ Failed to install ffmpeg: {e}")
    
    # Install libavif
    try:
        print("Installing libavif...")
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libavif-dev', 'libavif-bin'], check=True)
        print("✅ libavif installed successfully")
    except Exception as e:
        print(f"❌ Failed to install libavif: {e}")

def main():
    """Main function"""
    print("🚀 EC2 AVIF to JPG Conversion Test")
    print("=" * 50)
    
    # Check system dependencies
    check_system_dependencies()
    
    # Try to convert AVIF to JPG
    success = test_avif_conversion_methods()
    
    if not success:
        print("\n❌ All conversion methods failed!")
        print("\n🔧 Attempting to install missing dependencies...")
        install_missing_dependencies()
        
        print("\n🔄 Retrying conversion after dependency installation...")
        success = test_avif_conversion_methods()
        
        if success:
            print("\n🎉 Success! AVIF conversion is now working.")
        else:
            print("\n❌ Still failing after dependency installation.")
            print("\n📋 Manual steps to try:")
            print("1. sudo apt-get update")
            print("2. sudo apt-get install -y ffmpeg libavif-dev libavif-bin")
            print("3. pip install --upgrade pillow-avif-plugin")
            print("4. Restart your application")
    else:
        print("\n🎉 Success! AVIF conversion is working.")
    
    print("\n📁 Check the converted JPG files in /home/ubuntu/LTA_REFL/")
    print("   - converted_pil.jpg")
    print("   - converted_imageio.jpg") 
    print("   - converted_imageio_v2.jpg")
    print("   - converted_ffmpeg.jpg")
    print("   - converted_libavif.jpg")

if __name__ == "__main__":
    main()
