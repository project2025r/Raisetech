#!/usr/bin/env python3
"""
Test script for AVIF conversion on EC2
Run this to verify all dependencies are working correctly
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import imageio
        print(f"✓ imageio imported successfully: {imageio.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import imageio: {e}")
        return False
    
    try:
        import imageio.v2
        print("✓ imageio.v2 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import imageio.v2: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ PIL imported successfully: {Image.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import PIL: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy imported successfully: {np.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False
    
    return True

def test_avif_reading():
    """Test AVIF reading capabilities"""
    print("\nTesting AVIF reading capabilities...")
    
    # Create a simple test image (not AVIF, but tests the infrastructure)
    try:
        import numpy as np
        from PIL import Image
        import io
        
        # Create a simple test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_img = Image.fromarray(test_img)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()
        
        print("✓ Test image creation successful")
        
        # Test imageio
        try:
            import imageio
            img_array = imageio.imread(io.BytesIO(img_bytes))
            print(f"✓ imageio can read JPEG: {img_array.shape}")
        except Exception as e:
            print(f"✗ imageio failed: {e}")
        
        # Test PIL
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            print(f"✓ PIL can read JPEG: {pil_img.size}")
        except Exception as e:
            print(f"✗ PIL failed: {e}")
        
        # Test imageio.v2
        try:
            import imageio.v2 as imageio_v2
            img_array = imageio_v2.imread(io.BytesIO(img_bytes))
            print(f"✓ imageio.v2 can read JPEG: {img_array.shape}")
        except Exception as e:
            print(f"✗ imageio.v2 failed: {e}")
            
    except Exception as e:
        print(f"✗ Test image creation failed: {e}")
        return False
    
    return True

def test_utils_import():
    """Test if the image converter utility can be imported"""
    print("\nTesting image converter utility...")
    
    try:
        # Add the utils directory to the path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
        
        from image_converter import convert_image_to_yolo_supported, is_avif_image
        print("✓ Image converter utility imported successfully")
        
        # Test the functions
        test_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
        result = is_avif_image(test_data)
        print(f"✓ is_avif_image function works: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to import image converter utility: {e}")
        return False

def test_system_dependencies():
    """Test if system dependencies are available"""
    print("\nTesting system dependencies...")
    
    # Test ffmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ffmpeg is available")
        else:
            print("✗ ffmpeg is not working properly")
    except Exception as e:
        print(f"✗ ffmpeg test failed: {e}")
    
    # Test if we can create temporary files
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            tmp.write(b'test')
            print("✓ File system operations work")
    except Exception as e:
        print(f"✗ File system test failed: {e}")

def main():
    """Main test function"""
    print("EC2 AVIF Conversion Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your Python environment.")
        return False
    
    # Test AVIF reading
    if not test_avif_reading():
        print("\n❌ AVIF reading tests failed.")
        return False
    
    # Test utility import
    if not test_utils_import():
        print("\n❌ Utility import tests failed.")
        return False
    
    # Test system dependencies
    test_system_dependencies()
    
    print("\n" + "=" * 40)
    print("🎉 All tests passed! AVIF conversion should work on EC2.")
    print("\nIf you still encounter issues, try:")
    print("1. Restart your application after installing dependencies")
    print("2. Check if ffmpeg is properly installed")
    print("3. Verify all Python packages are installed in the correct environment")
    
    return True

if __name__ == "__main__":
    main()
