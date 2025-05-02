import os
import shutil
import sys

def copy_directory(src, dst):
    """
    Copy the contents of a directory from src to dst.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    print(f"Copying files from {src} to {dst}...")
    
    # Get total number of files for progress reporting
    total_files = sum([len(files) for _, _, files in os.walk(src)])
    copied_files = 0
    
    for root, dirs, files in os.walk(src):
        # Create subdirectories in destination
        for directory in dirs:
            src_path = os.path.join(root, directory)
            dst_path = os.path.join(dst, os.path.relpath(src_path, src))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
        
        # Copy files
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst, os.path.relpath(src_file, src))
            
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            
            # Copy file
            shutil.copy2(src_file, dst_file)
            copied_files += 1
            
            # Print progress
            progress = (copied_files / total_files) * 100
            print(f"Progress: {progress:.1f}% - Copied: {src_file} -> {dst_file}")

def copy_logo_to_frontend():
    """
    Copy the logo from assets to frontend for use in the UI.
    """
    src_dir = os.path.join("assets")
    dst_dir = os.path.join("frontend", "src", "assets")
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Look for logo files
    logo_files = [
        "Logo.jpg", 
        "logo_BG.png", 
        "LOGO AI SPRAY.png", 
        "aisprylogo.png",
        "raise.png"
    ]
    
    for logo_file in logo_files:
        src_file = os.path.join(src_dir, logo_file)
        if os.path.exists(src_file):
            dst_file = os.path.join(dst_dir, "logo.png" if logo_file == logo_files[0] else logo_file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied logo: {src_file} -> {dst_file}")
            break

def main():
    # Check if we're in the ReactApp directory
    if not os.path.basename(os.getcwd()) == "ReactApp":
        print("Error: Please run this script from the ReactApp directory.")
        sys.exit(1)
    
    # Set source and destination paths
    streamlit_app_dir = os.path.join("..", "StreamlitApp")
    
    if not os.path.exists(streamlit_app_dir):
        print(f"Error: Could not find StreamlitApp directory at {streamlit_app_dir}")
        sys.exit(1)
    
    # Define directories to copy
    asset_dirs = [
        ("assets", "assets"),
        ("dashboard", "dashboard"),
        ("map_frames", "map_frames")
    ]
    
    # Copy each directory
    for src_dir, dst_dir in asset_dirs:
        src_path = os.path.join(streamlit_app_dir, src_dir)
        dst_path = os.path.join(dst_dir)
        
        if os.path.exists(src_path):
            copy_directory(src_path, dst_path)
        else:
            print(f"Warning: Source directory not found: {src_path}")
    
    # Copy logo to frontend assets
    copy_logo_to_frontend()
    
    print("\nSetup complete! All assets have been copied.")
    print("\nNext steps:")
    print("1. Set up the backend: cd backend && pip install -r requirements.txt")
    print("2. Start the backend: python app.py")
    print("3. Set up the frontend: cd frontend && npm install")
    print("4. Start the frontend: npm start")

if __name__ == "__main__":
    main() 