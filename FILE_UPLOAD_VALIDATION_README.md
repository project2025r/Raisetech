# File Upload Validation Implementation

## Overview

This implementation provides comprehensive file upload validation for the LTA application to ensure that only valid image and video files are accepted, preventing system crashes and providing user-friendly error messages.

## Features Implemented

### ✅ Backend Validation

1. **File Type Validation**
   - MIME type checking
   - File extension validation
   - File signature verification (magic bytes)

2. **Supported File Types**
   - **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.avif`
   - **Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.ogv`

3. **File Size Limits**
   - Images: Maximum 50MB
   - Videos: Maximum 500MB

4. **Security Features**
   - Base64 image validation
   - Content validation
   - Malicious file detection

### ✅ Frontend Validation

1. **Client-side Validation**
   - File type checking before upload
   - File size validation
   - User-friendly error messages

2. **Real-time Feedback**
   - Immediate validation on file selection
   - Clear error messages
   - Input field clearing on invalid files

## Files Modified/Created

### Backend Files

1. **`utils/file_validation.py`** (NEW)
   - Core validation logic
   - MIME type and extension checking
   - File size validation
   - Context-specific error messages

2. **`routes/pavement.py`** (MODIFIED)
   - Added validation to all image detection endpoints
   - Added validation to video detection endpoint
   - Integrated with existing error handling

3. **`routes/road_infrastructure.py`** (MODIFIED)
   - Added validation to image and video uploads
   - Integrated with existing error handling

### Frontend Files

1. **`utils/fileValidation.js`** (NEW)
   - Client-side validation utilities
   - File type and size checking
   - Error message handling

2. **`pages/Pavement.js`** (MODIFIED)
   - Added validation to file upload handler
   - Integrated error display

3. **`components/VideoDefectDetection.js`** (MODIFIED)
   - Added validation to video upload handler
   - Integrated error display

4. **`pages/RoadInfrastructure.js`** (MODIFIED)
   - Added validation to image and video upload handlers
   - Integrated error display

### Test Files

1. **`simple_validation_test.py`** (NEW)
   - Comprehensive test suite
   - Demonstrates validation functionality
   - Real-world scenario testing

## Error Messages

The system provides context-specific error messages:

### Image Upload Errors
- **Pothole Detection**: "Invalid file format. Please upload only image files for pothole detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif)."
- **Crack Detection**: "Invalid file format. Please upload only image files for crack detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif)."
- **Kerb Detection**: "Invalid file format. Please upload only image files for kerb detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif)."

### Video Upload Errors
- "Invalid file format. Please upload only video files (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp, .ogv)."

### File Size Errors
- "Image file size (XX.XMB) exceeds maximum allowed size (50MB). Please compress your image or choose a smaller file."
- "Video file size (XX.XMB) exceeds maximum allowed size (500MB). Please compress your video or choose a smaller file."

## How It Works

### Backend Validation Flow

1. **File Upload Request** → File received via `request.files` or base64 data
2. **Validation Check** → `validate_upload_file()` or `validate_base64_image()`
3. **Multiple Checks**:
   - File extension validation
   - MIME type validation
   - File size validation
   - Content signature validation (optional)
4. **Result** → Return success or user-friendly error message

### Frontend Validation Flow

1. **File Selection** → User selects file(s)
2. **Immediate Validation** → `validateUploadFile()` or `validateMultipleFiles()`
3. **Error Handling** → Display error message and clear input if invalid
4. **Success** → Proceed with file processing

## Testing

Run the validation test suite:

```bash
cd LTA/backend
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Run tests
python simple_validation_test.py
```

## Security Benefits

1. **Prevents System Crashes** - Invalid files are caught before processing
2. **Malicious File Protection** - File signature checking prevents disguised files
3. **Resource Protection** - File size limits prevent DoS attacks
4. **User Experience** - Clear error messages guide users to correct actions

## Example Usage

### Backend (Python)
```python
from utils.file_validation import validate_upload_file, validate_base64_image

# For file uploads
is_valid, error_message = validate_upload_file(file_obj, 'image')
if not is_valid:
    return jsonify({"success": False, "message": error_message}), 400

# For base64 images
is_valid, error_message = validate_base64_image(image_data, 'pothole_detection')
if not is_valid:
    return jsonify({"success": False, "message": error_message}), 400
```

### Frontend (JavaScript)
```javascript
import { validateUploadFile, showFileValidationError } from '../utils/fileValidation';

const handleFileChange = (e) => {
  const file = e.target.files[0];
  if (file) {
    const validation = validateUploadFile(file, 'image', 'detection');
    if (!validation.isValid) {
      showFileValidationError(validation.errorMessage, setError);
      e.target.value = '';
      return;
    }
    // Proceed with valid file...
  }
};
```

## Maintenance

- **Adding New File Types**: Update `ALLOWED_*_EXTENSIONS` and `ALLOWED_*_MIMES` arrays
- **Changing Size Limits**: Modify `MAX_IMAGE_SIZE` and `MAX_VIDEO_SIZE` constants
- **Custom Error Messages**: Update `get_context_specific_error_message()` function

## Conclusion

This implementation ensures that the LTA application gracefully handles invalid file uploads with proper exception handling and user-friendly error messages, preventing system crashes and improving user experience.
