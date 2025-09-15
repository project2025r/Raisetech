/**
 * Client-side File Validation Utility
 * 
 * This module provides comprehensive file validation for image and video uploads
 * with user-friendly error messages that match the backend validation.
 */

// Allowed file types configuration
const ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.avif'];
const ALLOWED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'];

const ALLOWED_IMAGE_MIMES = [
  'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 
  'image/tiff', 'image/tif', 'image/webp', 'image/avif'
];

const ALLOWED_VIDEO_MIMES = [
  'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 
  'video/x-matroska', 'video/x-ms-wmv', 'video/x-flv', 'video/webm',
  'video/x-m4v', 'video/3gpp', 'video/ogg'
];

// File size limits (in bytes)
const MAX_IMAGE_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_VIDEO_SIZE = 500 * 1024 * 1024; // 500MB

/**
 * Get file extension from filename in lowercase
 * @param {string} filename - The filename
 * @returns {string} File extension in lowercase (including the dot)
 */
export const getFileExtension = (filename) => {
  if (!filename) return "";
  const lastDotIndex = filename.lastIndexOf('.');
  return lastDotIndex !== -1 ? filename.slice(lastDotIndex).toLowerCase() : "";
};

/**
 * Validate image file
 * @param {File} file - The file object to validate
 * @param {string} context - Additional context for error messages
 * @returns {Object} {isValid: boolean, errorMessage: string}
 */
export const validateImageFile = (file, context = "") => {
  try {
    if (!file) {
      return { isValid: false, errorMessage: "No image file provided." };
    }

    // Check file extension
    const fileExtension = getFileExtension(file.name);
    if (!ALLOWED_IMAGE_EXTENSIONS.includes(fileExtension)) {
      return {
        isValid: false,
        errorMessage: `Invalid file format. Please upload only image files (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif). Received: ${fileExtension || 'unknown'}`
      };
    }

    // Check MIME type
    if (file.type && !ALLOWED_IMAGE_MIMES.includes(file.type.toLowerCase())) {
      return {
        isValid: false,
        errorMessage: `Invalid file format. Please upload only image files. The file appears to be: ${file.type}`
      };
    }

    // Check file size
    if (file.size > MAX_IMAGE_SIZE) {
      return {
        isValid: false,
        errorMessage: `Image file size (${(file.size / (1024*1024)).toFixed(1)}MB) exceeds maximum allowed size (${MAX_IMAGE_SIZE / (1024*1024)}MB). Please compress your image or choose a smaller file.`
      };
    }

    return { isValid: true, errorMessage: "" };

  } catch (error) {
    console.error("Error during image validation:", error);
    return {
      isValid: false,
      errorMessage: "An error occurred while validating the image file. Please try again with a different file."
    };
  }
};

/**
 * Validate video file
 * @param {File} file - The file object to validate
 * @param {string} context - Additional context for error messages
 * @returns {Object} {isValid: boolean, errorMessage: string}
 */
export const validateVideoFile = (file, context = "") => {
  try {
    if (!file) {
      return { isValid: false, errorMessage: "No video file provided." };
    }

    // Check file extension
    const fileExtension = getFileExtension(file.name);
    if (!ALLOWED_VIDEO_EXTENSIONS.includes(fileExtension)) {
      return {
        isValid: false,
        errorMessage: `Invalid file format. Please upload only video files (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp, .ogv). Received: ${fileExtension || 'unknown'}`
      };
    }

    // Check MIME type
    if (file.type && !ALLOWED_VIDEO_MIMES.includes(file.type.toLowerCase())) {
      return {
        isValid: false,
        errorMessage: `Invalid file format. Please upload only video files. The file appears to be: ${file.type}`
      };
    }

    // Check file size
    if (file.size > MAX_VIDEO_SIZE) {
      return {
        isValid: false,
        errorMessage: `Video file size (${(file.size / (1024*1024)).toFixed(1)}MB) exceeds maximum allowed size (${MAX_VIDEO_SIZE / (1024*1024)}MB). Please compress your video or choose a smaller file.`
      };
    }

    return { isValid: true, errorMessage: "" };

  } catch (error) {
    console.error("Error during video validation:", error);
    return {
      isValid: false,
      errorMessage: "An error occurred while validating the video file. Please try again with a different file."
    };
  }
};

/**
 * Main validation function for uploaded files
 * @param {File} file - The file object to validate
 * @param {string} expectedType - Expected file type ('image' or 'video')
 * @param {string} context - Additional context for error messages
 * @returns {Object} {isValid: boolean, errorMessage: string}
 */
export const validateUploadFile = (file, expectedType, context = "") => {
  try {
    if (expectedType === 'image') {
      return validateImageFile(file, context);
    } else if (expectedType === 'video') {
      return validateVideoFile(file, context);
    } else {
      return {
        isValid: false,
        errorMessage: `Invalid expected file type: ${expectedType}. Must be 'image' or 'video'.`
      };
    }
  } catch (error) {
    console.error("Error during file validation:", error);
    return {
      isValid: false,
      errorMessage: "An error occurred while validating the file. Please try again."
    };
  }
};

/**
 * Validate multiple files
 * @param {FileList|Array} files - The files to validate
 * @param {string} expectedType - Expected file type ('image' or 'video')
 * @param {string} context - Additional context for error messages
 * @returns {Object} {isValid: boolean, errorMessage: string, invalidFiles: Array}
 */
export const validateMultipleFiles = (files, expectedType, context = "") => {
  const filesArray = Array.from(files);
  const invalidFiles = [];
  
  for (let i = 0; i < filesArray.length; i++) {
    const file = filesArray[i];
    const validation = validateUploadFile(file, expectedType, context);
    
    if (!validation.isValid) {
      invalidFiles.push({
        file: file,
        error: validation.errorMessage
      });
    }
  }
  
  if (invalidFiles.length > 0) {
    const firstError = invalidFiles[0].error;
    const errorMessage = invalidFiles.length === 1 
      ? `File "${invalidFiles[0].file.name}": ${firstError}`
      : `${invalidFiles.length} file(s) have validation errors. First error - "${invalidFiles[0].file.name}": ${firstError}`;
    
    return {
      isValid: false,
      errorMessage: errorMessage,
      invalidFiles: invalidFiles
    };
  }
  
  return {
    isValid: true,
    errorMessage: "",
    invalidFiles: []
  };
};

/**
 * Get context-specific error message for file validation
 * @param {string} expectedType - Expected file type ('image' or 'video')
 * @param {string} context - Additional context
 * @returns {string} Context-specific error message
 */
export const getContextSpecificErrorMessage = (expectedType, context = "") => {
  if (expectedType === 'image') {
    if (context.toLowerCase().includes('pothole')) {
      return "Invalid file format. Please upload only image files for pothole detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif).";
    } else if (context.toLowerCase().includes('crack')) {
      return "Invalid file format. Please upload only image files for crack detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif).";
    } else if (context.toLowerCase().includes('kerb')) {
      return "Invalid file format. Please upload only image files for kerb detection (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif).";
    } else {
      return "Invalid file format. Please upload only image files (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .avif).";
    }
  } else if (expectedType === 'video') {
    return "Invalid file format. Please upload only video files (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp, .ogv).";
  } else {
    return "Invalid file format. Please upload only image or video files.";
  }
};

/**
 * Show user-friendly file validation error
 * @param {string} errorMessage - The error message to display
 * @param {Function} setErrorCallback - Callback function to set error state
 */
export const showFileValidationError = (errorMessage, setErrorCallback) => {
  if (setErrorCallback) {
    setErrorCallback(errorMessage);
  } else {
    // Fallback to alert if no callback provided
    alert(errorMessage);
  }
};

// Export constants for use in components
export {
  ALLOWED_IMAGE_EXTENSIONS,
  ALLOWED_VIDEO_EXTENSIONS,
  ALLOWED_IMAGE_MIMES,
  ALLOWED_VIDEO_MIMES,
  MAX_IMAGE_SIZE,
  MAX_VIDEO_SIZE
};
