import React, { useState, useEffect } from 'react';
import { Card, Button, Form, Alert, Spinner, Badge, ProgressBar } from 'react-bootstrap';
import { getDeviceCoordinates, validateCoordinates, formatCoordinates, getAccuracyDescription } from '../utils/deviceCoordinates';

/**
 * Enhanced Image Upload Component with Device Coordinate Integration
 * Automatically captures device coordinates and integrates with EXIF data
 */
const EnhancedImageUpload = ({ onUpload, onCoordinatesUpdate }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [coordinates, setCoordinates] = useState(null);
  const [coordinateSource, setCoordinateSource] = useState(null);
  const [isGettingLocation, setIsGettingLocation] = useState(false);
  const [locationError, setLocationError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  // Automatically get coordinates when component mounts
  useEffect(() => {
    handleGetCoordinates();
  }, []);

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      
      // Try to extract EXIF coordinates from the image
      extractImageCoordinates(file);
    }
  };

  // Extract coordinates from image EXIF data
  const extractImageCoordinates = (file) => {
    console.log('📷 Attempting to extract EXIF coordinates from image...');
    
    // Use FileReader to read the image
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        // Create an image element to load the file
        const img = new Image();
        img.onload = () => {
          // Note: Browser-based EXIF extraction is limited
          // The backend will handle comprehensive EXIF extraction
          console.log('📷 Image loaded, EXIF extraction will be handled by backend');
        };
        img.src = e.target.result;
      } catch (error) {
        console.warn('❌ Client-side EXIF extraction failed:', error);
      }
    };
    reader.readAsDataURL(file);
  };

  // Get device coordinates
  const handleGetCoordinates = async () => {
    setIsGettingLocation(true);
    setLocationError(null);
    
    try {
      const coords = await getDeviceCoordinates({
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 300000, // 5 minutes
        fallbackToIP: true
      });
      
      setCoordinates(coords);
      setCoordinateSource(coords.source);
      
      // Notify parent component
      if (onCoordinatesUpdate) {
        onCoordinatesUpdate(coords);
      }
      
      console.log('✅ Device coordinates obtained:', coords);
    } catch (error) {
      setLocationError(error.message);
      console.error('❌ Failed to get coordinates:', error);
    } finally {
      setIsGettingLocation(false);
    }
  };

  // Handle upload
  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Convert file to base64
      const base64Image = await fileToBase64(selectedFile);
      
      // Prepare upload data with coordinates
      const uploadData = {
        image: base64Image,
        coordinates: coordinates ? coordinates.formatted : null,
        deviceCoordinates: coordinates,
        timestamp: new Date().toISOString(),
        filename: selectedFile.name,
        fileSize: selectedFile.size,
        fileType: selectedFile.type
      };

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 200);

      // Call parent upload handler
      if (onUpload) {
        await onUpload(uploadData);
      }

      setUploadProgress(100);
      
      // Reset form after successful upload
      setTimeout(() => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setUploadProgress(0);
        setIsUploading(false);
        
        // Get fresh coordinates for next upload
        handleGetCoordinates();
      }, 1000);

    } catch (error) {
      console.error('❌ Upload failed:', error);
      setIsUploading(false);
      setUploadProgress(0);
      alert('Upload failed: ' + error.message);
    }
  };

  // Convert file to base64
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  };

  return (
    <Card className="mb-4">
      <Card.Header>
        <h5 className="mb-0">📷 Enhanced Image Upload</h5>
      </Card.Header>
      <Card.Body>
        {/* File Selection */}
        <Form.Group className="mb-3">
          <Form.Label>Select Image</Form.Label>
          <Form.Control
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            disabled={isUploading}
          />
        </Form.Group>

        {/* Image Preview */}
        {previewUrl && (
          <div className="mb-3 text-center">
            <img
              src={previewUrl}
              alt="Preview"
              className="img-fluid border rounded"
              style={{ maxHeight: '200px' }}
            />
          </div>
        )}

        {/* Coordinate Information */}
        <Card className="mb-3 bg-light">
          <Card.Body>
            <div className="d-flex justify-content-between align-items-center mb-2">
              <h6 className="mb-0">📍 Location Information</h6>
              <Button
                variant="outline-primary"
                size="sm"
                onClick={handleGetCoordinates}
                disabled={isGettingLocation || isUploading}
              >
                {isGettingLocation ? (
                  <>
                    <Spinner animation="border" size="sm" className="me-1" />
                    Getting Location...
                  </>
                ) : (
                  '🔄 Refresh Location'
                )}
              </Button>
            </div>

            {coordinates && (
              <div>
                <div className="mb-2">
                  <strong>Coordinates:</strong> {formatCoordinates(coordinates)}
                </div>
                <div className="mb-2">
                  <Badge bg={coordinateSource === 'GPS' ? 'success' : 'warning'}>
                    {coordinateSource === 'GPS' ? '📡 GPS' : '🌐 IP-based'}
                  </Badge>
                  {coordinates.accuracy && (
                    <Badge bg="info" className="ms-2">
                      {getAccuracyDescription(coordinates.accuracy)}
                    </Badge>
                  )}
                </div>
                {coordinates.city && (
                  <div className="small text-muted">
                    📍 {coordinates.city}, {coordinates.region}, {coordinates.country}
                  </div>
                )}
              </div>
            )}

            {locationError && (
              <Alert variant="warning" className="mb-0">
                <small>⚠️ {locationError}</small>
              </Alert>
            )}

            {isGettingLocation && (
              <div className="text-center">
                <Spinner animation="border" size="sm" />
                <span className="ms-2">Getting your location...</span>
              </div>
            )}
          </Card.Body>
        </Card>

        {/* Upload Progress */}
        {isUploading && (
          <div className="mb-3">
            <div className="d-flex justify-content-between mb-1">
              <small>Uploading...</small>
              <small>{uploadProgress}%</small>
            </div>
            <ProgressBar now={uploadProgress} />
          </div>
        )}

        {/* Upload Button */}
        <div className="d-grid">
          <Button
            variant="primary"
            onClick={handleUpload}
            disabled={!selectedFile || isUploading || isGettingLocation}
            size="lg"
          >
            {isUploading ? (
              <>
                <Spinner animation="border" size="sm" className="me-2" />
                Uploading...
              </>
            ) : (
              '📤 Upload Image'
            )}
          </Button>
        </div>

        {/* Information */}
        <div className="mt-3">
          <small className="text-muted">
            <strong>Features:</strong>
            <ul className="mb-0 mt-1">
              <li>Automatic device coordinate capture</li>
              <li>EXIF metadata extraction and integration</li>
              <li>GPS and IP-based location fallback</li>
              <li>Real-time map coordinate updates</li>
            </ul>
          </small>
        </div>
      </Card.Body>
    </Card>
  );
};

export default EnhancedImageUpload;
