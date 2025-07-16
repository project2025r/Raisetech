import React, { useState, useRef, useEffect } from 'react';
import { Container, Card, Button, Form, Tabs, Tab, Alert, Spinner, OverlayTrigger, Popover } from 'react-bootstrap';
import axios from 'axios';
import Webcam from 'react-webcam';
import './Pavement.css';
import useResponsive from '../hooks/useResponsive';
import VideoDefectDetection from '../components/VideoDefectDetection';

const Pavement = () => {
  const [activeTab, setActiveTab] = useState('detection');
  const [detectionType, setDetectionType] = useState('all');
  const [imageFiles, setImageFiles] = useState([]);
  const [imagePreviewsMap, setImagePreviewsMap] = useState({});
  const [imageLocationMap, setImageLocationMap] = useState({});
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [processedImage, setProcessedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [coordinates, setCoordinates] = useState('Not Available');
  const [cameraOrientation, setCameraOrientation] = useState('environment');
  const [locationPermission, setLocationPermission] = useState('unknown');
  const [locationError, setLocationError] = useState('');
  const [locationLoading, setLocationLoading] = useState(false);
  
  // Add state for batch processing results
  const [batchResults, setBatchResults] = useState([]);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [processedCount, setProcessedCount] = useState(0);
  const [totalToProcess, setTotalToProcess] = useState(0);
  
  // Add state for auto-navigation through results
  const [autoNavigationActive, setAutoNavigationActive] = useState(false);
  const [autoNavigationIndex, setAutoNavigationIndex] = useState(0);
  const autoNavigationRef = useRef(null);
  
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const { isMobile } = useResponsive();

  // Create the popover content
  const reminderPopover = (
    <Popover id="reminder-popover" style={{ maxWidth: '300px' }}>
      <Popover.Header as="h3">📸 Image Upload Guidelines</Popover.Header>
      <Popover.Body>
        <p style={{ marginBottom: '10px' }}>
          Please ensure your uploaded images are:
        </p>
        <ul style={{ marginBottom: '0', paddingLeft: '20px' }}>
          <li>Focused directly on the road surface</li>
          <li>Well-lit and clear</li>
          <li>Showing the entire area of concern</li>
          <li>Taken from a reasonable distance to capture context</li>
        </ul>
      </Popover.Body>
    </Popover>
  );

  // Safari-compatible geolocation permission check
  const checkLocationPermission = async () => {
    if (!navigator.permissions || !navigator.permissions.query) {
      // Fallback for older browsers
      return 'prompt';
    }
    
    try {
      const permission = await navigator.permissions.query({ name: 'geolocation' });
      return permission.state;
    } catch (err) {
      console.warn('Permission API not supported or failed:', err);
      return 'prompt';
    }
  };

  // Safari-compatible geolocation request
  const requestLocation = () => {
    return new Promise((resolve, reject) => {
      // Check if geolocation is supported
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported by this browser'));
        return;
      }

      // Check if we're in a secure context (HTTPS)
      if (!window.isSecureContext) {
        reject(new Error('Geolocation requires a secure context (HTTPS)'));
        return;
      }

      const options = {
        enableHighAccuracy: true,
        timeout: 15000, // 15 seconds timeout
        maximumAge: 60000 // Accept cached position up to 1 minute old
      };

      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve(position);
        },
        (error) => {
          let errorMessage = 'Unable to retrieve location';
          
          switch (error.code) {
            case error.PERMISSION_DENIED:
              errorMessage = 'Location access denied. Please enable location permissions in your browser settings.';
              break;
            case error.POSITION_UNAVAILABLE:
              errorMessage = 'Location information is unavailable. Please try again.';
              break;
            case error.TIMEOUT:
              errorMessage = 'Location request timed out. Please try again.';
              break;
            default:
              errorMessage = `Location error: ${error.message}`;
              break;
          }
          
          reject(new Error(errorMessage));
        },
        options
      );
    });
  };

  // Enhanced location handler with Safari-specific fixes
  const handleLocationRequest = async () => {
    setLocationLoading(true);
    setLocationError('');
    
    try {
      // First check permission state
      const permissionState = await checkLocationPermission();
      setLocationPermission(permissionState);
      
      // If permission is denied, provide user guidance
      if (permissionState === 'denied') {
        const errorMsg = 'Location access denied. To enable location access:\n' +
                        '• Safari: Settings > Privacy & Security > Location Services\n' +
                        '• Chrome: Settings > Privacy > Location\n' +
                        '• Firefox: Settings > Privacy > Location\n' +
                        'Then refresh this page and try again.';
        setLocationError(errorMsg);
        setCoordinates('Permission Denied');
        return;
      }
      
      // Request location
      const position = await requestLocation();
      const { latitude, longitude } = position.coords;
      
      // Format coordinates with better precision
      const formattedCoords = `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
      setCoordinates(formattedCoords);
      setLocationPermission('granted');
      setLocationError('');
      
      console.log('Location acquired:', { latitude, longitude, accuracy: position.coords.accuracy });
      
    } catch (error) {
      console.error('Location request failed:', error);
      setLocationError(error.message);
      setCoordinates('Location Error');
      
      // Update permission state based on error
      if (error.message.includes('denied')) {
        setLocationPermission('denied');
      }
    } finally {
      setLocationLoading(false);
    }
  };

  // Handle multiple file input change
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      setImageFiles([...imageFiles, ...files]);
      
      // Create previews and location data for each file
      files.forEach(file => {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreviewsMap(prev => ({
            ...prev,
            [file.name]: reader.result
          }));
        };
        reader.readAsDataURL(file);
        
        // Store location as "Not Available" for uploaded files
        setImageLocationMap(prev => ({
          ...prev,
          [file.name]: 'Not Available'
        }));
      });
      
      // Reset results
      setProcessedImage(null);
      setResults(null);
      setError('');
    }
  };

  // Handle camera capture with location validation
  const handleCapture = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      // If we don't have location data, try to get it before capturing
      if (coordinates === 'Not Available' || coordinates === 'Location Error') {
        await handleLocationRequest();
      }
      
      const timestamp = new Date().toISOString();
      const filename = `camera_capture_${timestamp}.jpg`;
      const captureCoordinates = coordinates; // Capture current coordinates
      
      setImageFiles([...imageFiles, filename]);
      setImagePreviewsMap(prev => ({
        ...prev,
        [filename]: imageSrc
      }));
      setImageLocationMap(prev => ({
        ...prev,
        [filename]: captureCoordinates
      }));
      setCurrentImageIndex(imageFiles.length);
      
      setProcessedImage(null);
      setResults(null);
      setError('');
      
      // Log capture with current coordinates
      console.log('Photo captured with coordinates:', captureCoordinates);
    }
  };

  // Get location data for currently selected image
  const getCurrentImageLocation = () => {
    if (Object.keys(imagePreviewsMap).length === 0) {
      return coordinates; // Use current coordinates if no images
    }
    
    const currentFilename = Object.keys(imagePreviewsMap)[currentImageIndex];
    return imageLocationMap[currentFilename] || 'Not Available';
  };

  // Toggle camera with improved location handling
  const toggleCamera = async () => {
    const newCameraState = !cameraActive;
    setCameraActive(newCameraState);
    
    if (newCameraState) {
      // Get location when camera is activated
      await handleLocationRequest();
    } else {
      // Only reset location if no images are captured
      // This preserves location data for captured images
      if (Object.keys(imagePreviewsMap).length === 0) {
        setCoordinates('Not Available');
        setLocationError('');
        setLocationPermission('unknown');
      }
    }
  };

  // Toggle camera orientation (front/back) for mobile devices
  const toggleCameraOrientation = () => {
    setCameraOrientation(prev => prev === 'environment' ? 'user' : 'environment');
  };

  // Process image for detection
  const handleProcess = async () => {
    setLoading(true);
    setError('');

    try {
      // Get user info from session storage
      const userString = sessionStorage.getItem('user');
      const user = userString ? JSON.parse(userString) : null;
      
      // Get the currently selected image
      const currentImagePreview = Object.values(imagePreviewsMap)[currentImageIndex];
      
      if (!currentImagePreview) {
        setError('No image selected for processing');
        setLoading(false);
        return;
      }
      
      // Get coordinates for the current image
      const imageCoordinates = getCurrentImageLocation();
      
      // Prepare request data
      const requestData = {
        image: currentImagePreview,
        coordinates: imageCoordinates,
        username: user?.username || 'Unknown',
        role: user?.role || 'Unknown'
      };

      // Determine endpoint based on detection type
      let endpoint;
      switch(detectionType) {
        case 'all':
          endpoint = '/api/pavement/detect-all';
          break;
        case 'potholes':
          endpoint = '/api/pavement/detect-potholes';
          break;
        case 'cracks':
          endpoint = '/api/pavement/detect-cracks';
          break;
        case 'kerbs':
          endpoint = '/api/pavement/detect-kerbs';
          break;
        default:
          endpoint = '/api/pavement/detect-all';
      }

      // Make API request
      const response = await axios.post(endpoint, requestData);

      // Handle response
      if (response.data.success) {
        setProcessedImage(response.data.processed_image);
        setResults(response.data);
      } else {
        setError(response.data.message || 'Detection failed');
      }
    } catch (error) {
      setError(
        error.response?.data?.message || 
        'An error occurred during detection. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  // Add a new function to process all images
  const handleProcessAll = async () => {
    if (Object.keys(imagePreviewsMap).length === 0) {
      setError('No images to process');
      return;
    }

    setBatchProcessing(true);
    setError('');
    setBatchResults([]);
    setProcessedCount(0);
    setTotalToProcess(Object.keys(imagePreviewsMap).length);
    
    // Get user info from session storage
    const userString = sessionStorage.getItem('user');
    const user = userString ? JSON.parse(userString) : null;
    
    try {
      // Determine endpoint based on detection type
      let endpoint;
      switch(detectionType) {
        case 'all':
          endpoint = '/api/pavement/detect-all';
          break;
        case 'potholes':
          endpoint = '/api/pavement/detect-potholes';
          break;
        case 'cracks':
          endpoint = '/api/pavement/detect-cracks';
          break;
        case 'kerbs':
          endpoint = '/api/pavement/detect-kerbs';
          break;
        default:
          endpoint = '/api/pavement/detect-all';
      }
      
      const results = [];
      const filenames = Object.keys(imagePreviewsMap);
      
      // Process each image sequentially and display immediately
      for (let i = 0; i < filenames.length; i++) {
        const filename = filenames[i];
        const imageData = imagePreviewsMap[filename];
        
        try {
          // Update current image index to show which image is being processed
          setCurrentImageIndex(i);
          
          // Get coordinates for this specific image
          const imageCoordinates = imageLocationMap[filename] || 'Not Available';
          
          // Prepare request data
          const requestData = {
            image: imageData,
            coordinates: imageCoordinates,
            username: user?.username || 'Unknown',
            role: user?.role || 'Unknown'
          };
          
          // Make API request
          const response = await axios.post(endpoint, requestData);
          
          if (response.data.success) {
            // Immediately display the processed image
            setProcessedImage(response.data.processed_image);
            setResults(response.data);
            
            results.push({
              filename,
              success: true,
              processedImage: response.data.processed_image,
              data: response.data
            });
          } else {
            results.push({
              filename,
              success: false,
              error: response.data.message || 'Detection failed'
            });
          }
        } catch (error) {
          results.push({
            filename,
            success: false,
            error: error.response?.data?.message || 'An error occurred during detection'
          });
        }
        
        // Update progress
        setProcessedCount(prev => prev + 1);
        
        // Pause briefly to allow user to see the result before moving to next image
        // Only pause if not on the last image
        if (i < filenames.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 2000)); // 2 second pause
        }
      }
      
      // Store final results but don't show the batch summary
      setBatchResults(results);
      
    } catch (error) {
      setError('Failed to process batch: ' + (error.message || 'Unknown error'));
    } finally {
      setBatchProcessing(false);
    }
  };

  // Reset detection
  const handleReset = () => {
    setImageFiles([]);
    setImagePreviewsMap({});
    setImageLocationMap({});
    setCurrentImageIndex(0);
    setProcessedImage(null);
    setResults(null);
    setError('');
    setBatchResults([]);
    setProcessedCount(0);
    setTotalToProcess(0);
    
    // Reset coordinates when clearing all images
    setCoordinates('Not Available');
    setLocationError('');
    setLocationPermission('unknown');
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Add a function to handle auto-navigation through results
  const startAutoNavigation = () => {
    if (batchResults.length === 0) return;
    
    // Find only successful results
    const successfulResults = batchResults.filter(result => result.success);
    if (successfulResults.length === 0) return;
    
    setAutoNavigationActive(true);
    setAutoNavigationIndex(0);
    
    // Display the first result
    const firstResult = successfulResults[0];
    const fileIndex = Object.keys(imagePreviewsMap).findIndex(
      filename => filename === firstResult.filename
    );
    
    if (fileIndex !== -1) {
      setCurrentImageIndex(fileIndex);
      setProcessedImage(firstResult.processedImage);
      setResults(firstResult.data);
    }
    
    // Set up interval for auto-navigation
    autoNavigationRef.current = setInterval(() => {
      setAutoNavigationIndex(prevIndex => {
        const nextIndex = prevIndex + 1;
        
        // If we've reached the end, stop auto-navigation
        if (nextIndex >= successfulResults.length) {
          clearInterval(autoNavigationRef.current);
          setAutoNavigationActive(false);
          return prevIndex;
        }
        
        // Display the next result
        const nextResult = successfulResults[nextIndex];
        const nextFileIndex = Object.keys(imagePreviewsMap).findIndex(
          filename => filename === nextResult.filename
        );
        
        if (nextFileIndex !== -1) {
          setCurrentImageIndex(nextFileIndex);
          setProcessedImage(nextResult.processedImage);
          setResults(nextResult.data);
        }
        
        return nextIndex;
      });
    }, 3000); // Change results every 3 seconds
  };

  // Clean up interval on component unmount
  useEffect(() => {
    return () => {
      if (autoNavigationRef.current) {
        clearInterval(autoNavigationRef.current);
      }
    };
  }, []);

  // Handle location permission changes
  useEffect(() => {
    if (cameraActive && locationPermission === 'unknown') {
      // Try to get location when camera is first activated
      handleLocationRequest();
    }
  }, [cameraActive]);

  // Listen for permission changes if supported
  useEffect(() => {
    let permissionWatcher = null;
    
    const watchPermissions = async () => {
      try {
        if (navigator.permissions && navigator.permissions.query) {
          const permission = await navigator.permissions.query({ name: 'geolocation' });
          
          permissionWatcher = () => {
            setLocationPermission(permission.state);
            if (permission.state === 'granted' && cameraActive && coordinates === 'Not Available') {
              handleLocationRequest();
            }
          };
          
          permission.addEventListener('change', permissionWatcher);
        }
      } catch (err) {
        console.warn('Permission watching not supported:', err);
      }
    };
    
    watchPermissions();
    
    return () => {
      if (permissionWatcher) {
        try {
          const permission = navigator.permissions.query({ name: 'geolocation' });
          permission.then(p => p.removeEventListener('change', permissionWatcher));
        } catch (err) {
          console.warn('Error removing permission listener:', err);
        }
      }
    };
  }, [cameraActive, coordinates]);

  // Force re-render when current image changes to update location display
  useEffect(() => {
    // This effect ensures the UI updates when switching between images
    // The getCurrentImageLocation function will return the correct location for the selected image
  }, [currentImageIndex, imageLocationMap]);

  // Stop auto-navigation
  const stopAutoNavigation = () => {
    if (autoNavigationRef.current) {
      clearInterval(autoNavigationRef.current);
      setAutoNavigationActive(false);
    }
  };

  return (
    <Container className="pavement-page">
      <h2 className="big-font mb-4">Pavement Analysis</h2>
      
      <Tabs
        activeKey={activeTab}
        onSelect={(k) => setActiveTab(k)}
        className="mb-4"
      >
        <Tab eventKey="detection" title="Image Detection">
          <Card className="mb-4">
            <Card.Body>
              <Form.Group className="mb-3">
                <Form.Label>Detection Type</Form.Label>
                <Form.Select 
                  value={detectionType}
                  onChange={(e) => setDetectionType(e.target.value)}
                >
                  <option value="all">All (Potholes + Cracks + Kerbs)</option>
                  <option value="potholes">Potholes</option>
                  <option value="cracks">Alligator Cracks</option>
                  <option value="kerbs">Kerbs</option>
                </Form.Select>
              </Form.Group>

                              {/* Sticky note reminder */}
                <OverlayTrigger 
                  trigger="click" 
                  placement="right" 
                  overlay={reminderPopover}
                  rootClose
                >
                  <div 
                    className="sticky-note-icon mb-3"
                    style={{ cursor: 'pointer', display: 'inline-block' }}
                  >
                    <img 
                      src="/remindericon.svg" 
                      alt="Image Upload Guidelines" 
                      style={{ width: '32px', height: '32px' }}
                    />
                  </div>
                </OverlayTrigger>

              <div className="mb-3">
                <Form.Label>Image Source</Form.Label>
                <div className="d-flex gap-2 mb-2">
                  <Button 
                    variant={cameraActive ? "primary" : "outline-primary"}
                    onClick={toggleCamera}
                    disabled={locationLoading}
                  >
                    {locationLoading ? (
                      <>
                        <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                        <span className="ms-2">Getting Location...</span>
                      </>
                    ) : (
                      cameraActive ? "Disable Camera" : "Enable Camera"
                    )}
                  </Button>
                  <div className="file-input-container">
                    <label className="file-input-label">
                      Upload Image
                      <input
                        type="file"
                        className="file-input"
                        accept="image/*"
                        onChange={handleFileChange}
                        ref={fileInputRef}
                        disabled={cameraActive}
                        multiple
                      />
                    </label>
                  </div>
                </div>
                
                {/* Location Status Display */}
                {cameraActive && (
                  <div className="location-status mb-3">
                    <small className="text-muted">
                      <strong>Location Status:</strong> 
                      {locationPermission === 'granted' && <span className="text-success ms-1">✓ Enabled</span>}
                      {locationPermission === 'denied' && <span className="text-danger ms-1">✗ Denied</span>}
                      {locationPermission === 'prompt' && <span className="text-warning ms-1">⚠ Requesting...</span>}
                      {locationPermission === 'unknown' && <span className="text-secondary ms-1">? Unknown</span>}
                    </small>
                    {(coordinates !== 'Not Available' || Object.keys(imagePreviewsMap).length > 0) && (
                      <div className="mt-1">
                        <small className="text-muted">
                          <strong>Current Location:</strong> <span className="text-primary">{coordinates}</span>
                        </small>
                        {Object.keys(imagePreviewsMap).length > 0 && (
                          <div className="mt-1">
                            <small className="text-muted">
                              <strong>Selected Image Location:</strong> <span className="text-primary">{getCurrentImageLocation()}</span>
                            </small>
                          </div>
                        )}
                      </div>
                    )}
                    {locationError && (
                      <Alert variant="warning" className="mt-2 mb-0" style={{ fontSize: '0.875rem' }}>
                        <Alert.Heading as="h6">Location Access Issue</Alert.Heading>
                        <div style={{ whiteSpace: 'pre-line' }}>{locationError}</div>
                        <hr />
                        <div className="d-flex justify-content-end">
                          <Button variant="outline-warning" size="sm" onClick={handleLocationRequest}>
                            Retry Location Access
                          </Button>
                        </div>
                      </Alert>
                    )}
                  </div>
                )}
              </div>

              {cameraActive && (
                <div className="webcam-container mb-3">
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    className="webcam"
                    videoConstraints={{
                      width: 640,
                      height: 480,
                      facingMode: cameraOrientation
                    }}
                  />
                  {isMobile && (
                    <Button 
                      variant="outline-secondary" 
                      onClick={toggleCameraOrientation}
                      className="mt-2 mb-2"
                      size="sm"
                    >
                      Rotate Camera
                    </Button>
                  )}
                  <Button 
                    variant="success" 
                    onClick={handleCapture}
                    className="mt-2"
                  >
                    Capture Photo
                  </Button>
                </div>
              )}

              {Object.keys(imagePreviewsMap).length > 0 && (
                <div className="image-preview-container mb-3">
                  <h5>Previews</h5>
                  <div className="image-gallery">
                    {Object.entries(imagePreviewsMap).map(([name, preview], index) => (
                      <div 
                        key={name} 
                        className={`image-thumbnail ${index === currentImageIndex ? 'selected' : ''}`}
                        onClick={() => setCurrentImageIndex(index)}
                      >
                        <img 
                          src={preview} 
                          alt={`Preview ${index + 1}`} 
                          className="img-thumbnail" 
                        />
                        <button 
                          className="btn btn-sm btn-danger remove-image" 
                          onClick={(e) => {
                            e.stopPropagation();
                            const newFiles = imageFiles.filter((_, i) => i !== index);
                            const newPreviewsMap = {...imagePreviewsMap};
                            const newLocationMap = {...imageLocationMap};
                            delete newPreviewsMap[name];
                            delete newLocationMap[name];
                            setImageFiles(newFiles);
                            setImagePreviewsMap(newPreviewsMap);
                            setImageLocationMap(newLocationMap);
                            if (currentImageIndex >= newFiles.length) {
                              setCurrentImageIndex(Math.max(0, newFiles.length - 1));
                            }
                          }}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                  <div className="current-image-preview">
                    {Object.values(imagePreviewsMap)[currentImageIndex] && (
                      <img 
                        src={Object.values(imagePreviewsMap)[currentImageIndex]} 
                        alt="Current Preview" 
                        className="image-preview img-fluid" 
                      />
                    )}
                  </div>
                </div>
              )}

              {error && <Alert variant="danger">{error}</Alert>}

              <div className="d-flex gap-2 mb-3">
                <Button 
                  variant="primary" 
                  onClick={handleProcess}
                  disabled={Object.keys(imagePreviewsMap).length === 0 || loading || batchProcessing}
                >
                  {loading ? (
                    <>
                      <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                      <span className="ms-2">Detecting...</span>
                    </>
                  ) : (
                    `Detect Current Image`
                  )}
                </Button>
                
                <Button 
                  variant="success" 
                  onClick={handleProcessAll}
                  disabled={Object.keys(imagePreviewsMap).length === 0 || loading || batchProcessing}
                >
                  {batchProcessing ? (
                    <>
                      <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                      <span className="ms-2">Processing {processedCount}/{totalToProcess}</span>
                    </>
                  ) : (
                    `Process All Images`
                  )}
                </Button>
                
                <Button 
                  variant="secondary" 
                  onClick={handleReset}
                  disabled={loading || batchProcessing}
                >
                  Reset
                </Button>
              </div>
            </Card.Body>
          </Card>

          {processedImage && (
            <Card className="mb-4">
              <Card.Body>
                <h4 className="mb-3">Detection Results</h4>
                
                {/* Location status in results */}
                <div className="mb-3">
                  {(() => {
                    const imageLocation = getCurrentImageLocation();
                    return imageLocation !== 'Not Available' && imageLocation !== 'Location Error' && imageLocation !== 'Permission Denied' ? (
                      <Alert variant="success" className="py-2">
                        <small>
                          <i className="fas fa-map-marker-alt me-2"></i>
                          <strong>Image Location Data:</strong> {imageLocation}
                        </small>
                      </Alert>
                    ) : (
                      <Alert variant="warning" className="py-2">
                        <small>
                          <i className="fas fa-exclamation-triangle me-2"></i>
                          <strong>Location Warning:</strong> Location data was not available for this image. 
                          {Object.keys(imagePreviewsMap).length > 0 && Object.keys(imagePreviewsMap)[currentImageIndex].includes('camera_capture') 
                            ? 'The camera may have been disabled before capturing, or location access was denied.'
                            : 'Uploaded images do not contain GPS data. Use the live camera for location-tagged captures.'}
                        </small>
                      </Alert>
                    );
                  })()}
                </div>
                
                <div className="processed-image-container mb-3">
                  <img 
                    src={processedImage} 
                    alt="Processed" 
                    className="processed-image img-fluid" 
                  />
                </div>

                {results && (
                  <div className="results-summary">
                    {detectionType === 'all' && (
                      <div className="detection-summary-card">
                        <h4 className="mb-4 text-center">🔍 All Defects Detection Results</h4>
                        
                        {/* Display any error messages for failed models */}
                        {results.model_errors && Object.keys(results.model_errors).length > 0 && (
                          <Alert variant="warning" className="mb-3 model-error-alert">
                            <Alert.Heading as="h6">⚠️ Partial Detection Results</Alert.Heading>
                            <p>Some detection models encountered errors:</p>
                            <ul className="mb-0">
                              {Object.entries(results.model_errors).map(([model, error]) => (
                                <li key={model}><strong>{model}:</strong> {error}</li>
                              ))}
                            </ul>
                          </Alert>
                        )}
                        
                        {/* Potholes Section */}
                        <div className="defect-section potholes">
                          <h5 className="text-danger">
                            <span className="emoji">🕳️</span>
                            Potholes Detected: {results.potholes ? results.potholes.length : 0}
                          </h5>
                          {results.potholes && results.potholes.length > 0 ? (
                            <div className="scrollable-table mb-3">
                              <table className="table table-striped table-bordered">
                                <thead>
                                  <tr>
                                    <th>ID</th>
                                    <th>Area (cm²)</th>
                                    <th>Depth (cm)</th>
                                    <th>Volume (cm³)</th>
                                    <th>Volume Range</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {results.potholes.map((pothole) => (
                                    <tr key={pothole.pothole_id}>
                                      <td>{pothole.pothole_id}</td>
                                      <td>{pothole.area_cm2.toFixed(2)}</td>
                                      <td>{pothole.depth_cm.toFixed(2)}</td>
                                      <td>{pothole.volume.toFixed(2)}</td>
                                      <td>{pothole.volume_range}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          ) : (
                            <div className="no-defects-message">No potholes detected in this image.</div>
                          )}
                        </div>

                        {/* Cracks Section */}
                        <div className="defect-section cracks">
                          <h5 className="text-success">
                            <span className="emoji">🪨</span>
                            Alligator Cracks Detected: {results.cracks ? results.cracks.length : 0}
                          </h5>
                          {results.cracks && results.cracks.length > 0 ? (
                            <>
                              <div className="scrollable-table mb-3">
                                <table className="table table-striped table-bordered">
                                  <thead>
                                    <tr>
                                      <th>ID</th>
                                      <th>Type</th>
                                      <th>Area (cm²)</th>
                                      <th>Area Range</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {results.cracks.map((crack) => (
                                      <tr key={crack.crack_id}>
                                        <td>{crack.crack_id}</td>
                                        <td>{crack.crack_type}</td>
                                        <td>{crack.area_cm2.toFixed(2)}</td>
                                        <td>{crack.area_range}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                              
                              {results.type_counts && (
                                <div>
                                  <h6>Crack Types Summary</h6>
                                  <ul className="crack-types-list">
                                    {Object.entries(results.type_counts).map(([type, count]) => (
                                      count > 0 && (
                                        <li key={type}>
                                          <strong>{type}:</strong> {count}
                                        </li>
                                      )
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </>
                          ) : (
                            <div className="no-defects-message">No cracks detected in this image.</div>
                          )}
                        </div>

                        {/* Kerbs Section */}
                        <div className="defect-section kerbs">
                          <h5 className="text-primary">
                            <span className="emoji">🚧</span>
                            Kerbs Detected: {results.kerbs ? results.kerbs.length : 0}
                          </h5>
                          {results.kerbs && results.kerbs.length > 0 ? (
                            <>
                              <div className="scrollable-table mb-3">
                                <table className="table table-striped table-bordered">
                                  <thead>
                                    <tr>
                                      <th>ID</th>
                                      <th>Type</th>
                                      <th>Condition</th>
                                      <th>Length</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {results.kerbs.map((kerb) => (
                                      <tr key={kerb.kerb_id}>
                                        <td>{kerb.kerb_id}</td>
                                        <td>{kerb.kerb_type}</td>
                                        <td>{kerb.condition}</td>
                                        <td>{kerb.length_m.toFixed(2)}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                              
                              {results.condition_counts && (
                                <div>
                                  <h6>Kerb Conditions Summary</h6>
                                  <ul className="kerb-types-list">
                                    {Object.entries(results.condition_counts).map(([condition, count]) => (
                                      count > 0 && (
                                        <li key={condition}>
                                          <strong>{condition}:</strong> {count}
                                        </li>
                                      )
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </>
                          ) : (
                            <div className="no-defects-message">No kerbs detected in this image.</div>
                          )}
                        </div>

                        {/* Overall Summary */}
                        <div className="summary-stats">
                          <h6 className="mb-3">📊 Detection Summary</h6>
                          <div className="row">
                            <div className="col-md-4 stat-item">
                              <div className="stat-value text-danger">{results.potholes ? results.potholes.length : 0}</div>
                              <div className="stat-label">Potholes</div>
                            </div>
                            <div className="col-md-4 stat-item">
                              <div className="stat-value text-success">{results.cracks ? results.cracks.length : 0}</div>
                              <div className="stat-label">Cracks</div>
                            </div>
                            <div className="col-md-4 stat-item">
                              <div className="stat-value text-primary">{results.kerbs ? results.kerbs.length : 0}</div>
                              <div className="stat-label">Kerbs</div>
                            </div>
                          </div>
                          <div className="text-center mt-3">
                            <span className="total-defects-badge">
                              Total Defects: {(results.potholes ? results.potholes.length : 0) + (results.cracks ? results.cracks.length : 0) + (results.kerbs ? results.kerbs.length : 0)}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {detectionType === 'potholes' && results.potholes && (
                      <div>
                        <h5>Detected Potholes: {results.potholes.length}</h5>
                        {results.potholes.length > 0 && (
                          <div className="scrollable-table mb-3">
                            <table className="table table-striped table-bordered">
                              <thead>
                                <tr>
                                  <th>ID</th>
                                  <th>Area (cm²)</th>
                                  <th>Depth (cm)</th>
                                  <th>Volume (cm³)</th>
                                  <th>Volume Range</th>
                                </tr>
                              </thead>
                              <tbody>
                                {results.potholes.map((pothole) => (
                                  <tr key={pothole.pothole_id}>
                                    <td>{pothole.pothole_id}</td>
                                    <td>{pothole.area_cm2.toFixed(2)}</td>
                                    <td>{pothole.depth_cm.toFixed(2)}</td>
                                    <td>{pothole.volume.toFixed(2)}</td>
                                    <td>{pothole.volume_range}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    )}

                    {detectionType === 'cracks' && results.cracks && (
                      <div>
                        <h5>Detected Cracks: {results.cracks.length}</h5>
                        {results.cracks.length > 0 && (
                          <div className="scrollable-table mb-3">
                            <table className="table table-striped table-bordered">
                              <thead>
                                <tr>
                                  <th>ID</th>
                                  <th>Type</th>
                                  <th>Area (cm²)</th>
                                  <th>Area Range</th>
                                </tr>
                              </thead>
                              <tbody>
                                {results.cracks.map((crack) => (
                                  <tr key={crack.crack_id}>
                                    <td>{crack.crack_id}</td>
                                    <td>{crack.crack_type}</td>
                                    <td>{crack.area_cm2.toFixed(2)}</td>
                                    <td>{crack.area_range}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        {results.type_counts && (
                          <div>
                            <h5>Crack Types Summary</h5>
                            <ul className="crack-types-list">
                              {Object.entries(results.type_counts).map(([type, count]) => (
                                count > 0 && (
                                  <li key={type}>
                                    <strong>{type}:</strong> {count}
                                  </li>
                                )
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}

                    {detectionType === 'kerbs' && results.kerbs && (
                      <div>
                        <h5>Detected Kerbs: {results.kerbs.length}</h5>
                        {results.kerbs.length > 0 && (
                          <div className="scrollable-table mb-3">
                            <table className="table table-striped table-bordered">
                              <thead>
                                <tr>
                                  <th>ID</th>
                                  <th>Type</th>
                                  <th>Condition</th>
                                  <th>Length</th>
                                </tr>
                              </thead>
                              <tbody>
                                {results.kerbs.map((kerb) => (
                                  <tr key={kerb.kerb_id}>
                                    <td>{kerb.kerb_id}</td>
                                    <td>{kerb.kerb_type}</td>
                                    <td>{kerb.condition}</td>
                                    <td>{kerb.length_m.toFixed(2)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        {results.condition_counts && (
                          <div>
                            <h5>Kerb Conditions Summary</h5>
                            <ul className="kerb-conditions-list">
                              {Object.entries(results.condition_counts).map(([condition, count]) => (
                                count > 0 && (
                                  <li key={condition}>
                                    <strong>{condition}:</strong> {count}
                                  </li>
                                )
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </Card.Body>
            </Card>
          )}

          {/* Batch processing status indicator */}
          {batchProcessing && (
            <div className="batch-processing-status mt-3">
              <div className="d-flex align-items-center">
                <div className="me-3">
                  <Spinner animation="border" size="sm" role="status" />
                </div>
                <div>
                  <h6 className="mb-1">Processing images: {processedCount}/{totalToProcess}</h6>
                  <div className="progress" style={{ height: '10px' }}>
                    <div 
                      className="progress-bar" 
                      role="progressbar" 
                      style={{ width: `${(processedCount / totalToProcess) * 100}%` }}
                      aria-valuenow={processedCount}
                      aria-valuemin="0" 
                      aria-valuemax={totalToProcess}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {!batchProcessing && batchResults.length > 0 && (
            <div className="batch-complete-status mt-3">
              <Alert variant="success">
                <i className="fas fa-check-circle me-2"></i>
                Processed {batchResults.length} images. 
                {batchResults.filter(r => r.success).length} successful, 
                {batchResults.filter(r => !r.success).length} failed.
              </Alert>
            </div>
          )}

          {/* Navigation buttons for processed images */}
          {batchResults.length > 1 && !batchProcessing && (
            <div className="image-navigation mt-3 d-flex justify-content-between">
              <Button 
                variant="outline-secondary" 
                size="sm"
                disabled={currentImageIndex === 0}
                onClick={() => {
                  if (currentImageIndex > 0) {
                    const newIndex = currentImageIndex - 1;
                    setCurrentImageIndex(newIndex);
                    
                    // Find the result for this image
                    const filename = Object.keys(imagePreviewsMap)[newIndex];
                    const result = batchResults.find(r => r.filename === filename);
                    
                    if (result && result.success) {
                      setProcessedImage(result.processedImage);
                      setResults(result.data);
                    } else {
                      setProcessedImage(null);
                      setResults(null);
                    }
                  }
                }}
              >
                <i className="fas fa-arrow-left me-1"></i> Previous Image
              </Button>
              
              <div className="image-counter">
                Image {currentImageIndex + 1} of {Object.keys(imagePreviewsMap).length}
              </div>
              
              <Button 
                variant="outline-secondary" 
                size="sm"
                disabled={currentImageIndex >= Object.keys(imagePreviewsMap).length - 1}
                onClick={() => {
                  if (currentImageIndex < Object.keys(imagePreviewsMap).length - 1) {
                    const newIndex = currentImageIndex + 1;
                    setCurrentImageIndex(newIndex);
                    
                    // Find the result for this image
                    const filename = Object.keys(imagePreviewsMap)[newIndex];
                    const result = batchResults.find(r => r.filename === filename);
                    
                    if (result && result.success) {
                      setProcessedImage(result.processedImage);
                      setResults(result.data);
                    } else {
                      setProcessedImage(null);
                      setResults(null);
                    }
                  }
                }}
              >
                Next Image <i className="fas fa-arrow-right ms-1"></i>
              </Button>
            </div>
          )}
        </Tab>
        
        <Tab eventKey="video" title="Video Detection">
          <VideoDefectDetection />
        </Tab>
        
        <Tab eventKey="information" title="Information">
          <Card>
            <Card.Body>
              <h4>About Pavement Analysis</h4>
              <p>
                The Pavement Analysis module uses advanced computer vision to detect and analyze 
                various types of pavement defects and features:
              </p>
              
              <h5>1. Potholes</h5>
              <p>
                Potholes are bowl-shaped holes of various sizes in the road surface that can be a 
                serious hazard to vehicles. The system detects potholes and calculates:
              </p>
              <ul>
                <li>Area in square centimeters</li>
                <li>Depth in centimeters</li>
                <li>Volume</li>
                <li>Classification by size (Small, Medium, Large)</li>
              </ul>
              
              <h5>2. Alligator Cracks</h5>
              <p>
                Alligator cracks are a series of interconnected cracks creating a pattern resembling 
                an alligator's scales. These indicate underlying structural weakness. The system 
                identifies multiple types of cracks including:
              </p>
              <ul>
                <li>Alligator Cracks</li>
                <li>Edge Cracks</li>
                <li>Hairline Cracks</li>
                <li>Longitudinal Cracks</li>
                <li>Transverse Cracks</li>
              </ul>
              
              <h5>3. Kerbs</h5>
              <p>
                Kerbs are raised edges along a street or path that define boundaries between roadways 
                and other areas. The system identifies different kerb conditions including:
              </p>
              <ul>
                <li>Normal/Good Kerbs - Structurally sound and properly visible</li>
                <li>Faded Kerbs - Reduced visibility due to worn paint or weathering</li>
                <li>Damaged Kerbs - Physically damaged or broken kerbs requiring repair</li>
              </ul>
              
              <h5>Location Services & GPS Data</h5>
              <p>
                When using the live camera option, the application can capture GPS coordinates 
                to provide precise geolocation data for detected defects. This helps in:
              </p>
              <ul>
                <li>Accurately mapping defect locations</li>
                <li>Creating location-based reports</li>
                <li>Enabling field teams to find specific issues</li>
                <li>Tracking defect patterns by geographic area</li>
              </ul>
              
              <h6>Location Requirements:</h6>
              <ul>
                <li><strong>Secure Connection:</strong> Location services require HTTPS</li>
                <li><strong>Browser Permissions:</strong> You must allow location access when prompted</li>
                <li><strong>Safari Users:</strong> Enable location services in Safari settings</li>
                <li><strong>Mobile Devices:</strong> Ensure location services are enabled in device settings</li>
              </ul>
              
              <div className="alert alert-info">
                <h6><i className="fas fa-info-circle me-2"></i>Troubleshooting Location Issues</h6>
                <p><strong>If location access is denied:</strong></p>
                <ul className="mb-2">
                  <li><strong>Safari:</strong> Settings → Privacy & Security → Location Services</li>
                  <li><strong>Chrome:</strong> Settings → Privacy and security → Site Settings → Location</li>
                  <li><strong>Firefox:</strong> Settings → Privacy & Security → Permissions → Location</li>
                </ul>
                <p><strong>On mobile devices:</strong> Also check your device's location settings and ensure the browser has location permission.</p>
              </div>

              <h5>How to Use This Module</h5>
              <ol>
                <li>Select the detection type (Potholes, Alligator Cracks, or Kerbs)</li>
                <li>Upload an image or use the camera to capture a photo</li>
                <li>If using the camera, allow location access when prompted for GPS coordinates</li>
                <li>Click the Detect button to analyze the image</li>
                <li>Review the detection results and measurements</li>
              </ol>
              
              <p>
                The detected defects are automatically recorded in the database for tracking 
                and analysis in the Dashboard module.
              </p>
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>
    </Container>
  );
};

export default Pavement; 
 