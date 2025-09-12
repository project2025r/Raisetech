
import React, { useState, useRef, useEffect } from 'react';
import { Container, Card, Button, Form, Tabs, Tab, Alert, Spinner, OverlayTrigger, Popover, Modal } from 'react-bootstrap';
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
  
  // Add state for storing processed images for results table
  const [processedImagesData, setProcessedImagesData] = useState({});

  // Add state for classification error modal
  const [showClassificationModal, setShowClassificationModal] = useState(false);
  const [classificationError, setClassificationError] = useState('');
  const [totalToProcess, setTotalToProcess] = useState(0);
  
  // Add state for image modal
  const [showImageModal, setShowImageModal] = useState(false);
  const [selectedImageData, setSelectedImageData] = useState(null);

  // Add state for image status table filtering
  const [imageFilter, setImageFilter] = useState('all'); // 'all', 'road', 'non-road'



  // Add state for road classification toggle (default to false for better user experience)
  const [roadClassificationEnabled, setRoadClassificationEnabled] = useState(false);

  // Add state for enhanced detection results table
  const [detectionTableFilter, setDetectionTableFilter] = useState('all'); // 'all', 'potholes', 'cracks', 'kerbs'
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

  // Auto-clear is always enabled - no toggle needed
  
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
        
        // Show AVIF conversion notice if applicable
        if (file.name.toLowerCase().endsWith('.avif')) {
          console.log(`AVIF file detected: ${file.name} - will be automatically converted to JPG during processing`);
        }
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

  // Helper function to handle classification errors
  const handleClassificationError = (errorMessage) => {
    setClassificationError(errorMessage);
    setShowClassificationModal(true);
    setError(''); // Clear general error since we're showing specific modal
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

      // Get the current image filename
      const filenames = Object.keys(imagePreviewsMap);
      const currentFilename = filenames[currentImageIndex];

      // Prepare request data
      const requestData = {
        image: currentImagePreview,
        coordinates: imageCoordinates,
        username: user?.username || 'Unknown',
        role: user?.role || 'Unknown',
        skip_road_classification: !roadClassificationEnabled
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
        // Check if the image was actually processed (contains road) or just classified
        const isProcessed = response.data.processed !== false;
        const isRoad = response.data.classification?.is_road || false;

        // Set the processed image and results for display
        setProcessedImage(response.data.processed_image);
        setResults(response.data);

        // Extract detailed detection results for table display
        const detectionResults = {
          potholes: response.data.potholes || [],
          cracks: response.data.cracks || [],
          kerbs: response.data.kerbs || []
        };

        // Create batch result entry for the status table
        const batchResult = {
          filename: currentFilename,
          success: true,
          processed: isProcessed,
          isRoad: isRoad,
          classification: response.data.classification,
          processedImage: response.data.processed_image,
          originalImage: currentImagePreview, // Add original image data
          data: response.data,
          detectionResults: detectionResults,
          wasClassificationEnabled: roadClassificationEnabled, // ADD THIS LINE
          detectionCounts: {
            potholes: detectionResults.potholes.length,
            cracks: detectionResults.cracks.length,
            kerbs: detectionResults.kerbs.length,
            total: detectionResults.potholes.length + detectionResults.cracks.length + detectionResults.kerbs.length
          }
        };

        // Update batch results to show the status table
        setBatchResults([batchResult]);

        // Auto-clear uploaded image icons after successful single image processing
        // Store the processed image data before clearing (for both road and non-road)
        setProcessedImagesData(prev => ({
          ...prev,
          [currentFilename]: {
            originalImage: currentImagePreview,
            processedImage: isRoad ? response.data.processed_image : null,
            results: response.data,
            isRoad: isRoad
          }
        }));
          
          // Clear image previews and files but keep results
          setImageFiles([]);
          setImagePreviewsMap({});
          setImageLocationMap({});
          setCurrentImageIndex(0);
          
          // Reset coordinates when clearing all images
          setCoordinates('Not Available');
          setLocationError('');
          setLocationPermission('unknown');
          
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
      } else {
        const errorMessage = response.data.message || 'Detection failed';

        // Create batch result entry for failed processing
        const batchResult = {
          filename: currentFilename,
          success: false,
          processed: false,
          isRoad: false,
          error: errorMessage,
          isClassificationError: errorMessage.includes('No road detected')
        };

        // Update batch results to show the status table
        setBatchResults([batchResult]);

        setError(errorMessage);
      }
    } catch (error) {
      const errorMessage = error.response?.data?.message || 'An error occurred during detection. Please try again.';

      // Get the current image filename for batch results
      const filenames = Object.keys(imagePreviewsMap);
      const currentFilename = filenames[currentImageIndex];

      // Create batch result entry for error case
      const batchResult = {
        filename: currentFilename,
        success: false,
        processed: false,
        isRoad: false,
        error: errorMessage,
        isClassificationError: errorMessage.includes('No road detected')
      };

      // Update batch results to show the status table
      setBatchResults([batchResult]);

      // Check if this is a classification error (no road detected)
      if (errorMessage.includes('No road detected')) {
        handleClassificationError(errorMessage);
      } else {
        setError(errorMessage);
      }
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
            role: user?.role || 'Unknown',
            skip_road_classification: !roadClassificationEnabled
          };
          
          // Make API request
          const response = await axios.post(endpoint, requestData);
          
          if (response.data.success) {
            // Check if the image was actually processed (contains road) or just classified
            const isProcessed = response.data.processed !== false;
            const isRoad = response.data.classification?.is_road || false;

            if (isProcessed && isRoad) {
              // Road image that was processed - display the results
              setProcessedImage(response.data.processed_image);
              setResults(response.data);
            }

            // Extract detailed detection results for table display
            const detectionResults = {
              potholes: response.data.potholes || [],
              cracks: response.data.cracks || [],
              kerbs: response.data.kerbs || []
            };

            results.push({
              filename,
              success: true,
              processed: isProcessed,
              isRoad: isRoad,
              classification: response.data.classification,
              processedImage: response.data.processed_image,
              originalImage: imageData, // Add original image data
              data: response.data,
              detectionResults: detectionResults,
              wasClassificationEnabled: roadClassificationEnabled, // ADD THIS LINE
              detectionCounts: {
                potholes: detectionResults.potholes.length,
                cracks: detectionResults.cracks.length,
                kerbs: detectionResults.kerbs.length,
                total: detectionResults.potholes.length + detectionResults.cracks.length + detectionResults.kerbs.length
              }
            });
          } else {
            const errorMessage = response.data.message || 'Detection failed';
            results.push({
              filename,
              success: false,
              processed: false,
              isRoad: false,
              error: errorMessage,
              isClassificationError: errorMessage.includes('No road detected')
            });
          }
        } catch (error) {
          const errorMessage = error.response?.data?.message || 'An error occurred during detection';
          results.push({
            filename,
            success: false,
            processed: false,
            isRoad: false,
            error: errorMessage,
            isClassificationError: errorMessage.includes('No road detected')
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
      
      // Store final results
      setBatchResults(results);

      // After batch processing is complete, display the first successfully processed road image
      const processedRoadImages = results.filter(r => r.success && r.processed && r.isRoad);
      if (processedRoadImages.length > 0) {
        const firstProcessedRoadImage = processedRoadImages[0];
        setProcessedImage(firstProcessedRoadImage.processedImage);
        setResults(firstProcessedRoadImage.data);

        // Set the current image index to 0 (first processed road image)
        setCurrentImageIndex(0);
      } else {
        // No road images were processed, clear the display
        setProcessedImage(null);
        setResults(null);
        setCurrentImageIndex(0);
      }

      // Auto-clear uploaded image icons after processing is complete
      // Store processed images data before clearing (for both road and non-road)
      const processedData = {};
      results.forEach(result => {
        if (result.success) {
          const originalImage = imagePreviewsMap[result.filename];
          processedData[result.filename] = {
            originalImage: originalImage,
            processedImage: result.isRoad ? result.processedImage : null,
            results: result.data,
            isRoad: result.isRoad
          };
          console.log('Storing image data for:', result.filename, 'isRoad:', result.isRoad, 'hasOriginalImage:', !!originalImage);
        }
      });
      setProcessedImagesData(prev => ({ ...prev, ...processedData }));
      
      // Clear image previews and files but keep results
      setImageFiles([]);
      setImagePreviewsMap({});
      setImageLocationMap({});
      setCurrentImageIndex(0);
      
      // Reset coordinates when clearing all images
      setCoordinates('Not Available');
      setLocationError('');
      setLocationPermission('unknown');
      
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

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
    setProcessedImagesData({});
    
    // Reset coordinates when clearing all images
    setCoordinates('Not Available');
    setLocationError('');
    setLocationPermission('unknown');
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };





  // Add function to handle thumbnail clicks
  const handleThumbnailClick = (imageData) => {
    setSelectedImageData(imageData);
    setShowImageModal(true);
  };

  // Add sorting function for detection results
  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  // Function to sort detection results
  const sortDetections = (detections) => {
    if (!sortConfig.key) return detections;

    return [...detections].sort((a, b) => {
      let aValue = a[sortConfig.key];
      let bValue = b[sortConfig.key];

      // Handle numeric values
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortConfig.direction === 'asc' ? aValue - bValue : bValue - aValue;
      }

      // Handle string values
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortConfig.direction === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      // Handle null/undefined values
      if (aValue == null && bValue == null) return 0;
      if (aValue == null) return sortConfig.direction === 'asc' ? 1 : -1;
      if (bValue == null) return sortConfig.direction === 'asc' ? -1 : 1;

      return 0;
    });
  };

  // Function to export detection results to CSV
  const exportToCSV = () => {
    // Flatten all detection results
    const allDetections = [];

    batchResults.forEach(result => {
      if (result.success && result.processed && result.detectionResults) {
        const { potholes, cracks, kerbs } = result.detectionResults;

        // Add potholes
        potholes.forEach(pothole => {
          allDetections.push({
            filename: result.filename,
            type: 'Pothole',
            id: pothole.pothole_id,
            area_cm2: pothole.area_cm2,
            depth_cm: pothole.depth_cm,
            volume: pothole.volume,
            volume_range: pothole.volume_range,
            crack_type: '',
            area_range: '',
            kerb_type: '',
            condition: '',
            length_m: '',
            confidence: pothole.confidence
          });
        });

        // Add cracks
        cracks.forEach(crack => {
          allDetections.push({
            filename: result.filename,
            type: 'Crack',
            id: crack.crack_id,
            area_cm2: crack.area_cm2,
            depth_cm: '',
            volume: '',
            volume_range: '',
            crack_type: crack.crack_type,
            area_range: crack.area_range,
            kerb_type: '',
            condition: '',
            length_m: '',
            confidence: crack.confidence
          });
        });

        // Add kerbs
        kerbs.forEach(kerb => {
          allDetections.push({
            filename: result.filename,
            type: 'Kerb',
            id: kerb.kerb_id,
            area_cm2: '',
            depth_cm: '',
            volume: '',
            volume_range: '',
            crack_type: '',
            area_range: '',
            kerb_type: kerb.kerb_type,
            condition: kerb.condition,
            length_m: kerb.length_m,
            confidence: kerb.confidence
          });
        });
      }
    });

    if (allDetections.length === 0) {
      alert('No detection results to export.');
      return;
    }

    // Create CSV content
    const headers = [
      'Image Filename',
      'Detection Type',
      'ID',
      'Area (cm²)',
      'Depth (cm)',
      'Volume (cm³)',
      'Volume Range',
      'Crack Type',
      'Area Range',
      'Kerb Type',
      'Condition',
      'Length (m)',
      'Confidence'
    ];

    const csvContent = [
      headers.join(','),
      ...allDetections.map(detection => [
        detection.filename,
        detection.type,
        detection.id || '',
        detection.area_cm2 || '',
        detection.depth_cm || '',
        detection.volume || '',
        detection.volume_range || '',
        detection.crack_type || '',
        detection.area_range || '',
        detection.kerb_type || '',
        detection.condition || '',
        detection.length_m || '',
        detection.confidence || ''
      ].join(','))
    ].join('\n');

    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `pavement_detection_results_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };


// Clear results when road classification toggle is changed
  useEffect(() => {
    if (batchResults.length > 0) {
      // Clear batch results when toggle changes to prevent showing outdated classification data
      setBatchResults([]);
      setProcessedImage(null);
      setResults(null);
    }
  }, [roadClassificationEnabled]);


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



  return (
    <Container className="pavement-page">
      
      <Tabs
        activeKey={activeTab}
        onSelect={(k) => setActiveTab(k)}
        className="mb-3"
      >
        <Tab eventKey="detection" title="Image Detection">
          <Card className="mb-3">
            <Card.Body className="py-3">
              <Form.Group className="mb-3">
                <Form.Label className="mb-1">Detection Type</Form.Label>
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

              {/* Sticky note reminder and road classification toggle */}
              <div className="d-flex align-items-start gap-2 mb-3">
                <OverlayTrigger
                  trigger="click"
                  placement="right"
                  overlay={reminderPopover}
                  rootClose
                >
                  <div
                    className="sticky-note-icon"
                    style={{ cursor: 'pointer', display: 'inline-block' }}
                  >
                    <img
                      src="/remindericon.svg"
                      alt="Image Upload Guidelines"
                      style={{ width: '28px', height: '28px' }}
                    />
                  </div>
                </OverlayTrigger>

                {/* Road Classification Toggle - Improved Design */}
                <div className="road-classification-control">
                  <div className="d-flex align-items-center justify-content-between mb-1">
                    <span className="me-2" style={{ fontSize: '0.9rem', fontWeight: '500', color: '#495057' }}>
                      Road Classification
                    </span>
                      <OverlayTrigger
                        placement="right"
                        delay={{ show: 200, hide: 100 }}
                        overlay={
                          <Popover id="road-classification-detailed-info" style={{ maxWidth: '350px' }}>
                            <Popover.Header as="h3">
                              <i className="fas fa-brain me-2 text-primary"></i>
                              Road Classification Feature
                            </Popover.Header>
                            <Popover.Body>
                              <div className="mb-2">
                                <div className="mb-1">
                                  <i className="fas fa-toggle-on text-success me-2"></i>
                                  <strong>ENABLED (ON):</strong>
                                </div>
                                <div style={{ fontSize: '12px', color: '#6c757d', marginLeft: '20px' }}>
                                  • AI analyzes images for road content first<br/>
                                  • Only road images get defect detection<br/>
                                  • More accurate results, slightly slower
                                </div>
                              </div>

                              <div className="mb-2">
                                <div className="mb-1">
                                  <i className="fas fa-toggle-off text-secondary me-2"></i>
                                  <strong>DISABLED (OFF):</strong>
                                </div>
                                <div style={{ fontSize: '12px', color: '#6c757d', marginLeft: '20px' }}>
                                  • All images processed directly<br/>
                                  • No road verification step<br/>
                                  • Faster processing, may have false positives
                                </div>
                              </div>

                              <div className="alert alert-info py-2 px-2 mb-0" style={{ fontSize: '11px' }}>
                                <i className="fas fa-lightbulb me-1"></i>
                                <strong>Recommendation:</strong> Keep enabled for mixed image types.
                                Disable only when all images contain roads and speed is priority.
                              </div>
                            </Popover.Body>
                          </Popover>
                        }
                      >
                        <span className="info-icon-wrapper">
                          <span className="road-classification-info-icon"
                             style={{
                               fontSize: '14px',
                               cursor: 'help',
                               color: '#007bff',
                               display: 'inline-flex',
                               alignItems: 'center',
                               justifyContent: 'center',
                               position: 'relative',
                               zIndex: '1000',
                               fontWeight: 'bold'
                             }}
                          >i</span>
                        </span>
                      </OverlayTrigger>
                    </div>
                    <div className="d-flex align-items-center">
                      <div
                        className="toggle-switch me-2"
                        onClick={() => setRoadClassificationEnabled(!roadClassificationEnabled)}
                        style={{
                          width: '60px',
                          height: '30px',
                          backgroundColor: roadClassificationEnabled ? '#28a745' : '#6c757d',
                          borderRadius: '15px',
                          position: 'relative',
                          cursor: 'pointer',
                          transition: 'background-color 0.3s ease',
                          border: '2px solid transparent'
                        }}
                      >
                        <div
                          className="toggle-slider"
                          style={{
                            width: '22px',
                            height: '22px',
                            backgroundColor: 'white',
                            borderRadius: '50%',
                            position: 'absolute',
                            top: '2px',
                            left: roadClassificationEnabled ? '34px' : '2px',
                            transition: 'left 0.3s ease',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                          }}
                        />
                        <span
                          style={{
                            position: 'absolute',
                            top: '50%',
                            left: roadClassificationEnabled ? '8px' : '32px',
                            transform: 'translateY(-50%)',
                            fontSize: '10px',
                            fontWeight: '600',
                            color: 'white',
                            transition: 'all 0.3s ease',
                            userSelect: 'none'
                          }}
                        >
                          {roadClassificationEnabled ? 'ON' : 'OFF'}
                        </span>
                      </div>
                      <small className="text-muted" style={{ fontSize: '11px' }}>
                        {roadClassificationEnabled ? "Only road images processed" : "All images processed"}
                      </small>
                    </div>
                  </div>
                  

                </div>

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
                          accept="image/*,.avif"
                          onChange={handleFileChange}
                          ref={fileInputRef}
                          disabled={cameraActive}
                          multiple
                        />
                      </label>
                      <small className="text-muted d-block mt-1">
                        <i className="fas fa-info-circle me-1"></i>
                        Supports all image formats including AVIF (automatically converted to JPG for processing)
                      </small>
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
                        {/* AVIF conversion indicator */}
                        {name.toLowerCase().endsWith('.avif') && (
                          <div className="avif-indicator" title="AVIF file - will be converted to JPG during processing">
                            <i className="fas fa-sync-alt text-info"></i>
                            <small className="ms-1">AVIF</small>
                          </div>
                        )}
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


          {/* Enhanced Detection Results Table */}
          {batchResults.some(result => result.success && result.processed && result.detectionCounts?.total > 0) && (
            <Card className="mb-4">
              <Card.Header>
                <div className="d-flex justify-content-between align-items-center">
                  <h5 className="mb-0">
                    <i className="fas fa-table me-2"></i>
                    Detailed Detection Results
                  </h5>
                  <div className="d-flex gap-2">
                    <Button
                      variant="success"
                      size="sm"
                      onClick={exportToCSV}
                      title="Export results to CSV"
                    >
                      <i className="fas fa-download me-1"></i>
                      Export CSV
                    </Button>
                  </div>
                </div>
              </Card.Header>
              <Card.Body>
                  {/* Filter Controls */}
                  <div className="mb-3">
                    <div className="d-flex gap-2 flex-wrap">
                      <Button
                        variant={detectionTableFilter === 'all' ? 'primary' : 'outline-primary'}
                        size="sm"
                        onClick={() => setDetectionTableFilter('all')}
                      >
                        All Detections
                      </Button>
                      <Button
                        variant={detectionTableFilter === 'potholes' ? 'danger' : 'outline-danger'}
                        size="sm"
                        onClick={() => setDetectionTableFilter('potholes')}
                      >
                        Potholes Only
                      </Button>
                      <Button
                        variant={detectionTableFilter === 'cracks' ? 'warning' : 'outline-warning'}
                        size="sm"
                        onClick={() => setDetectionTableFilter('cracks')}
                      >
                        Cracks Only
                      </Button>
                      <Button
                        variant={detectionTableFilter === 'kerbs' ? 'info' : 'outline-info'}
                        size="sm"
                        onClick={() => setDetectionTableFilter('kerbs')}
                      >
                        Kerbs Only
                      </Button>
                    </div>
                  </div>

                  {/* Summary Statistics */}
                  <div className="mb-4 detection-summary-cards">
                    <div className="row">
                      <div className="col-md-3">
                        <div className="card bg-danger text-white">
                          <div className="card-body text-center py-2">
                            <h6 className="mb-1">Total Potholes</h6>
                            <h4 className="mb-0">
                              {batchResults.reduce((sum, result) => sum + (result.detectionCounts?.potholes || 0), 0)}
                            </h4>
                          </div>
                        </div>
                      </div>
                      <div className="col-md-3">
                        <div className="card bg-warning text-white">
                          <div className="card-body text-center py-2">
                            <h6 className="mb-1">Total Cracks</h6>
                            <h4 className="mb-0">
                              {batchResults.reduce((sum, result) => sum + (result.detectionCounts?.cracks || 0), 0)}
                            </h4>
                          </div>
                        </div>
                      </div>
                      <div className="col-md-3">
                        <div className="card bg-info text-white">
                          <div className="card-body text-center py-2">
                            <h6 className="mb-1">Total Kerbs</h6>
                            <h4 className="mb-0">
                              {batchResults.reduce((sum, result) => sum + (result.detectionCounts?.kerbs || 0), 0)}
                            </h4>
                          </div>
                        </div>
                      </div>
                      <div className="col-md-3">
                        <div className="card bg-success text-white">
                          <div className="card-body text-center py-2">
                            <h6 className="mb-1">Total Detections</h6>
                            <h4 className="mb-0">
                              {batchResults.reduce((sum, result) => sum + (result.detectionCounts?.total || 0), 0)}
                            </h4>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Detailed Results Table */}
                  <div className="table-responsive detection-table-container">
                    {(() => {
                      // Flatten all detection results into a single array
                      const allDetections = [];

                      batchResults.forEach(result => {
                        if (result.success && result.processed && result.detectionResults) {
                          const { potholes, cracks, kerbs } = result.detectionResults;

                          // Add potholes
                          if (detectionTableFilter === 'all' || detectionTableFilter === 'potholes') {
                            potholes.forEach(pothole => {
                              allDetections.push({
                                ...pothole,
                                type: 'Pothole',
                                filename: result.filename,
                                detectionType: 'potholes'
                              });
                            });
                          }

                          // Add cracks
                          if (detectionTableFilter === 'all' || detectionTableFilter === 'cracks') {
                            cracks.forEach(crack => {
                              allDetections.push({
                                ...crack,
                                type: 'Crack',
                                filename: result.filename,
                                detectionType: 'cracks'
                              });
                            });
                          }

                          // Add kerbs
                          if (detectionTableFilter === 'all' || detectionTableFilter === 'kerbs') {
                            kerbs.forEach(kerb => {
                              allDetections.push({
                                ...kerb,
                                type: 'Kerb',
                                filename: result.filename,
                                detectionType: 'kerbs'
                              });
                            });
                          }
                        }
                      });

                      if (allDetections.length === 0) {
                        return (
                          <div className="text-center py-5">
                            <div className="mb-3">
                              <i className="fas fa-search fa-3x text-muted"></i>
                            </div>
                            <h6 className="text-muted">No detections found</h6>
                            <p className="text-muted mb-0">
                              {detectionTableFilter === 'all'
                                ? 'No defects were detected in the processed images.'
                                : `No ${detectionTableFilter} were detected in the processed images.`}
                            </p>
                          </div>
                        );
                      }

                      return (
                        <table className="table table-striped table-bordered">
                          <thead>
                            <tr>
                              <th>Original Image</th>
                              <th>Processed Image</th>
                              <th
                                style={{ cursor: 'pointer' }}
                                onClick={() => handleSort('detectionType')}
                              >
                                Type {sortConfig.key === 'detectionType' && (
                                  <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                )}
                              </th>
                              <th>ID</th>
                              {(detectionTableFilter === 'all' || detectionTableFilter === 'potholes') && (
                                <>
                                  <th
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleSort('area_cm2')}
                                  >
                                    Area (cm²) {sortConfig.key === 'area_cm2' && (
                                      <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                    )}
                                  </th>
                                  <th
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleSort('depth_cm')}
                                  >
                                    Depth (cm) {sortConfig.key === 'depth_cm' && (
                                      <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                    )}
                                  </th>
                                  <th
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleSort('volume')}
                                  >
                                    Volume (cm³) {sortConfig.key === 'volume' && (
                                      <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                    )}
                                  </th>
                                  <th>Volume Range</th>
                                </>
                              )}
                              {(detectionTableFilter === 'all' || detectionTableFilter === 'cracks') && (
                                <>
                                  <th
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleSort('crack_type')}
                                  >
                                    Crack Type {sortConfig.key === 'crack_type' && (
                                      <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                    )}
                                  </th>
                                  <th
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleSort('area_cm2')}
                                  >
                                    Area (cm²) {sortConfig.key === 'area_cm2' && (
                                      <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                    )}
                                  </th>
                                  <th>Area Range</th>
                                </>
                              )}
                              {(detectionTableFilter === 'all' || detectionTableFilter === 'kerbs') && (
                                <>
                                  <th
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleSort('kerb_type')}
                                  >
                                    Kerb Type {sortConfig.key === 'kerb_type' && (
                                      <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                    )}
                                  </th>
                                  <th
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleSort('condition')}
                                  >
                                    Condition {sortConfig.key === 'condition' && (
                                      <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                    )}
                                  </th>
                                  <th
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleSort('length_m')}
                                  >
                                    Length (m) {sortConfig.key === 'length_m' && (
                                      <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                    )}
                                  </th>
                                </>
                              )}
                              <th
                                style={{ cursor: 'pointer' }}
                                onClick={() => handleSort('confidence')}
                              >
                                Confidence {sortConfig.key === 'confidence' && (
                                  <i className={`fas fa-sort-${sortConfig.direction === 'asc' ? 'up' : 'down'} ms-1`}></i>
                                )}
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {sortDetections(allDetections).map((detection, index) => (
                              <tr key={`${detection.filename}-${detection.detectionType}-${index}`}>
                                {/* Original Image Column */}
                                <td style={{ width: '120px', textAlign: 'center' }}>
                                  {(() => {
                                    const result = batchResults.find(r => r.filename === detection.filename);
                                    const originalImage = result?.originalImage;
                                    return originalImage ? (
                                      <div
                                        className="image-thumbnail-container"
                                        onClick={() => handleThumbnailClick({
                                          originalImage: originalImage,
                                          processedImage: result?.processedImage,
                                          isRoad: result?.isRoad,
                                          filename: detection.filename
                                        })}
                                        title="Click to enlarge 🔍"
                                      >
                                        <img
                                          src={originalImage}
                                          alt="Original"
                                          className="image-thumbnail"
                                        />
                                        <div className="thumbnail-overlay">
                                          🔍
                                        </div>
                                      </div>
                                    ) : (
                                      <small className="text-muted">No image</small>
                                    );
                                  })()}
                                </td>

                                {/* Processed Image Column */}
                                <td style={{ width: '120px', textAlign: 'center' }}>
                                  {(() => {
                                    const result = batchResults.find(r => r.filename === detection.filename);
                                    const processedImage = result?.processedImage;
                                    const originalImage = result?.originalImage;

                                    return processedImage && result?.isRoad ? (
                                      <div
                                        className="image-thumbnail-container"
                                        onClick={() => handleThumbnailClick({
                                          originalImage: originalImage,
                                          processedImage: processedImage,
                                          isRoad: result.isRoad,
                                          filename: detection.filename
                                        })}
                                        title="Click to enlarge 🔍"
                                      >
                                        <img
                                          src={processedImage}
                                          alt="Processed"
                                          className="image-thumbnail"
                                        />
                                        <div className="thumbnail-overlay">
                                          🔍
                                        </div>
                                      </div>
                                    ) : (
                                      <small className="text-muted">No processed image</small>
                                    );
                                  })()}
                                </td>
                                <td>
                                  <span className={`badge ${
                                    detection.detectionType === 'potholes' ? 'bg-danger' :
                                    detection.detectionType === 'cracks' ? 'bg-warning' : 'bg-info'
                                  }`}>
                                    {detection.type}
                                  </span>
                                </td>
                                <td>{detection.pothole_id || detection.crack_id || detection.kerb_id || index + 1}</td>

                                {/* Pothole-specific columns */}
                                {(detectionTableFilter === 'all' || detectionTableFilter === 'potholes') && (
                                  <>
                                    <td>{detection.area_cm2 ? detection.area_cm2.toFixed(2) : 'N/A'}</td>
                                    <td>{detection.depth_cm ? detection.depth_cm.toFixed(2) : 'N/A'}</td>
                                    <td>{detection.volume ? detection.volume.toFixed(2) : 'N/A'}</td>
                                    <td>{detection.volume_range || 'N/A'}</td>
                                  </>
                                )}

                                {/* Crack-specific columns */}
                                {(detectionTableFilter === 'all' || detectionTableFilter === 'cracks') && (
                                  <>
                                    <td>{detection.crack_type || 'N/A'}</td>
                                    <td>{detection.area_cm2 ? detection.area_cm2.toFixed(2) : 'N/A'}</td>
                                    <td>{detection.area_range || 'N/A'}</td>
                                  </>
                                )}

                                {/* Kerb-specific columns */}
                                {(detectionTableFilter === 'all' || detectionTableFilter === 'kerbs') && (
                                  <>
                                    <td>{detection.kerb_type || 'N/A'}</td>
                                    <td>{detection.condition || 'N/A'}</td>
                                    <td>{detection.length_m ? detection.length_m.toFixed(2) : 'N/A'}</td>
                                  </>
                                )}

                                <td>{detection.confidence ? (detection.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      );
                    })()}
                  </div>
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





          {/* Batch Processing Summary */}
          {!batchProcessing && batchResults.length > 0 && (() => {
            const totalImages = batchResults.length;
            const successfulImages = batchResults.filter(r => r.success).length;
            const failedImages = batchResults.filter(r => !r.success).length;

            let alertVariant = 'light';
            let alertClass = '';

            if (batchResults.some(result => result.wasClassificationEnabled)) {
              // When classification was enabled during processing, use road/non-road logic
              // When classification is enabled, use road/non-road logic
              const nonRoadImages = batchResults.filter(r => !r.isRoad).length;
              const nonRoadPercentage = totalImages > 0 ? (nonRoadImages / totalImages) * 100 : 0;

              if (totalImages > 0) {
                if (nonRoadPercentage === 0) {
                  // 100% road detection - Green
                  alertVariant = 'success';
                } else if (nonRoadPercentage === 100) {
                  // 100% non-road detection - Red
                  alertVariant = 'danger';
                } else {
                  // Combined detection (mixed results) - Light Orange
                  alertVariant = 'warning';
                  alertClass = 'summary-light-orange';
                }
              }
            } else {
              // When classification is disabled, use success/failure logic
              if (failedImages === 0) {
                // All successful - Green
                alertVariant = 'success';
              } else if (successfulImages === 0) {
                // All failed - Red
                alertVariant = 'danger';
              } else {
                // Mixed results - Warning
                alertVariant = 'warning';
              }
            }

            return (
              <div className="batch-complete-status mt-4">
                <Alert variant={alertVariant} className={alertClass}>
                  <i className="fas fa-check-circle me-2"></i>
                  Processed {batchResults.length} images.
                  {roadClassificationEnabled ? (
                    <>
                      {batchResults.filter(r => r.success && r.processed).length} road images processed,
                      {batchResults.filter(r => r.success && !r.processed).length} non-road images detected,
                      {batchResults.filter(r => !r.success).length} failed.
                    </>
                  ) : (
                    <>
                      {batchResults.filter(r => r.success).length} images processed successfully,
                      {batchResults.filter(r => !r.success).length} failed.
                    </>
                  )}
                </Alert>
              </div>
            );
          })()}

          {/* Image Status Table - Only show when road classification is enabled */}
          {/* Image Status Table - Only show when road classification was enabled during processing */}
          {!batchProcessing && batchResults.length > 0 && batchResults.some(result => result.wasClassificationEnabled) && (
            <div className="image-status-table mt-4">
              <Card>
                <Card.Header>
                  <div className="d-flex justify-content-between align-items-center">
                    <h5 className="mb-0">Image Processing Status</h5>
                    <div className="filter-buttons">
                      <Button
                        variant={imageFilter === 'all' ? 'primary' : 'outline-primary'}
                        size="sm"
                        className="me-2"
                        onClick={() => setImageFilter('all')}
                      >
                        Show All Images
                      </Button>
                      <Button
                        variant={imageFilter === 'road' ? 'success' : 'outline-success'}
                        size="sm"
                        className="me-2"
                        onClick={() => setImageFilter('road')}
                      >
                        Show Only Road Images
                      </Button>
                      <Button
                        variant={imageFilter === 'non-road' ? 'danger' : 'outline-danger'}
                        size="sm"
                        onClick={() => setImageFilter('non-road')}
                      >
                        Show Only Non-Road Images
                      </Button>
                    </div>
                  </div>
                </Card.Header>
                <Card.Body>
                  <div className="table-responsive">
                    <table className="table table-striped">
                      <thead>
                        <tr>
                          <th>Image</th>
                          <th>Detection Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults
                          .filter(result => {
                            if (imageFilter === 'road') return result.isRoad;
                            if (imageFilter === 'non-road') return !result.isRoad;
                            return true; // 'all'
                          })
                          .map((result, index) => {
                            const filename = result.filename;
                            const isRoad = result.isRoad;
                            
                            // Get image from stored processed data
                            let imagePreview = null;
                            let imageData = null;
                            
                            if (processedImagesData[filename]) {
                              // Use stored processed image data
                              imagePreview = processedImagesData[filename].originalImage;
                              imageData = processedImagesData[filename];
                              console.log('Found stored data for:', filename, 'hasImage:', !!imagePreview);
                            } else if (imagePreviewsMap[filename]) {
                              // Fallback to current preview (for any remaining unprocessed images)
                              imagePreview = imagePreviewsMap[filename];
                              imageData = {
                                originalImage: imagePreview,
                                processedImage: null,
                                results: null,
                                isRoad: isRoad
                              };
                              console.log('Using fallback data for:', filename, 'hasImage:', !!imagePreview);
                            } else {
                              console.log('No image data found for:', filename);
                            }

                            return (
                              <tr key={filename}>
                                <td>
                                  <div className="d-flex align-items-center">
                                    {imagePreview ? (
                                      <img
                                        src={imagePreview}
                                        alt={`Thumbnail ${index + 1}`}
                                        className="img-thumbnail me-2"
                                        style={{ 
                                          width: '60px', 
                                          height: '60px', 
                                          objectFit: 'cover',
                                          cursor: 'pointer'
                                        }}
                                        onClick={() => handleThumbnailClick(imageData)}
                                        title="Click to view full size"
                                      />
                                    ) : (
                                      <div 
                                        className="img-thumbnail me-2 d-flex align-items-center justify-content-center"
                                        style={{ 
                                          width: '60px', 
                                          height: '60px', 
                                          backgroundColor: '#f8f9fa',
                                          border: '1px solid #dee2e6'
                                        }}
                                      >
                                        <small className="text-muted">No Image</small>
                                      </div>
                                    )}
                                    <small className="text-muted">{filename}</small>
                                  </div>
                                </td>
                                <td>
                                  <span className={`badge ${isRoad ? 'bg-success' : 'bg-danger'}`}>
                                    {isRoad ? 'Road' : 'Non-Road'}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  </div>




                </Card.Body>
              </Card>
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
              
              <h5>Supported Image Formats</h5>
              <p>
                The system supports all common image formats including:
              </p>
              <ul>
                <li><strong>JPEG/JPG</strong> - Standard format, optimal for processing</li>
                <li><strong>PNG</strong> - Lossless format, good for detailed images</li>
                <li><strong>AVIF</strong> - Modern format, automatically converted to JPG for processing</li>
                <li><strong>WebP</strong> - Google's format, also automatically converted</li>
                <li><strong>Other formats</strong> - BMP, TIFF, etc. are also supported</li>
              </ul>
              
              <div className="alert alert-info">
                <h6><i className="fas fa-info-circle me-2"></i>Automatic Format Conversion</h6>
                <p className="mb-0">
                  <strong>AVIF and WebP images</strong> are automatically converted to JPG format during processing 
                  to ensure compatibility with the YOLO detection models. This conversion maintains image quality 
                  while ensuring reliable detection results.
                </p>
              </div>
              
              <p>
                The detected defects are automatically recorded in the database for tracking 
                and analysis in the Dashboard module.
              </p>
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>

      {/* Classification Error Modal */}
      {/* / <Modal
        show={showClassificationModal}
        onHide={() => setShowClassificationModal(false)}
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>
            <i className="fas fa-exclamation-triangle text-warning me-2"></i>
            Road Detection Failed
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="text-center">
            <div className="mb-3">
              <i className="fas fa-road fa-3x text-muted"></i>
            </div>
            <h5 className="text-danger mb-3">No Road Detected</h5>
            <p className="mb-3">
              {classificationError || 'The uploaded image does not appear to contain a road. Please upload an image that clearly shows a road surface for defect detection.'}
            </p>
            <div className="alert alert-info">
              <strong>Tips for better results:</strong>
              <ul className="mb-0 mt-2 text-start">
                <li>Ensure the image clearly shows a road surface</li>
                <li>Avoid images with only buildings, sky, or vegetation</li>
                <li>Make sure the road takes up a significant portion of the image</li>
                <li>Use good lighting conditions for clearer road visibility</li>
              </ul>
            </div>
          </div>
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="primary"
            onClick={() => setShowClassificationModal(false)}
          >
            Try Another Image
          </Button>
        </Modal.Footer>
      </Modal> */}

      {/* Image Modal for Full-Size View */}
      <Modal
        show={showImageModal}
        onHide={() => setShowImageModal(false)}
        size="lg"
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>
            <i className="fas fa-image me-2"></i>
            Image View {selectedImageData?.filename && (
              <small className="text-muted">- {selectedImageData.filename}</small>
            )}
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedImageData && (
            <div className="text-center">
              <div className="mb-4">
                <h6 className="mb-3">
                  <i className="fas fa-camera me-2"></i>
                  Original Image
                </h6>
                <img
                  src={selectedImageData.originalImage}
                  alt="Original Image"
                  className="img-fluid"
                  style={{
                    maxHeight: '400px',
                    borderRadius: '8px',
                    border: '2px solid #dee2e6',
                    boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                  }}
                />
              </div>
              {selectedImageData.processedImage && selectedImageData.isRoad && (
                <div className="mt-4">
                  <h6 className="mb-3">
                    <i className="fas fa-search me-2"></i>
                    Processed Image (Detection Results)
                  </h6>
                  <img
                    src={selectedImageData.processedImage}
                    alt="Processed Image"
                    className="img-fluid"
                    style={{
                      maxHeight: '400px',
                      borderRadius: '8px',
                      border: '2px solid #28a745',
                      boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                    }}
                  />
                </div>
              )}
              {!selectedImageData.isRoad && (
                <div className="mt-3">
                  <Alert variant="info">
                    <i className="fas fa-info-circle me-2"></i>
                    This image was classified as non-road and therefore no defect detection was performed.
                  </Alert>
                </div>
              )}
            </div>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="secondary"
            onClick={() => setShowImageModal(false)}
          >
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

// Add CSS styles for the enhanced detection table
const styles = `
  .detection-table-container {
    max-height: 600px;
    overflow-y: auto;
  }

  .detection-table-container th {
    position: sticky;
    top: 0;
    background-color: #f8f9fa;
    z-index: 10;
  }

  .detection-table-container th:hover {
    background-color: #e9ecef;
  }

  .detection-summary-cards .card {
    transition: transform 0.2s ease-in-out;
  }

  .detection-summary-cards .card:hover {
    transform: translateY(-2px);
  }

  .table-responsive {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }

  .badge {
    font-size: 0.75em;
  }

  .avif-indicator {
    position: absolute;
    top: 5px;
    left: 5px;
    background-color: rgba(23, 162, 184, 0.9);
    color: white;
    padding: 2px 6px;
    border-radius: 12px;
    font-size: 10px;
    display: flex;
    align-items: center;
    z-index: 5;
  }

  .image-thumbnail {
    position: relative;
  }

  @media (max-width: 768px) {
    .detection-summary-cards .col-md-3 {
      margin-bottom: 1rem;
    }

    .d-flex.gap-2.flex-wrap {
      flex-direction: column;
    }

    .d-flex.gap-2.flex-wrap .btn {
      margin-bottom: 0.5rem;
    }
  }
`;

// Inject styles into the document head
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');

  styleSheet.innerText = styles;
  document.head.appendChild(styleSheet);
}

export default Pavement;
 


