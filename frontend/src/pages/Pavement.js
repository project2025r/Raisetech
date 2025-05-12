import React, { useState, useRef, useEffect } from 'react';
import { Container, Card, Button, Form, Tabs, Tab, Alert, Spinner } from 'react-bootstrap';
import axios from 'axios';
import Webcam from 'react-webcam';
import './Pavement.css';
import useResponsive from '../hooks/useResponsive';

const Pavement = () => {
  const [activeTab, setActiveTab] = useState('detection');
  const [detectionType, setDetectionType] = useState('potholes');
  const [imageFiles, setImageFiles] = useState([]);
  const [imagePreviewsMap, setImagePreviewsMap] = useState({});
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [processedImage, setProcessedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [coordinates, setCoordinates] = useState('Not Available');
  const [cameraOrientation, setCameraOrientation] = useState('environment');
  
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

  // Handle multiple file input change
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      setImageFiles([...imageFiles, ...files]);
      
      // Create previews for each file
      files.forEach(file => {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImagePreviewsMap(prev => ({
            ...prev,
            [file.name]: reader.result
          }));
        };
        reader.readAsDataURL(file);
      });
      
      // Reset results
      setProcessedImage(null);
      setResults(null);
      setError('');
    }
  };

  // Handle camera capture
  const handleCapture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      const timestamp = new Date().toISOString();
      const filename = `camera_capture_${timestamp}.jpg`;
      
      setImageFiles([...imageFiles, filename]);
      setImagePreviewsMap(prev => ({
        ...prev,
        [filename]: imageSrc
      }));
      setCurrentImageIndex(imageFiles.length);
      
      setProcessedImage(null);
      setResults(null);
      setError('');
      // Note: We intentionally don't reset coordinates here to preserve location data from the camera
    }
  };

  // Toggle camera
  const toggleCamera = () => {
    setCameraActive(!cameraActive);
    if (!cameraActive) {
      // Get location when camera is activated
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const { latitude, longitude } = position.coords;
            setCoordinates(`${latitude}, ${longitude}`);
          },
          (err) => {
            console.error("Error getting location:", err);
          }
        );
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
      
      // Prepare request data
      const requestData = {
        image: currentImagePreview,
        coordinates: coordinates,
        username: user?.username || 'Unknown',
        role: user?.role || 'Unknown'
      };

      // Determine endpoint based on detection type
      let endpoint;
      switch(detectionType) {
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
          endpoint = '/api/pavement/detect-potholes';
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
          endpoint = '/api/pavement/detect-potholes';
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
          
          // Prepare request data
          const requestData = {
            image: imageData,
            coordinates: coordinates,
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
    setCurrentImageIndex(0);
    setProcessedImage(null);
    setResults(null);
    setError('');
    setBatchResults([]);
    setProcessedCount(0);
    setTotalToProcess(0);
    
    // Only reset coordinates if the camera is not active
    if (!cameraActive) {
      setCoordinates('Not Available');
    }
    
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
        <Tab eventKey="detection" title="Detection">
          <Card className="mb-4">
            <Card.Body>
              <Form.Group className="mb-3">
                <Form.Label>Detection Type</Form.Label>
                <Form.Select 
                  value={detectionType}
                  onChange={(e) => setDetectionType(e.target.value)}
                >
                  <option value="potholes">Potholes</option>
                  <option value="cracks">Alligator Cracks</option>
                  <option value="kerbs">Kerbs</option>
                </Form.Select>
              </Form.Group>

              <div className="mb-3">
                <Form.Label>Image Source</Form.Label>
                <div className="d-flex gap-2">
                  <Button 
                    variant={cameraActive ? "primary" : "outline-primary"}
                    onClick={toggleCamera}
                  >
                    {cameraActive ? "Disable Camera" : "Enable Camera"}
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
                            delete newPreviewsMap[name];
                            setImageFiles(newFiles);
                            setImagePreviewsMap(newPreviewsMap);
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
                <div className="processed-image-container mb-3">
                  <img 
                    src={processedImage} 
                    alt="Processed" 
                    className="processed-image img-fluid" 
                  />
                </div>

                {results && (
                  <div className="results-summary">
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
                                  <th>Volume</th>
                                  <th>Volume Range</th>
                                  <th>Coordinates</th>
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
                                    <td>{pothole.coordinates}</td>
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
                                  <th>Confidence</th>
                                  <th>Coordinates</th>
                                </tr>
                              </thead>
                              <tbody>
                                {results.cracks.map((crack) => (
                                  <tr key={crack.crack_id}>
                                    <td>{crack.crack_id}</td>
                                    <td>{crack.crack_type}</td>
                                    <td>{crack.area_cm2.toFixed(2)}</td>
                                    <td>{crack.area_range}</td>
                                    <td>{(crack.confidence * 100).toFixed(1)}%</td>
                                    <td>{crack.coordinates}</td>
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
                                  <th>Length (m)</th>
                                  <th>Condition</th>
                                  <th>Confidence</th>
                                  <th>Coordinates</th>
                                </tr>
                              </thead>
                              <tbody>
                                {results.kerbs.map((kerb) => (
                                  <tr key={kerb.kerb_id}>
                                    <td>{kerb.kerb_id}</td>
                                    <td>{kerb.kerb_type}</td>
                                    <td>{kerb.length_m.toFixed(2)}</td>
                                    <td>{kerb.condition}</td>
                                    <td>{(kerb.confidence * 100).toFixed(1)}%</td>
                                    <td>{kerb.coordinates}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        {results.condition_counts && (
                          <div>
                            <h5>Kerb Condition Summary</h5>
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
              
              <h5>How to Use This Module</h5>
              <ol>
                <li>Select the detection type (Potholes, Alligator Cracks, or Kerbs)</li>
                <li>Upload an image or use the camera to capture a photo</li>
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
 