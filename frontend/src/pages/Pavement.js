import React, { useState, useRef } from 'react';
import { Container, Row, Col, Card, Button, Form, Tabs, Tab, Alert, Spinner } from 'react-bootstrap';
import axios from 'axios';
import Webcam from 'react-webcam';
import './Pavement.css';

const Pavement = () => {
  const [activeTab, setActiveTab] = useState('detection');
  const [detectionType, setDetectionType] = useState('potholes');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [cameraActive, setCameraActive] = useState(false);
  const [coordinates, setCoordinates] = useState('Not Available');
  
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  // Handle file input change
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
        
        // Try to extract EXIF coordinates from the image
        // We'll handle this on the backend to avoid requiring EXIF libraries in the frontend
        // The coordinates will be extracted when we send the image for processing
      };
      reader.readAsDataURL(file);
      
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
      setImagePreview(imageSrc);
      setImageFile(null);
      setProcessedImage(null);
      setResults(null);
      setError('');
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

  // Process image for detection
  const handleDetect = async () => {
    if (!imagePreview) {
      setError('Please upload or capture an image first');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Get user info from session storage
      const userString = sessionStorage.getItem('user');
      const user = userString ? JSON.parse(userString) : null;
      
      // Prepare request data
      const requestData = {
        image: imagePreview,
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

  // Reset detection
  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setProcessedImage(null);
    setResults(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
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
                  />
                  <Button 
                    variant="success" 
                    onClick={handleCapture}
                    className="mt-2"
                  >
                    Capture Photo
                  </Button>
                </div>
              )}

              {imagePreview && (
                <div className="image-preview-container mb-3">
                  <h5>Preview</h5>
                  <img 
                    src={imagePreview} 
                    alt="Preview" 
                    className="image-preview img-fluid" 
                  />
                </div>
              )}

              {error && <Alert variant="danger">{error}</Alert>}

              <div className="d-flex gap-2">
                <Button 
                  variant="primary" 
                  onClick={handleDetect}
                  disabled={!imagePreview || loading}
                >
                  {loading ? (
                    <>
                      <Spinner
                        as="span"
                        animation="border"
                        size="sm"
                        role="status"
                        aria-hidden="true"
                      />
                      <span className="ms-2">Detecting...</span>
                    </>
                  ) : (
                    `Detect ${detectionType === 'potholes' ? 'Potholes' : detectionType === 'cracks' ? 'Cracks' : 'Kerbs'}`
                  )}
                </Button>
                <Button 
                  variant="secondary" 
                  onClick={handleReset}
                  disabled={loading}
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