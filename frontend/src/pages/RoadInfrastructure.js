import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card, Button, Form, Alert, Spinner, Tab, Tabs } from 'react-bootstrap';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import Webcam from 'react-webcam';
import useResponsive from '../hooks/useResponsive';

// Fix the marker icon issue with Leaflet in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

function RoadInfrastructure() {
  const [selectedClasses, setSelectedClasses] = useState([]);
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [detectionResults, setDetectionResults] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [coordinates, setCoordinates] = useState('Not Available');
  const [inputSource, setInputSource] = useState('video');
  const [activeTab, setActiveTab] = useState('detection');
  const [cameraOrientation, setCameraOrientation] = useState('environment');
  
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const { isMobile } = useResponsive();

  // Road infrastructure classes (from the Python model)
  const roadInfraClasses = [
    "Cold Plastic Rumble Marking Paint",
    "Raised Pavement Markers",
    "Rubber Speed Breaker",
    "SW_Beam_Crash_Barrier",
    "Water-Based Kerb Paint",
    "YNM Informatory Sign Boards",
    "HTP-edge_line",
    "HTP-lane_line"
  ];

  // Find user's location
  useEffect(() => {
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
  }, []);

  // Handle file input change for video
  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setProcessedImage(null);
      setDetectionResults(null);
      setError('');
    }
  };

  // Handle file input change for image
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      
      // Reset results
      setProcessedImage(null);
      setDetectionResults(null);
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
      setDetectionResults(null);
      setError('');
    }
  };

  // Toggle camera
  const toggleCamera = () => {
    setCameraActive(!cameraActive);
    if (!cameraActive) {
      // Update coordinates when camera is activated
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

  // Process image/video for detection
  const handleDetect = async () => {
    if ((!imagePreview && !videoPreview) || selectedClasses.length === 0) {
      setError('Please select input media and at least one detection class');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Prepare request data
      const requestData = {
        type: 'road_infra',
        image: imagePreview,
        coordinates: coordinates,
        selectedClasses: selectedClasses
      };

      // Make API request
      const response = await axios.post('/api/road-infrastructure/detect', requestData);

      // Handle response
      if (response.data.success) {
        setProcessedImage(response.data.processed_image);
        setDetectionResults(response.data);
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
    setVideoFile(null);
    setVideoPreview(null);
    setImageFile(null);
    setImagePreview(null);
    setProcessedImage(null);
    setDetectionResults(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Group detections by class
  const getClassCounts = () => {
    if (!detectionResults || !detectionResults.detections) return {};
    
    return detectionResults.detections.reduce((acc, det) => {
      acc[det.class] = (acc[det.class] || 0) + 1;
      return acc;
    }, {});
  };

  return (
    <Container fluid className="mt-4">
      <h1 className="text-center mb-4">Road Infrastructure Analysis</h1>
      
      <Tabs
        activeKey={activeTab}
        onSelect={(k) => setActiveTab(k)}
        className="mb-4"
      >
        <Tab eventKey="detection" title="Detection">
        <Row>
          <Col md={6}>
            <Card className="mb-4 shadow-sm">
              <Card.Header className="bg-primary text-white">
                  <h5 className="mb-0">Detection Settings</h5>
              </Card.Header>
              <Card.Body>
                  <Form.Group className="mb-3">
                    <Form.Label>Select Infrastructure Classes to Detect</Form.Label>
                    <Form.Control
                      as="select"
                      multiple
                      value={selectedClasses}
                      onChange={(e) => setSelectedClasses([...e.target.selectedOptions].map(opt => opt.value))}
                      style={{ height: '150px' }}
                    >
                      {roadInfraClasses.map((cls) => (
                        <option key={cls} value={cls}>{cls}</option>
                      ))}
                    </Form.Control>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>Input Source</Form.Label>
                    <Form.Select
                      value={inputSource}
                      onChange={(e) => setInputSource(e.target.value)}
                    >
                      <option value="video">Video</option>
                      <option value="image">Image</option>
                      <option value="camera">Live Camera</option>
                    </Form.Select>
                  </Form.Group>

                  {inputSource === 'video' && (
                    <Form.Group className="mb-3">
                      <Form.Label>Upload Video</Form.Label>
                      <Form.Control 
                        type="file" 
                        accept="video/*"
                        onChange={handleVideoChange}
                        ref={fileInputRef}
                      />
                      {videoPreview && (
                        <div className="mt-3">
                          <video 
                            src={videoPreview} 
                            controls 
                            style={{ maxWidth: '100%', maxHeight: '300px' }} 
                          />
                        </div>
                      )}
                    </Form.Group>
                  )}

                  {inputSource === 'image' && (
                    <Form.Group className="mb-3">
                      <Form.Label>Upload Image</Form.Label>
                      <Form.Control 
                        type="file" 
                        accept="image/*"
                        onChange={handleImageChange}
                        ref={fileInputRef}
                      />
                      {imagePreview && (
                        <div className="mt-3">
                          <img 
                            src={imagePreview} 
                            alt="Preview" 
                            style={{ maxWidth: '100%', maxHeight: '300px' }} 
                          />
                        </div>
                      )}
                    </Form.Group>
                  )}

                  {inputSource === 'camera' && (
                    <div className="mb-3">
                      <Button 
                        variant={cameraActive ? "primary" : "outline-primary"}
                        onClick={toggleCamera}
                        className="mb-3"
                      >
                        {cameraActive ? "Disable Camera" : "Enable Camera"}
                      </Button>
                      
                      {cameraActive && (
                        <>
                          <div className="webcam-container mb-3">
                            <Webcam
                              audio={false}
                              ref={webcamRef}
                              screenshotFormat="image/jpeg"
                              width="100%"
                              height="auto"
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
                          </div>
                          <Button 
                            variant="success" 
                            onClick={handleCapture}
                          >
                            Capture Photo
                          </Button>
                          {imagePreview && (
                            <div className="mt-3">
                              <img 
                                src={imagePreview} 
                                alt="Captured" 
                                style={{ maxWidth: '100%', maxHeight: '300px' }} 
                              />
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )}

                  {error && <Alert variant="danger">{error}</Alert>}

                  <div className="d-flex gap-2 mt-3">
                    <Button 
                      variant="primary" 
                      onClick={handleDetect}
                      disabled={loading || 
                        (inputSource === 'video' && !videoPreview) || 
                        (inputSource === 'image' && !imagePreview) || 
                        (inputSource === 'camera' && !imagePreview) ||
                        selectedClasses.length === 0}
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
                          <span className="ms-2">Processing...</span>
                        </>
                      ) : "Detect Infrastructure"}
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
          </Col>
          
          <Col md={6}>
              {processedImage ? (
            <Card className="mb-4 shadow-sm">
                  <Card.Header className="bg-success text-white">
                    <h5 className="mb-0">Detection Results</h5>
              </Card.Header>
              <Card.Body>
                    <div className="processed-image-container mb-3">
                      <img 
                        src={processedImage} 
                        alt="Processed" 
                        style={{ maxWidth: '100%' }}
                      />
                    </div>
                    
                    {detectionResults && detectionResults.detections && (
                      <>
                        <h5>Detection Summary</h5>
                <div className="table-responsive">
                  <table className="table table-striped">
                    <thead>
                      <tr>
                                <th>Infrastructure Type</th>
                                <th>Count</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(getClassCounts()).map(([cls, count]) => (
                                <tr key={cls}>
                                  <td>{cls}</td>
                                  <td>{count}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        
                        <h5>Detection Details</h5>
                        <div className="table-responsive" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                          <table className="table table-striped">
                            <thead>
                              <tr>
                                <th>ID</th>
                                <th>Class</th>
                                <th>Confidence</th>
                                <th>Coordinates</th>
                      </tr>
                    </thead>
                    <tbody>
                              {detectionResults.detections.map((detection) => (
                                <tr key={detection.id}>
                                  <td>{detection.id}</td>
                                  <td>{detection.class}</td>
                                  <td>{(detection.confidence * 100).toFixed(1)}%</td>
                                  <td>{detection.coordinates}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                      </>
                    )}
                  </Card.Body>
                </Card>
              ) : (
                <Card className="mb-4 shadow-sm">
                  <Card.Header className="bg-info text-white">
                    <h5 className="mb-0">Instructions</h5>
                  </Card.Header>
                  <Card.Body>
                    <ol>
                      <li>Select one or more infrastructure types to detect</li>
                      <li>Choose your input source (video, image, or camera)</li>
                      <li>Upload media or capture a photo</li>
                      <li>Click "Detect Infrastructure" to analyze</li>
                    </ol>
                    <p>Detection will identify and highlight road infrastructure features such as:</p>
                    <ul>
                      <li>Pavement markings</li>
                      <li>Road signs</li>
                      <li>Safety barriers</li>
                      <li>Road edge lines</li>
                      <li>Lane markings</li>
                    </ul>
              </Card.Body>
            </Card>
              )}
          </Col>
        </Row>
        </Tab>
        
        <Tab eventKey="information" title="Information">
          <Card className="shadow-sm">
            <Card.Body>
              <h4>About Road Infrastructure Analysis</h4>
              <p>
                The Road Infrastructure Analysis module uses computer vision to detect, classify, and 
                analyze various road infrastructure elements. This helps in infrastructure inventory 
                management and maintenance planning.
              </p>
              
              <h5>Detectable Infrastructure Types</h5>
              <ul>
                <li><strong>Cold Plastic Rumble Marking Paint</strong> - Textured road markings that provide tactile and auditory warnings</li>
                <li><strong>Raised Pavement Markers</strong> - Reflective or non-reflective markers installed on roadways</li>
                <li><strong>Rubber Speed Breaker</strong> - Traffic calming devices to reduce vehicle speeds</li>
                <li><strong>SW_Beam_Crash_Barrier</strong> - Safety barriers to prevent vehicles from veering off the road</li>
                <li><strong>Water-Based Kerb Paint</strong> - Paint used for road edge visibility and demarcation</li>
                <li><strong>YNM Informatory Sign Boards</strong> - Road signs providing information to road users</li>
                <li><strong>HTP-edge_line</strong> - High-performance thermoplastic road edge line markings</li>
                <li><strong>HTP-lane_line</strong> - High-performance thermoplastic lane separators</li>
              </ul>
              
              <h5>How The System Works</h5>
              <p>
                The system uses a trained YOLOv8 object detection model to identify infrastructure elements 
                in images or video. For each detection, the system records:
              </p>
              <ul>
                <li>The type of infrastructure element</li>
                <li>Confidence score of the detection</li>
                <li>Physical dimensions (where applicable)</li>
                <li>Geolocation (when available from device GPS)</li>
              </ul>
              
              <h5>Use Cases</h5>
              <ul>
                <li>Road infrastructure inventory management</li>
                <li>Monitoring road marking conditions</li>
                <li>Planning maintenance and replacement schedules</li>
                <li>Assessing compliance with safety standards</li>
                <li>Calculating infrastructure density and distribution</li>
              </ul>
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>
    </Container>
  );
}

export default RoadInfrastructure; 