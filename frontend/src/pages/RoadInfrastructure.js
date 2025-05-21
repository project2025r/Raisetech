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
  const [isProcessing, setIsProcessing] = useState(false);
  const [shouldStop, setShouldStop] = useState(false);
  const [isBuffering, setIsBuffering] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [frameBuffer, setFrameBuffer] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const BUFFER_SIZE = 10; // Number of frames to buffer before playback
  const PLAYBACK_FPS = 15; // Playback frame rate
  const [liveDistinctTable, setLiveDistinctTable] = useState([]);
  const [liveContinuousTable, setLiveContinuousTable] = useState([]);
  const [classNames, setClassNames] = useState([]);
  
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const { isMobile } = useResponsive();
  const eventSourceRef = useRef(null);

  // Road infrastructure classes (from the Python model)
  const roadInfraClasses = [
    'Hot Thermoplastic Paint-edge_line-',
    'Water-Based Kerb Paint',
    'Single W Metal Beam Crash Barrier',
    'Hot Thermoplastic Paint-lane_line-',
    'Rubber Speed Breaker',
    'YNM Informatory Sign Boards',
    'Cold Plastic Rumble Marking Paint',
    'Raised Pavement Markers'
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

  // Handle input source change
  useEffect(() => {
    // Reset relevant states when input source changes
    setImagePreview(null);
    setVideoPreview(null);
    setProcessedImage(null);
    setDetectionResults(null);
    setError('');
    
    if (inputSource === 'camera') {
      setCameraActive(true);
    } else {
      setCameraActive(false);
    }
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [inputSource]);

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
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setImagePreview(imageSrc);
        setImageFile(null);
        setProcessedImage(null);
        setDetectionResults(null);
        setError('');
      }
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

  // Check if we have media for detection
  const hasMediaForDetection = () => {
    if (inputSource === 'video') return !!videoPreview;
    if (inputSource === 'image' || inputSource === 'camera') return !!imagePreview;
    return false;
  };

  // Reset detection
  const handleReset = () => {
    setShouldStop(true);
    setVideoFile(null);
    setVideoPreview(null);
    setImageFile(null);
    setImagePreview(null);
    setProcessedImage(null);
    setDetectionResults(null);
    setError('');
    setIsProcessing(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Process image/video for detection
  const handleDetect = async () => {
    if (!hasMediaForDetection() || selectedClasses.length === 0) {
      setError('Please select input media and at least one detection class');
      return;
    }

    console.log('Starting detection process...');
    console.log('Selected classes:', selectedClasses);
    console.log('Input source:', inputSource);
    console.log('Coordinates:', coordinates);

    setLoading(true);
    setError('');
    setProcessedImage(null);

    try {
      const formData = new FormData();
      formData.append('type', 'road_infra');
      formData.append('selectedClasses', JSON.stringify(selectedClasses));
      formData.append('coordinates', coordinates);

      // Handle different input sources
      if (inputSource === 'video' && videoFile) {
        // Video processing setup
        console.log('Processing video file:', videoFile.name);
        formData.append('video', videoFile);
        setIsProcessing(true);
        setShouldStop(false);
        setIsBuffering(true);
        setIsPlaying(false);
        setFrameBuffer([]);
        setCurrentFrameIndex(0);
      } else if ((inputSource === 'image' || inputSource === 'camera') && imagePreview) {
        // Image/camera processing setup
        console.log('Processing image/camera input');
        const blob = await (await fetch(imagePreview)).blob();
        formData.append('image', blob, 'capture.jpg');
      }

      console.log('Sending request to backend...');
      
      // Send the request to the backend
      const uploadResponse = await axios.post('/api/road-infrastructure/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      console.log('Upload response:', uploadResponse.data);
      
      if (!uploadResponse.data.success) {
        throw new Error(uploadResponse.data.message);
      }
      
      // Handle different responses based on input type
      if (inputSource === 'video') {
        // For video, establish SSE connection for streaming results
        const eventSource = new EventSource('/api/road-infrastructure/detect');
        eventSourceRef.current = eventSource;
        
        eventSource.onmessage = (event) => {
          if (eventSourceRef.current === null) return; // If stopped, ignore
          const data = JSON.parse(event.data);
          console.log('Received frame data:', data);
          
          if (data.success === false) {
            setError(data.message || 'Detection failed');
            eventSource.close();
            eventSourceRef.current = null;
            setIsProcessing(false);
            setLoading(false);
            return;
          }
          
          if (data.frame && typeof data.frame === 'string' && data.frame.length > 1000) {
            setFrameBuffer(prev => {
              const newBuffer = [...prev, data.frame];
              if (newBuffer.length >= BUFFER_SIZE && !isPlaying) {
                setIsBuffering(false);
                setIsPlaying(true);
              }
              return newBuffer;
            });
          } else if (data.frame && data.frame.length <= 1000) {
            console.warn('Received a frame, but it is too short to be valid. Skipping.');
          }
          
          // Update detection results
          if (data.detections) {
            setDetectionResults(prev => ({
              ...prev,
              total_frames: data.total_frames,
              processed_frames: data.frame_count,
              detections: [...(prev?.detections || []), ...data.detections],
              continuous_lengths: data.continuous_lengths,
              output_path: data.output_path
            }));
          }
          
          // Update live tables
          if (data.live_distinct_table) setLiveDistinctTable(data.live_distinct_table);
          if (data.live_continuous_table) setLiveContinuousTable(data.live_continuous_table);
          
          if (data.class_names && Array.isArray(data.class_names)) {
            setClassNames(data.class_names);
          }
          
          // Check if this is the final message
          if (data.tracked_objects) {
            eventSource.close();
            eventSourceRef.current = null;
            setIsProcessing(false);
            setLoading(false);
          }

          if (data.stopped_early !== undefined) {
            console.log('Processing ended:', data.stopped_early ? 'stopped early' : 'completed');
            setIsProcessing(false);
            setIsBuffering(false);
            setLoading(false);
            if (data.output_path) {
              // setError(`Processing ${data.stopped_early ? 'stopped' : 'completed'}. Video saved to: ${data.output_path}`);
            }
          }
        };
        
        eventSource.onerror = (error) => {
          console.error('EventSource error:', error);
          eventSource.close();
          eventSourceRef.current = null;
          setIsProcessing(false);
          setLoading(false);
        };
        
        // Handle stop request
        if (shouldStop) {
          eventSource.close();
          eventSourceRef.current = null;
          setIsProcessing(false);
          setLoading(false);
        }
      } else if (inputSource === 'image' || inputSource === 'camera') {
        // For image/camera, process the direct response
        console.log('Processing image response:', uploadResponse.data);
        
        // Set the processed image
        if (uploadResponse.data.frame) {
          setProcessedImage(uploadResponse.data.frame);
        }
        
        // Set detection results
        if (uploadResponse.data.detections) {
          setDetectionResults({
            detections: uploadResponse.data.detections,
            total_frames: 1,
            processed_frames: 1
          });
        }
        
        setLoading(false);
      }

    } catch (error) {
      console.error('Detection error:', error);
      setError(
        error.response?.data?.message || 
        error.message ||
        'An error occurred during detection. Please try again.'
      );
      setLoading(false);
      setIsProcessing(false);
    }
  };

  // Playback timer: play frames from buffer at fixed FPS
  useEffect(() => {
    let playbackInterval;
    if (isPlaying && frameBuffer.length > 0) {
      playbackInterval = setInterval(() => {
        setCurrentFrameIndex(prev => {
          if (prev < frameBuffer.length - 1) {
            return prev + 1;
          } else {
            setIsPlaying(false); // Stop at the end
            return prev;
          }
        });
      }, 1000 / PLAYBACK_FPS);
    }
    return () => {
      if (playbackInterval) clearInterval(playbackInterval);
    };
  }, [isPlaying, frameBuffer]);

  // Update processedImage when currentFrameIndex changes
  useEffect(() => {
    if (frameBuffer.length > 0 && currentFrameIndex < frameBuffer.length) {
      setProcessedImage(frameBuffer[currentFrameIndex]);
    }
  }, [currentFrameIndex, frameBuffer]);

  // Playback controls
  const handlePlayPause = () => setIsPlaying(p => !p);
  const handleRewind = () => setCurrentFrameIndex(i => Math.max(i - 5, 0));
  const handleForward = () => setCurrentFrameIndex(i => Math.min(i + 5, frameBuffer.length - 1));
  const handleSliderChange = (e) => setCurrentFrameIndex(Number(e.target.value));

  // Group detections by class
  const getClassCounts = () => {
    if (!detectionResults || !detectionResults.detections) return {};
    
    return detectionResults.detections.reduce((acc, det) => {
      acc[det.class] = (acc[det.class] || 0) + 1;
      return acc;
    }, {});
  };

  // Add this new function after the getClassCounts function (around line 380-390)
  const stopProcessing = async () => {
    try {
      // Close the EventSource connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }

      // Send stop signal to backend
      const response = await axios.post('/api/road-infrastructure/stop_processing');
      console.log('Stop processing response:', response.data);

      // Update UI state
      setIsProcessing(false);
      setShouldStop(true);
      setIsBuffering(false);
      setIsPlaying(false);
      setLoading(false);

      // Keep the last frame and tables visible
      if (frameBuffer.length > 0) {
        setProcessedImage(frameBuffer[frameBuffer.length - 1]);
      }

      return true;
    } catch (error) {
      console.error('Error stopping processing:', error);
      setError('Failed to stop processing: ' + error.message);
      setLoading(false);
      return false;
    }
  };

  // Then modify the existing handleStopProcessing function (around line 399-407)
  const handleStopProcessing = async () => {
    if (isProcessing) {
      const stopped = await stopProcessing();
      setLoading(false);
      if (stopped) {
        setError('Processing stopped. Video has been saved.');
      }
    }
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
                      {(classNames.length > 0 ? classNames : [
                        'Hot Thermoplastic Paint-edge_line-',
                        'Water-Based Kerb Paint',
                        'Single W Metal Beam Crash Barrier',
                        'Hot Thermoplastic Paint-lane_line-',
                        'Rubber Speed Breaker',
                        'YNM Informatory Sign Boards',
                        'Cold Plastic Rumble Marking Paint',
                        'Raised Pavement Markers'
                      ]).map((cls) => (
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
                    <div className="text-center mt-3">
                      <Button 
                        variant={cameraActive ? "danger" : "info"} 
                        onClick={toggleCamera}
                        className="mb-2"
                      >
                        {cameraActive ? 'Stop Camera' : 'Start Camera'}
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
                        </>
                      )}
                      
                      {imagePreview && (
                        <div className="mt-3">
                          <img 
                            src={imagePreview} 
                            alt="Captured" 
                            style={{ maxWidth: '100%', maxHeight: '300px' }} 
                          />
                        </div>
                      )}
                    </div>
                  )}

                  {error && <Alert variant="danger">{error}</Alert>}
                  <div className="d-flex gap-2 mt-3">
                    <Button 
                      variant="primary" 
                      onClick={handleDetect}
                      disabled={loading || 
                        !hasMediaForDetection() ||
                        selectedClasses.length === 0 ||
                        isProcessing}
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
                      ) : (
                        "Detect Infrastructure"
                      )}
                    </Button>
              
                    <Button 
                      variant="secondary" 
                      onClick={isProcessing ? handleStopProcessing : handleReset}
                      disabled={loading && !isProcessing}
                    >
                      {isProcessing ? "Stop Processing" : "Reset"}
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
                      {isBuffering && (
                        <div className="processing-overlay">
                          <Spinner animation="border" role="status">
                            <span className="visually-hidden">Buffering video...</span>
                          </Spinner>
                          <span style={{ color: 'white', marginLeft: 10 }}>Buffering video...</span>
                        </div>
                      )}
                      <img
                        src={processedImage ? (processedImage.startsWith('data:') ? processedImage : `data:image/jpeg;base64,${processedImage}`) : ''}
                        alt="Processed"
                        style={{ maxWidth: '100%' }}
                        onError={(e) => { e.target.onerror = null; e.target.src = ''; setError('Failed to display processed frame.'); }}
                      />
                      {isProcessing && !isBuffering && (
                        <div className="processing-overlay">
                          <Spinner animation="border" role="status">
                            <span className="visually-hidden">Processing...</span>
                          </Spinner>
                        </div>
                      )}
                      {/* Playback controls */}
                      {frameBuffer.length > 0 && !isBuffering && (
                        <div style={{ display: 'flex', alignItems: 'center', marginTop: 10, gap: 10 }}>
                          <button onClick={handleRewind} disabled={currentFrameIndex === 0}>⏪</button>
                          <button onClick={handlePlayPause}>{isPlaying ? '⏸️ Pause' : '▶️ Play'}</button>
                          <button onClick={handleForward} disabled={currentFrameIndex >= frameBuffer.length - 1}>⏩</button>
                          <input
                            type="range"
                            min={0}
                            max={frameBuffer.length - 1}
                            value={currentFrameIndex}
                            onChange={handleSliderChange}
                            style={{ flex: 1 }}
                          />
                          <span style={{ minWidth: 60 }}>{currentFrameIndex + 1} / {frameBuffer.length}</span>
                        </div>
                      )}
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
                        
                        {detectionResults.continuous_lengths && Object.keys(detectionResults.continuous_lengths).length > 0 && (
                          <>
                            <h5>Cumulative Lengths (Continuous Classes)</h5>
                            <div className="table-responsive">
                              <table className="table table-striped">
                                <thead>
                                  <tr>
                                    <th>Infrastructure Type</th>
                                    <th>Cumulative Length (km)</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(detectionResults.continuous_lengths).map(([cls, length]) => (
                                    <tr key={cls}>
                                      <td>{cls}</td>
                                      <td>{length}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </>
                        )}
                      </>
                    )}
                    {/* Live Discrete Table */}
                    {liveDistinctTable.length > 0 && (
                      <div style={{ marginTop: 20 }}>
                        <h5>Live Discrete Detections</h5>
                        <div className="table-responsive" style={{ maxHeight: 200, overflowY: 'auto' }}>
                          <table className="table table-striped table-sm">
                            <thead>
                              <tr>
                                <th>ID</th>
                                <th>Class</th>
                                <th>GPS</th>
                                <th>Frame</th>
                                <th>Second</th>
                              </tr>
                            </thead>
                            <tbody>
                              {liveDistinctTable.map(row => (
                                <tr key={row.ID}>
                                  <td>{row.ID}</td>
                                  <td>{row.Class}</td>
                                  <td>{row.GPS ? `${row.GPS[0].toFixed(6)}, ${row.GPS[1].toFixed(6)}` : '-'}</td>
                                  <td>{row.Frame}</td>
                                  <td>{row.Second}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                    {/* Live Continuous Table */}
                    {liveContinuousTable.length > 0 && (
                      <div style={{ marginTop: 20 }}>
                        <h5>Live Continuous (Cumulative) Data</h5>
                        <div className="table-responsive" style={{ maxHeight: 150, overflowY: 'auto' }}>
                          <table className="table table-striped table-sm">
                            <thead>
                              <tr>
                                <th>Class</th>
                                <th>Cumulative Length (km)</th>
                              </tr>
                            </thead>
                            <tbody>
                              {liveContinuousTable.map(row => (
                                <tr key={row.Class}>
                                  <td>{row.Class}</td>
                                  <td>{row['Cumulative Length (km)']}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
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