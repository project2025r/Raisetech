import React, { useState, useRef, useEffect } from 'react';
import { Card, Button, Form, Alert, Spinner, Table, Row, Col } from 'react-bootstrap';
import axios from 'axios';
import Webcam from 'react-webcam';
import useResponsive from '../hooks/useResponsive';
import './VideoDefectDetection.css';

const VideoDefectDetection = () => {
  const [selectedModel, setSelectedModel] = useState('All');
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [processedVideo, setProcessedVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [shouldStop, setShouldStop] = useState(false);
  const [coordinates, setCoordinates] = useState('Not Available');
  const [inputSource, setInputSource] = useState('video');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraOrientation, setCameraOrientation] = useState('environment');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordedChunks, setRecordedChunks] = useState([]);
  
  // Video processing states
  const [frameBuffer, setFrameBuffer] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isBuffering, setIsBuffering] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [allDetections, setAllDetections] = useState([]);
  const [videoResults, setVideoResults] = useState(null);
  
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const { isMobile } = useResponsive();
  
  const BUFFER_SIZE = 10;
  const PLAYBACK_FPS = 15;
  const MAX_RECORDING_TIME = 60; // 1 minute limit

  // Available models
  const modelOptions = [
    { value: 'All', label: 'All (detect all types of defects)' },
    { value: 'Potholes', label: 'Potholes' },
    { value: 'Alligator Cracks', label: 'Alligator Cracks' },
    { value: 'Kerbs', label: 'Kerbs' }
  ];

  // Get user location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setCoordinates(`${latitude.toFixed(6)}, ${longitude.toFixed(6)}`);
        },
        (err) => {
          console.error("Error getting location:", err);
          setCoordinates('Location unavailable');
        }
      );
    }
  }, []);

  // Recording timer
  useEffect(() => {
    if (isRecording) {
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= MAX_RECORDING_TIME) {
            handleStopRecording();
            return MAX_RECORDING_TIME;
          }
          return prev + 1;
        });
      }, 1000);
    } else {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    }
    
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, [isRecording]);

  // Video playback effect
  useEffect(() => {
    let playbackInterval;
    if (isPlaying && frameBuffer.length > 0) {
      playbackInterval = setInterval(() => {
        setCurrentFrameIndex(prev => {
          if (prev < frameBuffer.length - 1) {
            return prev + 1;
          } else {
            setIsPlaying(false);
            return prev;
          }
        });
      }, 1000 / PLAYBACK_FPS);
    }
    return () => {
      if (playbackInterval) clearInterval(playbackInterval);
    };
  }, [isPlaying, frameBuffer]);

  // Update processed video when frame changes
  useEffect(() => {
    if (frameBuffer.length > 0 && currentFrameIndex < frameBuffer.length) {
      setProcessedVideo(frameBuffer[currentFrameIndex]);
    }
  }, [currentFrameIndex, frameBuffer]);

  // Handle video file selection
  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setProcessedVideo(null);
      setVideoResults(null);
      setAllDetections([]);
      setError('');
    }
  };

  // Handle camera activation
  const toggleCamera = () => {
    setCameraActive(!cameraActive);
    if (!cameraActive) {
      setVideoFile(null);
      setVideoPreview(null);
      setProcessedVideo(null);
      setVideoResults(null);
      setAllDetections([]);
      setError('');
    }
  };

  // Start recording
  const handleStartRecording = async () => {
    if (!webcamRef.current || !webcamRef.current.stream) {
      setError('Camera not available');
      return;
    }

    try {
      setRecordedChunks([]);
      setRecordingTime(0);
      setIsRecording(true);
      setError('');

      const mediaRecorder = new MediaRecorder(webcamRef.current.stream, {
        mimeType: 'video/webm'
      });

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks(prev => [...prev, event.data]);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const file = new File([blob], `recorded_video_${Date.now()}.webm`, { type: 'video/webm' });
        setVideoFile(file);
        setVideoPreview(URL.createObjectURL(blob));
        setIsRecording(false);
        setRecordingTime(0);
      };

      mediaRecorder.start();
    } catch (error) {
      setError('Failed to start recording: ' + error.message);
      setIsRecording(false);
    }
  };

  // Stop recording
  const handleStopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingTime(0);
    }
  };

  // Toggle camera orientation
  const toggleCameraOrientation = () => {
    setCameraOrientation(prev => prev === 'environment' ? 'user' : 'environment');
  };

  // Check if ready for processing
  const isReadyForProcessing = () => {
    return (inputSource === 'video' && videoFile) || 
           (inputSource === 'camera' && videoFile);
  };

  // Handle video processing
  const handleProcess = async () => {
    if (!isReadyForProcessing()) {
      setError('Please provide a video file first');
      return;
    }

    setLoading(true);
    setError('');
    setIsProcessing(true);
    setShouldStop(false);
    setIsBuffering(true);
    setIsPlaying(false);
    setFrameBuffer([]);
    setCurrentFrameIndex(0);
    setProcessingProgress(0);
    setAllDetections([]);
    setProcessedVideo(null); // Reset processed video
    setVideoResults(null);   // Reset video results

    try {
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('selectedModel', selectedModel);
      formData.append('coordinates', coordinates);

      console.log('Starting video processing with model:', selectedModel);

      // Create FormData for SSE request
      const sseUrl = '/api/pavement/detect-video';
      
      // Use fetch for SSE with FormData
      const response = await fetch(sseUrl, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      const processStream = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              console.log('Stream ended naturally');
              setIsProcessing(false);
              setLoading(false);
              setIsBuffering(false);
              break;
            }

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.substring(6));
                  console.log('Received SSE data:', {
                    hasFrame: !!data.frame,
                    frameLength: data.frame ? data.frame.length : 0,
                    progress: data.progress,
                    frameCount: data.frame_count,
                    detections: data.detections ? data.detections.length : 0
                  });

                  if (data.success === false) {
                    setError(data.message || 'Video processing failed');
                    setIsProcessing(false);
                    setLoading(false);
                    setIsBuffering(false);
                    return;
                  }

                  if (data.frame && typeof data.frame === 'string' && data.frame.length > 1000) {
                    // Update frame buffer and current index for real-time display
                    setFrameBuffer(prev => {
                      const newBuffer = [...prev, data.frame];
                      
                      // Update current frame index to the latest frame for live preview
                      setCurrentFrameIndex(newBuffer.length - 1);
                      
                      // Start showing frames immediately
                      if (newBuffer.length === 1) {
                        setIsBuffering(false);
                        setIsPlaying(false); // Don't auto-play during live processing
                        setProcessedVideo(data.frame); // Show the first frame immediately
                      }
                      
                      return newBuffer;
                    });
                    
                    // Update the displayed frame for real-time preview
                    setProcessedVideo(data.frame);
                  }

                  // Update progress
                  if (data.progress !== undefined) {
                    setProcessingProgress(data.progress);
                    console.log(`Processing progress: ${data.progress.toFixed(1)}%`);
                  }

                  // Update detections
                  if (data.detections && data.detections.length > 0) {
                    setAllDetections(prev => [...prev, ...data.detections]);
                  }

                  // Handle final results
                  if (data.all_detections) {
                    setVideoResults(data);
                    setAllDetections(data.all_detections);
                    setIsProcessing(false);
                    setLoading(false);
                    setIsBuffering(false);
                    setProcessingProgress(100); // Ensure progress shows 100%
                    
                    // Reset to first frame for playback after processing
                    setCurrentFrameIndex(0);
                    setIsPlaying(false);
                    
                    console.log('Video processing completed');
                    console.log(`Total unique detections: ${data.total_unique_detections || data.all_detections.length}`);
                    console.log(`Total frame detections: ${data.total_frame_detections || data.all_detections.length}`);
                    console.log(`Total frames processed: ${frameBuffer.length}`);
                    return;
                  }

                  // Handle explicit end signal
                  if (data.end) {
                    console.log('Received end signal, closing stream');
                    setIsProcessing(false);
                    setLoading(false);
                    setIsBuffering(false);
                    return;
                  }
                } catch (parseError) {
                  console.warn('Error parsing SSE data:', parseError);
                }
              }
            }
          }
        } catch (streamError) {
          console.error('Stream processing error:', streamError);
          setError('Error processing video stream');
          setIsProcessing(false);
          setLoading(false);
          setIsBuffering(false);
        } finally {
          // Clean up reader
          if (reader) {
            try {
              reader.releaseLock();
            } catch (e) {
              console.warn('Error releasing reader lock:', e);
            }
          }
        }
      };

      processStream();

    } catch (error) {
      console.error('Video processing error:', error);
      setError(error.message || 'Video processing failed');
      setLoading(false);
      setIsProcessing(false);
    }
  };

  // Stop processing
  const handleStopProcessing = async () => {
    try {
      await axios.post('/api/pavement/stop-video-processing');
      
      setIsProcessing(false);
      setShouldStop(true);
      setIsBuffering(false);
      setIsPlaying(false);
      setLoading(false);
      setError('Video processing stopped');
    } catch (error) {
      console.error('Error stopping processing:', error);
      setError('Failed to stop processing');
    }
  };

  // Reset all
  const handleReset = () => {
    setVideoFile(null);
    setVideoPreview(null);
    setProcessedVideo(null);
    setVideoResults(null);
    setAllDetections([]);
    setFrameBuffer([]);
    setCurrentFrameIndex(0);
    setIsProcessing(false);
    setShouldStop(false);
    setIsBuffering(false);
    setIsPlaying(false);
    setProcessingProgress(0);
    setError('');
    setSelectedModel('All');
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Playback controls
  const handlePlayPause = () => setIsPlaying(!isPlaying);
  const handleRewind = () => setCurrentFrameIndex(Math.max(currentFrameIndex - 5, 0));
  const handleForward = () => setCurrentFrameIndex(Math.min(currentFrameIndex + 5, frameBuffer.length - 1));

  // Group detections by type
  const getDetectionSummary = () => {
    const summary = {};
    allDetections.forEach(det => {
      summary[det.type] = (summary[det.type] || 0) + 1;
    });
    return summary;
  };

  // Get tracking statistics
  const getTrackingStats = () => {
    if (videoResults) {
      return {
        uniqueDetections: videoResults.total_unique_detections || allDetections.length,
        frameDetections: videoResults.total_frame_detections || allDetections.length,
        duplicatesRemoved: (videoResults.total_frame_detections || allDetections.length) - (videoResults.total_unique_detections || allDetections.length)
      };
    }
    return {
      uniqueDetections: allDetections.length,
      frameDetections: allDetections.length,
      duplicatesRemoved: 0
    };
  };

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="video-defect-detection">
      <Row>
        <Col md={6}>
          <Card className="mb-4">
            <Card.Header className="bg-primary text-white">
              <h5 className="mb-0">Video Defect Detection</h5>
            </Card.Header>
            <Card.Body>
              {error && (
                <Alert variant="danger" className="mb-3">
                  {error}
                </Alert>
              )}

              {/* Model Selection */}
              <Form.Group className="mb-3">
                <Form.Label>Detection Model</Form.Label>
                <Form.Select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={isProcessing}
                >
                  {modelOptions.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>

              {/* Input Source Selection */}
              <Form.Group className="mb-3">
                <Form.Label>Input Source</Form.Label>
                <Form.Select
                  value={inputSource}
                  onChange={(e) => setInputSource(e.target.value)}
                  disabled={isProcessing}
                >
                  <option value="video">Video Upload</option>
                  <option value="camera">Live Camera Recording</option>
                </Form.Select>
              </Form.Group>

              {/* Video Upload */}
              {inputSource === 'video' && (
                <Form.Group className="mb-3">
                  <Form.Label>Upload Video</Form.Label>
                  <Form.Control
                    type="file"
                    accept="video/*"
                    onChange={handleVideoChange}
                    ref={fileInputRef}
                    disabled={isProcessing}
                  />
                  {videoPreview && (
                    <div className="mt-3">
                      <video
                        src={videoPreview}
                        controls
                        className="video-preview"
                        style={{ maxHeight: '200px' }}
                      />
                    </div>
                  )}
                </Form.Group>
              )}

              {/* Camera Recording */}
              {inputSource === 'camera' && (
                <div className="mb-3">
                  <div className="d-flex gap-2 mb-2">
                    <Button
                      variant={cameraActive ? "danger" : "info"}
                      onClick={toggleCamera}
                      disabled={isProcessing}
                    >
                      {cameraActive ? 'Stop Camera' : 'Start Camera'}
                    </Button>
                    {isMobile && cameraActive && (
                      <Button
                        variant="outline-secondary"
                        onClick={toggleCameraOrientation}
                        size="sm"
                      >
                        Rotate Camera
                      </Button>
                    )}
                  </div>

                  {cameraActive && (
                    <div className="webcam-container">
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
                      
                      <div className="mt-2">
                        {!isRecording ? (
                          <Button
                            variant="success"
                            onClick={handleStartRecording}
                            disabled={isProcessing}
                          >
                            Start Recording
                          </Button>
                        ) : (
                          <div className="d-flex align-items-center gap-2">
                            <Button
                              variant="danger"
                              onClick={handleStopRecording}
                            >
                              Stop Recording
                            </Button>
                            <span className="text-danger">
                              Recording: {formatTime(recordingTime)} / {formatTime(MAX_RECORDING_TIME)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {videoPreview && (
                    <div className="mt-3">
                      <video
                        src={videoPreview}
                        controls
                        className="video-preview"
                        style={{ maxHeight: '200px' }}
                      />
                    </div>
                  )}
                </div>
              )}

              {/* Action Buttons */}
              <div className="action-buttons">
                <Button
                  variant="primary"
                  onClick={handleProcess}
                  disabled={!isReadyForProcessing() || isProcessing}
                >
                  {loading ? (
                    <>
                      <Spinner size="sm" className="me-2" />
                      Processing...
                    </>
                  ) : (
                    'Process Video'
                  )}
                </Button>
                
                {isProcessing && (
                  <Button
                    variant="warning"
                    onClick={handleStopProcessing}
                  >
                    Stop Processing
                  </Button>
                )}
                
                <Button
                  variant="secondary"
                  onClick={handleReset}
                  disabled={isProcessing}
                >
                  Reset
                </Button>
              </div>

              {/* Processing Progress */}
              {isProcessing && (
                <div className="mt-3">
                  <div className="d-flex justify-content-between">
                    <span>Processing Progress:</span>
                    <span>{processingProgress.toFixed(1)}%</span>
                  </div>
                  <div className="progress mt-1">
                    <div
                      className="progress-bar"
                      role="progressbar"
                      style={{ width: `${processingProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          {/* Processed Video Display */}
          {processedVideo && (
            <Card className="mb-4">
              <Card.Header className="bg-success text-white">
                <h5 className="mb-0">Processed Video</h5>
              </Card.Header>
              <Card.Body>
                <div className="processed-video-container">
                  {isBuffering && (
                    <div className="processing-overlay">
                      <Spinner animation="border" />
                      <span className="ms-2">Buffering video...</span>
                    </div>
                  )}
                  
                  <img
                    src={processedVideo.startsWith('data:') ? processedVideo : `data:image/jpeg;base64,${processedVideo}`}
                    alt="Processed frame"
                    style={{ maxWidth: '100%' }}
                  />
                  
                  {/* Video Controls */}
                  {frameBuffer.length > 0 && (
                    <div className="video-controls mt-3">
                      <div className="d-flex gap-2 mb-2">
                        <Button size="sm" variant="outline-primary" onClick={handleRewind}>
                          ⏪
                        </Button>
                        <Button size="sm" variant="outline-primary" onClick={handlePlayPause}>
                          {isPlaying ? '⏸️' : '▶️'}
                        </Button>
                        <Button size="sm" variant="outline-primary" onClick={handleForward}>
                          ⏩
                        </Button>
                      </div>
                      
                      <Form.Range
                        min={0}
                        max={frameBuffer.length - 1}
                        value={currentFrameIndex}
                        onChange={(e) => setCurrentFrameIndex(Number(e.target.value))}
                      />
                      
                      <div className="text-center small text-muted">
                        Frame {currentFrameIndex + 1} of {frameBuffer.length}
                      </div>
                    </div>
                  )}
                </div>
              </Card.Body>
            </Card>
          )}

          {/* Detection Results Table */}
          {allDetections.length > 0 && (
            <Card>
              <Card.Header className="bg-info text-white">
                <h5 className="mb-0">Detection Results</h5>
              </Card.Header>
              <Card.Body>
                {/* Summary */}
                <div className="detection-summary mb-3">
                  <h6>Detection Summary:</h6>
                  <div className="mb-2">
                    {Object.entries(getDetectionSummary()).map(([type, count]) => (
                      <span key={type} className="badge bg-secondary me-1">
                        {type}: {count}
                      </span>
                    ))}
                  </div>
                  
                  {/* Tracking Statistics */}
                  <div className="tracking-stats">
                    <small className="text-muted">
                      <strong>Tracking Stats:</strong> {' '}
                      <span className="badge bg-success me-1">
                        Unique: {getTrackingStats().uniqueDetections}
                      </span>
                      <span className="badge bg-info me-1">
                        Total Frames: {getTrackingStats().frameDetections}
                      </span>
                      <span className="badge bg-warning">
                        Duplicates Removed: {getTrackingStats().duplicatesRemoved}
                      </span>
                    </small>
                  </div>
                </div>

                {/* Separate Tables for Each Defect Type */}
                {(() => {
                  const potholeDetections = allDetections.filter(d => d.type === 'Pothole');
                  const crackDetections = allDetections.filter(d => d.type.includes('Crack'));
                  const kerbDetections = allDetections.filter(d => d.type.includes('Kerb'));

                  return (
                    <div>
                      {/* Pothole Table - Show only if "All" or "Potholes" is selected */}
                      {(selectedModel === 'All' || selectedModel === 'Potholes') && (
                        <div className="defect-section potholes mb-4">
                          <h6 className="text-danger">
                            <span className="emoji">🕳️</span>
                            Potholes Detected: {potholeDetections.length}
                          </h6>
                          {potholeDetections.length > 0 ? (
                            <div className="detection-table-container">
                              <Table striped bordered hover size="sm">
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
                                  {potholeDetections.map((detection, index) => (
                                    <tr key={index}>
                                      <td>{detection.track_id || index + 1}</td>
                                      <td>{detection.area_cm2 ? detection.area_cm2.toFixed(2) : 'N/A'}</td>
                                      <td>{detection.depth_cm ? detection.depth_cm.toFixed(2) : 'N/A'}</td>
                                      <td>{detection.volume ? detection.volume.toFixed(2) : 'N/A'}</td>
                                      <td>{detection.volume_range || 'N/A'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </Table>
                            </div>
                          ) : (
                            <div className="no-defects-message">No potholes detected</div>
                          )}
                        </div>
                      )}

                      {/* Cracks Table - Show only if "All" or "Alligator Cracks" is selected */}
                      {(selectedModel === 'All' || selectedModel === 'Alligator Cracks') && (
                        <div className="defect-section cracks mb-4">
                          <h6 className="text-success">
                            <span className="emoji">🪨</span>
                            Cracks Detected: {crackDetections.length}
                          </h6>
                          {crackDetections.length > 0 ? (
                            <div className="detection-table-container">
                              <Table striped bordered hover size="sm">
                                <thead>
                                  <tr>
                                    <th>ID</th>
                                    <th>Type</th>
                                    <th>Area (cm²)</th>
                                    <th>Area Range</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {crackDetections.map((detection, index) => (
                                    <tr key={index}>
                                      <td>{detection.track_id || index + 1}</td>
                                      <td>{detection.type}</td>
                                      <td>{detection.area_cm2 ? detection.area_cm2.toFixed(2) : 'N/A'}</td>
                                      <td>{detection.area_range || 'N/A'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </Table>
                            </div>
                          ) : (
                            <div className="no-defects-message">No cracks detected</div>
                          )}
                        </div>
                      )}

                      {/* Kerbs Table - Show only if "All" or "Kerbs" is selected */}
                      {(selectedModel === 'All' || selectedModel === 'Kerbs') && (
                        <div className="defect-section kerbs mb-4">
                          <h6 className="text-primary">
                            <span className="emoji">🚧</span>
                            Kerbs Detected: {kerbDetections.length}
                          </h6>
                          {kerbDetections.length > 0 ? (
                            <div className="detection-table-container">
                              <Table striped bordered hover size="sm">
                                <thead>
                                  <tr>
                                    <th>ID</th>
                                    <th>Type</th>
                                    <th>Condition</th>
                                    <th>Length</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {kerbDetections.map((detection, index) => (
                                    <tr key={index}>
                                      <td>{detection.track_id || index + 1}</td>
                                      <td>{detection.kerb_type || 'Concrete Kerb'}</td>
                                      <td>{detection.condition || detection.type}</td>
                                      <td>{detection.length_m ? detection.length_m.toFixed(2) : 'N/A'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </Table>
                            </div>
                          ) : (
                            <div className="no-defects-message">No kerbs detected</div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })()}
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};

export default VideoDefectDetection; 