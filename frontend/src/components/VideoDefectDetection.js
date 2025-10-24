import React, { useState, useRef, useEffect, memo } from 'react';
import { Card, Button, Form, Alert, Spinner, Table, Row, Col, Modal } from 'react-bootstrap';
import MapPanel from './MapPanel';
import axios from 'axios';
import Webcam from 'react-webcam';
import useResponsive from '../hooks/useResponsive';
import './VideoDefectDetection.css';
import { validateUploadFile, showFileValidationError } from '../utils/fileValidation';

// Helper functions for GPS range checking
function lngLatToMeters(lat, lon) {
	const originShift = 2 * Math.PI * 6378137 / 2.0;
	const mx = lon * originShift / 180.0;
	let my = Math.log(Math.tan((90 + lat) * Math.PI / 360.0)) / (Math.PI / 180.0);
	my = my * originShift / 180.0;
	return { x: mx, y: my };
}

function pointToSegmentDistanceMeters(p, a, b) {
	const P = lngLatToMeters(p[0], p[1]);
	const A = lngLatToMeters(a[0], a[1]);
	const B = lngLatToMeters(b[0], b[1]);
	const vx = B.x - A.x;
	const vy = B.y - A.y;
	const wx = P.x - A.x;
	const wy = P.y - A.y;
	const c1 = vx * wx + vy * wy;
	if (c1 <= 0) return Math.hypot(P.x - A.x, P.y - A.y);
	const c2 = vx * vx + vy * vy;
	if (c2 <= c1) return Math.hypot(P.x - B.x, P.y - B.y);
	const t = c1 / c2;
	const projx = A.x + t * vx;
	const projy = A.y + t * vy;
	return Math.hypot(P.x - projx, P.y - projy);
}


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
  const [fromLoc, setFromLoc] = useState(null);
  const [toLoc, setToLoc] = useState(null);
  const [gpsTrack, setGpsTrack] = useState([]);
  const [routes, setRoutes] = useState([]);
  const [routeReady, setRouteReady] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraOrientation, setCameraOrientation] = useState('environment');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const recordedChunksRef = useRef([]); // <-- Add this line

  // Add new state for live camera location handling
  const [liveFromLoc, setLiveFromLoc] = useState('');
  const [liveToLoc, setLiveToLoc] = useState('');
  const [liveCurrentLoc, setLiveCurrentLoc] = useState(null);
  const [locationUpdated, setLocationUpdated] = useState(false);
  const liveLocationIntervalRef = useRef(null);

  // Video processing states
  const [frameBuffer, setFrameBuffer] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isBuffering, setIsBuffering] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [allDetections, setAllDetections] = useState([]);
  const [videoResults, setVideoResults] = useState(null);

  // Add new state for frame-by-frame updates
  const [currentDetections, setCurrentDetections] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const streamRef = useRef(null);
  const [showDronePopup, setShowDronePopup] = useState(false);

  useEffect(() => {
    if (inputSource === 'camera') {
        if (liveFromLoc && liveToLoc) {
            const fromCoords = liveFromLoc.split(',').map(c => parseFloat(c.trim()));
            const toCoords = liveToLoc.split(',').map(c => parseFloat(c.trim()));
            if (fromCoords.length === 2 && toCoords.length === 2 && !fromCoords.some(isNaN) && !toCoords.some(isNaN)) {
                setFromLoc({ lat: fromCoords[0], lon: fromCoords[1] });
                setToLoc({ lat: toCoords[0], lon: toCoords[1] });
                setGpsTrack([fromCoords, toCoords]);
                setRouteReady(true);
            }
        }
        return;
    }
    if (!gpsTrack || gpsTrack.length < 2) {
      setRouteReady(false);
      return;
    }

    const validPoints = gpsTrack.filter((point) => Array.isArray(point) && point.length === 2 &&
      typeof point[0] === 'number' && typeof point[1] === 'number' &&
      !Number.isNaN(point[0]) && !Number.isNaN(point[1]) &&
      point[0] >= -90 && point[0] <= 90 && point[1] >= -180 && point[1] <= 180
    );

    if (validPoints.length < 2) {
      setRouteReady(false);
      return;
    }

    const newFrom = validPoints[0];
    const newTo = validPoints[validPoints.length - 1];
    const formattedFrom = { lat: newFrom[0].toFixed(6), lon: newFrom[1].toFixed(6) };
    const formattedTo = { lat: newTo[0].toFixed(6), lon: newTo[1].toFixed(6) };

    const hasChanged =
      !fromLoc || !toLoc ||
      fromLoc.lat !== formattedFrom.lat || fromLoc.lon !== formattedFrom.lon ||
      toLoc.lat !== formattedTo.lat || toLoc.lon !== formattedTo.lon;

    if (hasChanged) {
      setFromLoc(formattedFrom);
      setToLoc(formattedTo);
    }

    setRouteReady(true);
  }, [gpsTrack, fromLoc, toLoc, inputSource, liveFromLoc, liveToLoc]);

  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const { isMobile } = useResponsive();

  const BUFFER_SIZE = 100; // Increased buffer size for smoother playback
  const PLAYBACK_FPS = 15;
  const MAX_RECORDING_TIME = 30; // 30 seconds limit for live camera

  const [totalFrames, setTotalFrames] = useState(null);
  const totalFramesValid = Number.isFinite(totalFrames) && totalFrames > 0;

  const [videoDuration, setVideoDuration] = useState(null);
  const [videoFPS, setVideoFPS] = useState(30); // Default FPS

  // Available models
  const modelOptions = [
    { value: 'All', label: 'All (detect all types of defects)' },
    { value: 'Potholes', label: 'Potholes' },
    { value: 'Alligator Cracks', label: 'Alligator Cracks' },
    { value: 'Kerbs', label: 'Kerbs' }
  ];

  // Get user location
  useEffect(() => {
    if (inputSource === 'camera' && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setCoordinates(`${latitude.toFixed(6)}, ${longitude.toFixed(6)}`);
          // Initialize From default once
          if (!fromLoc) setFromLoc({ lat: latitude.toFixed(6), lon: longitude.toFixed(6) });
        },
        (err) => {
          console.error("Error getting location:", err);
          setCoordinates('Location unavailable');
        }
      );
    }
  }, [inputSource]);

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

  // When videoPreview is set, extract duration and FPS
  useEffect(() => {
    if (videoPreview) {
      const video = document.createElement('video');
      video.src = videoPreview;
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        setVideoDuration(video.duration);
        // Try to get FPS from video tracks if available
        if (video.webkitVideoDecodedByteCount !== undefined) {
          // Not standard, but some browsers may expose frameRate
          try {
            const tracks = video.videoTracks || (video.captureStream && video.captureStream().getVideoTracks());
            if (tracks && tracks.length > 0 && tracks[0].getSettings) {
              const settings = tracks[0].getSettings();
              if (settings.frameRate) {
                setVideoFPS(settings.frameRate);
              }
            }
          } catch (e) {}
        }
      };
    }
  }, [videoPreview]);

  // Helper to estimate total frames if backend total_frames is invalid
  const estimatedTotalFrames = videoDuration && videoFPS ? Math.round(videoDuration * videoFPS) : null;

  // Add cleanup effect for SSE stream
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.abort();
      }
    };
  }, []);

  const [warning, setWarning] = useState('');
  const [showRangeModal, setShowRangeModal] = useState(false);
  const [rangeMessage, setRangeMessage] = useState('');

  // Handle video file selection
  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate video file first
      const validation = validateUploadFile(file, 'video', 'video_defect_detection');
      if (!validation.isValid) {
        showFileValidationError(validation.errorMessage, setError);
        // Clear the file input
        if (e.target) {
          e.target.value = '';
        }
        return;
      }

      // Clear any previous errors
      setError('');

      const video = document.createElement('video');
      video.preload = 'metadata';
      video.onloadedmetadata = () => {
        if (video.duration > 60) {
          setWarning('Video upload is restricted to 1 minute. Please select a shorter video.');
          setVideoFile(null);
          setVideoPreview(null);
          setProcessedVideo(null);
          setVideoResults(null);
          setAllDetections([]);
          setError('');
          if (fileInputRef.current) fileInputRef.current.value = '';
        } else {
          setWarning('');
          setVideoFile(file);
          setVideoPreview(URL.createObjectURL(file));
          setProcessedVideo(null);
          setVideoResults(null);
          setAllDetections([]);
          setError('');
        }
      };
      video.src = URL.createObjectURL(file);
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
    if (inputSource === 'camera') {
      if (!liveFromLoc || !liveToLoc) {
        setError('Please enter both "From" and "To" coordinates for live camera mode.');
        return;
      }
      const fromCoords = liveFromLoc.split(',').map(c => parseFloat(c.trim()));
      const toCoords = liveToLoc.split(',').map(c => parseFloat(c.trim()));

      if (fromCoords.length !== 2 || toCoords.length !== 2 || fromCoords.some(isNaN) || toCoords.some(isNaN)) {
        setError('Invalid coordinate format. Please use "latitude, longitude".');
        return;
      }

      setFromLoc({ lat: fromCoords[0], lon: fromCoords[1] });
      setToLoc({ lat: toCoords[0], lon: toCoords[1] });
      setGpsTrack([fromCoords, toCoords]);
      setRouteReady(true);
    }

    console.log('handleStartRecording called');
    if (!webcamRef.current || !webcamRef.current.stream) {
      setError('Camera not available');
      return;
    }

    try {
      setRecordedChunks([]);
      recordedChunksRef.current = [];
      setRecordingTime(0);
      setIsRecording(true);
      setError('');

      const mediaRecorder = new MediaRecorder(webcamRef.current.stream, {
        mimeType: 'video/webm'
      });

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        console.log('ondataavailable fired, size:', event.data.size);
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        console.log('mediaRecorder.onstop fired');
        // Use the ref to get the latest chunks
        const chunks = recordedChunksRef.current;
        // Debug logs
        console.log('onstop: recordedChunks length:', chunks.length);
        let totalSize = 0;
        chunks.forEach((c, i) => {
          console.log(`Chunk ${i} size:`, c.size);
          totalSize += c.size;
        });
        console.log('Total recorded size:', totalSize);
        const blob = new Blob(chunks, { type: 'video/webm' });
        const file = new File([blob], `recorded_video_${Date.now()}.webm`, { type: 'video/webm' });
        setVideoFile(file);
        setVideoPreview(URL.createObjectURL(blob));
        setIsRecording(false);
        setRecordingTime(0);
        // Reset the ref and state for next recording
        recordedChunksRef.current = [];
        setRecordedChunks([]);
        if (liveLocationIntervalRef.current) {
          clearInterval(liveLocationIntervalRef.current);
        }
      };

      mediaRecorder.onstart = () => {
        console.log('mediaRecorder.onstart fired');
        if (inputSource === 'camera') {
          let frame = 0;
          liveLocationIntervalRef.current = setInterval(() => {
            frame++;
            const progress = frame / (MAX_RECORDING_TIME / 5);
            const fromCoords = liveFromLoc.split(',').map(c => parseFloat(c.trim()));
            const toCoords = liveToLoc.split(',').map(c => parseFloat(c.trim()));
            const lat = fromCoords[0] + (toCoords[0] - fromCoords[0]) * progress;
            const lon = fromCoords[1] + (toCoords[1] - fromCoords[1]) * progress;
            const newCoord = [lat, lon];
            setLiveCurrentLoc(newCoord);
            setGpsTrack(prev => [...prev, newCoord]);
            setLocationUpdated(false);
            setTimeout(() => setLocationUpdated(true), 1000);

            // Check if out of range
            const path = [fromCoords, toCoords];
            const halfWidthM = 50; // 50 meters tolerance
            const d = pointToSegmentDistanceMeters(newCoord, path[0], path[1]);
            if (d > halfWidthM) {
              setRangeMessage('Range exceeded, please stay within the specified coordinates.');
              setShowRangeModal(true);
            }

          }, 5000);
        } 
      };
      mediaRecorder.onerror = (e) => {
        console.error('mediaRecorder.onerror', e);
      };

      console.log('Calling mediaRecorder.start(1000)');
      mediaRecorder.start(1000); // timeslice: 1000ms
    } catch (error) {
      setError('Failed to start recording: ' + error.message);
      setIsRecording(false);
    }
  };

  // Stop recording
  const handleStopRecording = () => {
    console.log('handleStopRecording called');
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingTime(0);
      // Stop GPS interval
      const intervalId = mediaRecorderRef.current.__gpsIntervalId;
      if (intervalId) {
        clearInterval(intervalId);
        mediaRecorderRef.current.__gpsIntervalId = null;
      }
      if (liveLocationIntervalRef.current) {
        clearInterval(liveLocationIntervalRef.current);
      }
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

    if (inputSource === 'camera' && (!fromLoc || !toLoc)) {
      setError('Waiting for GPS track to determine route. Please ensure location access is enabled.');
      return;
    }

    // Reset states
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
    setCurrentDetections([]);
    setProcessedVideo(null);
    setVideoResults(null);
    setShowResults(false);

    // Create abort controller for cleanup
    streamRef.current = new AbortController();

    try {
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('selectedModel', selectedModel);
      if (inputSource === 'camera') {
        formData.append('coordinates', coordinates);
        if (fromLoc && toLoc) {
          formData.append('fromLocation', JSON.stringify(fromLoc));
          formData.append('toLocation', JSON.stringify(toLoc));
        }
        if (gpsTrack && gpsTrack.length > 0) {
          formData.append('gpsTrack', JSON.stringify(gpsTrack));
        }
      }
      const userString = sessionStorage.getItem('user');
      const user = userString ? JSON.parse(userString) : null;
      formData.append('username', user?.username || 'Unknown');
      formData.append('role', user?.role || 'Unknown');

      console.log('Starting video processing with model:', selectedModel);

      const sseUrl = '/api/pavement/detect-video';
      
      const response = await fetch(sseUrl, {
        method: 'POST',
        body: formData,
        signal: streamRef.current.signal
      });

      if (!response.ok) {
        let serverMsg = `HTTP error! status: ${response.status}`;
        try {
          const errData = await response.json();
          if (errData && errData.code === 'GPS_VALIDATION_FAILED') {
            setRangeMessage('You are outside the defined range — please return within the route.');
            setShowRangeModal(true);
            setIsProcessing(false);
            setLoading(false);
            return;
          } else if (errData && (errData.code === 'GPS_INPUT_INVALID' || errData.code === 'GPS_INPUT_MISSING')) {
            serverMsg = errData.message || serverMsg;
          } else if (errData && errData.message) {
            serverMsg = errData.message;
          }
        } catch (e) {
          // Fallback to text if JSON parsing fails
          try {
            const txt = await response.text();
            if (txt) serverMsg = txt;
          } catch (_) {}
        }
        throw new Error(serverMsg);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      // Helper to accumulate detections
      const appendDetections = (detections) => {
        if (detections && Array.isArray(detections) && detections.length > 0) {
          setAllDetections(prev => [...prev, ...detections].slice(-100)); // Limit detections to 100
        }
      };

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

            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            // Keep the last line in buffer if it's incomplete
            buffer = lines.pop();

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.substring(6));
                  // Debug log for every SSE message
                  console.log('SSE data:', data);

                  // Handle error case
                  if (data.success === false) {
                    setError(data.message || 'Video processing failed');
                    setIsProcessing(false);
                    setLoading(false);
                    setIsBuffering(false);
                    return;
                  }

                  // Update progress immediately
                  if (data.progress !== undefined && totalFramesValid) {
                    setProcessingProgress(data.progress);
                    if (!showResults && data.progress > 0) {
                      setShowResults(true);
                    }
                  } else if (data.frame_count !== undefined && totalFramesValid) {
                    // Calculate progress if not provided but backend totalFrames is valid
                    const progress = (data.frame_count / totalFrames) * 100;
                    setProcessingProgress(progress);
                    if (!showResults && progress > 0) {
                      setShowResults(true);
                    }
                  } else if (data.frame_count !== undefined && estimatedTotalFrames) {
                    // Fallback: use estimated total frames from duration and FPS
                    const progress = (data.frame_count / estimatedTotalFrames) * 100;
                    setProcessingProgress(progress);
                    if (!showResults && progress > 0) {
                      setShowResults(true);
                    }
                  }

                  // Update frame display immediately
                  if (data.frame && typeof data.frame === 'string' && data.frame.length > 1000) {
                    setFrameBuffer(prev => [...prev, data.frame].slice(-BUFFER_SIZE)); // Limit frame buffer
                    setProcessedVideo(data.frame);
                    setCurrentFrameIndex(prev => prev + 1);
                    if (isBuffering) {
                      setIsBuffering(false);
                    }
                  }

                  // Accumulate detections per frame
                  if (data.detections && data.detections.length > 0) {
                    setCurrentDetections(data.detections);
                    appendDetections(data.detections);
                  }

                  // Handle final results
                  if (data.all_detections) {
                    setVideoResults(data);
                    setAllDetections(data.all_detections);
                    // If backend returns a GPS track, update the state to draw it on the map
                    if (data.gps_track && Array.isArray(data.gps_track) && data.gps_track.length > 0) {
                      setGpsTrack(data.gps_track);
                    }
                    setIsProcessing(false);
                    setLoading(false);
                    setIsBuffering(false);
                    setProcessingProgress(100);
                    setCurrentFrameIndex(0);
                    setIsPlaying(false);
                    console.log('Video processing completed');
                    return;
                  }

                  // Handle end signal
                  if (data.end) {
                    console.log('Received end signal');
                    setIsProcessing(false);
                    setLoading(false);
                    setIsBuffering(false);
                    return;
                  }

                  // Update totalFrames when receiving SSE data:
                  if (data.total_frames !== undefined) {
                    setTotalFrames(data.total_frames);
                  }
                } catch (parseError) {
                  console.warn('Error parsing SSE data:', parseError);
                }
              }
            }
          }
        } catch (streamError) {
          if (streamError.name === 'AbortError') {
            console.log('Stream aborted by user');
          } else {
            console.error('Stream processing error:', streamError);
            setError('Error processing video stream');
          }
          setIsProcessing(false);
          setLoading(false);
          setIsBuffering(false);
        } finally {
          if (reader) {
            try {
              reader.releaseLock();
            } catch (e) {
              console.warn('Error releasing reader lock:', e);
            }
          }
        }
      };

      await processStream();
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
    setCurrentDetections([]);
    setFrameBuffer([]);
    setCurrentFrameIndex(0);
    setIsProcessing(false);
    setShouldStop(false);
    setIsBuffering(false);
    setIsPlaying(false);
    setProcessingProgress(0);
    setError('');
    setSelectedModel('All');
    setShowResults(false); // <-- Ensure table is hidden after reset
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setLiveFromLoc('');
    setLiveToLoc('');
    setLiveCurrentLoc(null);
    setGpsTrack([]);
    setRouteReady(false);
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
            {warning && (
              <Alert variant="warning" className="mb-3">
                {warning}
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
                      key={videoPreview}
                    />
                  </div>
                )}
              </Form.Group>
            )}

            {/* Input Source Selection */}
            <Form.Group className="mb-3">
              <Form.Label>Input Source</Form.Label>
              <Form.Select
                value={inputSource}
                onChange={(e) => {
                  const value = e.target.value;
                  if (value === 'drone') {
                    setShowDronePopup(true);
                  } else {
                    setInputSource(value);
                  }
                }}
                disabled={isProcessing}
              >
                <option value="video">Upload Video</option>
                <option value="camera">Live Camera Recording</option>
                <option value="drone">Drone View</option>
              </Form.Select>
            </Form.Group>

            {/* Camera Recording */}
            {inputSource === 'camera' && (
              <div className="mb-3">
                {/* From/To Location Inputs for Live Camera */}
                <Row className="mb-3">
                  <Col md={6}>
                    <Form.Group>
                      <Form.Label>From Coordinates (Lat, Lng)</Form.Label>
                      <Form.Control
                        type="text"
                        placeholder="e.g., 1.290270, 103.851959"
                        value={liveFromLoc}
                        onChange={(e) => setLiveFromLoc(e.target.value)}
                        disabled={isRecording || isProcessing}
                      />
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group>
                      <Form.Label>To Coordinates (Lat, Lng)</Form.Label>
                      <Form.Control
                        type="text"
                        placeholder="e.g., 1.352083, 103.819836"
                        value={liveToLoc}
                        onChange={(e) => setLiveToLoc(e.target.value)}
                        disabled={isRecording || isProcessing}
                      />
                    </Form.Group>
                  </Col>
                </Row>

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
                      key={videoPreview}
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

            {/* Always show progress during processing */}
            {isProcessing && (
              <div className="mt-3">
                {(totalFramesValid || estimatedTotalFrames) && processingProgress > 0 && Number.isFinite(processingProgress) ? (
                  <>
                    <div className="d-flex justify-content-between">
                      <span>Processing Progress:</span>
                      <span>{Math.max(0, Math.min(100, processingProgress)).toFixed(1)}%</span>
                    </div>
                    <div className="progress mt-1">
                      <div
                        className="progress-bar progress-bar-striped progress-bar-animated"
                        role="progressbar"
                        style={{ width: `${Math.max(0, Math.min(100, processingProgress))}%` }}
                        aria-valuenow={Math.max(0, Math.min(100, processingProgress))}
                        aria-valuemin="0"
                        aria-valuemax="100"
                      ></div>
                    </div>
                  </>
                ) : (
                  <div className="d-flex align-items-center mt-3">
                    <span>Processing...</span>
                    <div className="progress flex-grow-1 ms-2" style={{ height: '20px' }}>
                      <div
                        className="progress-bar progress-bar-striped progress-bar-animated"
                        role="progressbar"
                        style={{ width: `100%`, backgroundColor: '#e0e0e0' }}
                        aria-valuenow={0}
                        aria-valuemin="0"
                        aria-valuemax="100"
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </Card.Body>
        </Card>

        {(inputSource === 'camera' ? routeReady : true) && 
            <MapPanel
              panelTitle="Video Route Map"
              fromLocation={fromLoc}
              toLocation={toLoc}
              onFromChange={setFromLoc}
              onToChange={setToLoc}
              onRoutesLoaded={setRoutes}
              gpsTrack={gpsTrack}
              onOutOfRange={() => {
                setRangeMessage('You are outside the defined range — please return within the route.');
                setShowRangeModal(true);
              }}
            />
        }

        {/* Live Location Update Display */}
        {inputSource === 'camera' && isRecording && (
          <Card className="mt-4">
            <Card.Body>
                <Row className="align-items-center">
                    <Col>
                        <strong>Location:</strong>
                        {liveFromLoc && <span> From: {liveFromLoc}</span>}
                        {liveToLoc && <span> To: {liveToLoc}</span>}
                        {liveCurrentLoc ? (
                        <span> Current: {` ${liveCurrentLoc[0].toFixed(6)}, ${liveCurrentLoc[1].toFixed(6)}`}
                        </span>
                        ) : (
                        <span> Waiting for update...</span>
                        )}
                    </Col>
                    <Col xs="auto">
                        {locationUpdated ? (
                        <span className="text-success">✓ Updated</span>
                        ) : (
                        <Spinner animation="border" size="sm" role="status">
                            <span className="visually-hidden">Loading...</span>
                        </Spinner>
                        )}
                    </Col>
                </Row>
            </Card.Body>
          </Card>
        )}

        {/* Show detection results as soon as we have any, and always after processing is complete if results exist */}
        {((showResults || allDetections.length > 0 || (!isProcessing && videoResults && allDetections.length > 0))) && (
          <Card className="mt-4">
            <Card.Header className="bg-info text-white">
              <h5 className="mb-0">Detection Results</h5>
              {isProcessing && (
                <small className="text-white-50">
                  Results update in real-time as processing continues...
                </small>
              )}
              {!isProcessing && videoResults && (
                <small className="text-success">
                  <b>Processing Complete.</b> Final results are shown below.
                </small>
              )}
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

      {/* Range Exceeded Modal */}
      <Modal show={showRangeModal} onHide={() => setShowRangeModal(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Out of Route</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {rangeMessage || 'You are outside the defined range — please return within the route.'}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="primary" onClick={() => setShowRangeModal(false)}>
            OK
          </Button>
        </Modal.Footer>
      </Modal>

      <Modal show={showDronePopup} onHide={() => setShowDronePopup(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Feature Update</Modal.Title>
        </Modal.Header>
        <Modal.Body>This feature will be updated soon.</Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowDronePopup(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};

export default memo(VideoDefectDetection);