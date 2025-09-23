import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Container, Row, Col, Card, Button, Spinner, Alert, Badge } from 'react-bootstrap';
import axios from 'axios';
import './dashboard.css';

/**
 * Image URL Resolution Logic - EXACT SAME AS DASHBOARD
 * Handles both S3 URLs (new data) and GridFS IDs (legacy data)
 */
const getImageUrlForDisplay = (imageData, imageType = 'original') => {
  console.log('DefectDetail getImageUrlForDisplay called:', { imageData, imageType });

  if (!imageData) {
    console.log('No imageData provided');
    return null;
  }

  // Check if this is video data with representative frame
  if (imageData.media_type === 'video' && imageData.representative_frame) {
    console.log('Using representative frame for video data');
    return `data:image/jpeg;base64,${imageData.representative_frame}`;
  }

  // Try S3 full URL first (new images with pre-generated URLs) - proxy through backend
  const fullUrlField = `${imageType}_image_full_url`;
  if (imageData[fullUrlField]) {
    console.log('Using full URL field:', fullUrlField, imageData[fullUrlField]);
    // Extract S3 key from full URL and use proxy endpoint
    const urlParts = imageData[fullUrlField].split('/');
    const bucketIndex = urlParts.findIndex(part => part.includes('.s3.'));
    if (bucketIndex !== -1 && bucketIndex + 1 < urlParts.length) {
      const s3Key = urlParts.slice(bucketIndex + 1).join('/');
      const proxyUrl = `/api/pavement/get-s3-image/${encodeURIComponent(s3Key)}`;
      console.log('Generated proxy URL from full URL:', proxyUrl);
      return proxyUrl;
    }
  }

  // Try S3 key with proxy endpoint (new images without full URL)
  const s3KeyField = `${imageType}_image_s3_url`;
  if (imageData[s3KeyField]) {
    console.log('Using S3 key field:', s3KeyField, imageData[s3KeyField]);

    // Properly encode the S3 key for URL path
    const s3Key = imageData[s3KeyField];
    const encodedKey = s3Key.split('/').map(part => encodeURIComponent(part)).join('/');
    const url = `/api/pavement/get-s3-image/${encodedKey}`;

    console.log('Generated proxy URL from S3 key:', url);
    console.log('Original S3 key:', s3Key);
    console.log('Encoded S3 key:', encodedKey);

    return url;
  }

  // Fall back to GridFS endpoint (legacy images)
  const gridfsIdField = `${imageType}_image_id`;
  if (imageData[gridfsIdField]) {
    console.log('Using GridFS field:', gridfsIdField, imageData[gridfsIdField]);
    const url = `/api/pavement/get-image/${imageData[gridfsIdField]}`;
    console.log('Generated GridFS URL:', url);
    return url;
  }

  // No image URL available
  console.log('No image URL available for:', imageType, imageData);
  return null;
};

/**
 * Enhanced Image Display Component - EXACT SAME AS DASHBOARD
 * Supports both original and processed images with toggle
 */
const EnhancedDefectImageDisplay = ({ imageData, imageType, defectType }) => {
  const [currentImageUrl, setCurrentImageUrl] = useState(null);
  const [hasError, setHasError] = useState(false);
  const [fallbackAttempts, setFallbackAttempts] = useState(0);

  useEffect(() => {
    // Reset state when imageData changes
    setHasError(false);
    setFallbackAttempts(0);

    // Get initial image URL using Dashboard logic
    const imageUrl = getImageUrlForDisplay(imageData, imageType);

    // Debug logging
    console.log('DefectDetail EnhancedImageDisplay Debug:', {
      imageType,
      imageData,
      generatedUrl: imageUrl,
      s3KeyField: `${imageType}_image_s3_url`,
      s3KeyValue: imageData?.[`${imageType}_image_s3_url`],
      fullUrlField: `${imageType}_image_full_url`,
      fullUrlValue: imageData?.[`${imageType}_image_full_url`]
    });

    setCurrentImageUrl(imageUrl);
  }, [imageData, imageType]);

  const getFallbackImageUrl = (imageData, imageType) => {
    console.log('🔄 Getting fallback URL for:', imageType, imageData);

    // Try direct S3 URL if we have the full URL field
    const fullUrlField = `${imageType}_image_full_url`;
    if (imageData[fullUrlField]) {
      console.log('🔄 Trying direct S3 URL:', imageData[fullUrlField]);
      return imageData[fullUrlField];
    }

    // Try GridFS if S3 failed
    const gridfsIdField = `${imageType}_image_id`;
    if (imageData[gridfsIdField]) {
      console.log('🔄 Trying GridFS URL:', imageData[gridfsIdField]);
      return `/api/pavement/get-image/${imageData[gridfsIdField]}`;
    }

    // Try alternative S3 proxy with different encoding
    const s3KeyField = `${imageType}_image_s3_url`;
    if (imageData[s3KeyField]) {
      console.log('🔄 Trying alternative S3 proxy encoding');
      const s3Key = imageData[s3KeyField];
      const alternativeUrl = `/api/pavement/get-s3-image/${encodeURIComponent(s3Key)}`;
      console.log('🔄 Alternative proxy URL:', alternativeUrl);
      return alternativeUrl;
    }

    console.log('❌ No fallback URL available');
    return null;
  };

  const handleImageError = () => {
    console.error('❌ DefectDetail image loading failed:', currentImageUrl);

    // Try fallback URLs
    if (fallbackAttempts === 0) {
      const fallbackUrl = getFallbackImageUrl(imageData, imageType);

      if (fallbackUrl && fallbackUrl !== currentImageUrl) {
        console.log('🔄 Trying fallback URL:', fallbackUrl);
        setCurrentImageUrl(fallbackUrl);
        setFallbackAttempts(1);
        return;
      }
    }

    if (fallbackAttempts === 1) {
      // Second fallback: try alternative image type
      const alternativeType = imageType === 'original' ? 'processed' : 'original';
      const alternativeUrl = getImageUrlForDisplay(imageData, alternativeType);

      if (alternativeUrl && alternativeUrl !== currentImageUrl) {
        console.log('🔄 Trying alternative image type:', alternativeType);
        setCurrentImageUrl(alternativeUrl);
        setFallbackAttempts(2);
        return;
      }
    }

    // All fallbacks exhausted
    console.error('❌ All fallbacks exhausted for DefectDetail image');
    setHasError(true);
  };

  // Clean error state - same as Dashboard
  if (hasError || !currentImageUrl) {
    return (
      <div className="text-muted d-flex align-items-center justify-content-center" style={{ minHeight: '200px' }}>
        <div className="text-center">
          <i className="fas fa-image-slash fa-2x mb-2"></i>
          <div>No image available</div>
          {fallbackAttempts > 0 && (
            <small className="text-warning d-block mt-1">
              (Tried {fallbackAttempts} fallback{fallbackAttempts > 1 ? 's' : ''})
            </small>
          )}
        </div>
      </div>
    );
  }

  // Clean image display - same as Dashboard
  return (
    <div className="mb-3">
      <div className="text-center">
        <img
          src={currentImageUrl}
          alt={`${imageType} defect image`}
          className="img-fluid border rounded"
          style={{ maxHeight: '400px', maxWidth: '100%' }}
          onError={handleImageError}
          onLoad={() => {
            console.log('✅ DefectDetail image loaded successfully:', currentImageUrl);
          }}
          loading="lazy"
        />

        {/* Image Type Label */}
        <div className="mt-2">
          <small className="text-primary fw-bold">
            📷 {imageType === 'original' ? 'Original' : 'Processed'} Image
          </small>
          {fallbackAttempts > 0 && (
            <div>
              <small className="text-warning">(Fallback source)</small>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

function DefectDetail() {
  const { imageId } = useParams();
  const [defectData, setDefectData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [imageType, setImageType] = useState('processed'); // 'original' or 'processed'

  useEffect(() => {
    const fetchDefectDetail = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`/api/pavement/images/${imageId}`);
        
        if (response.data.success) {
          setDefectData(response.data);
        } else {
          setError('Failed to load defect details');
        }
        setLoading(false);
      } catch (err) {
        console.error('Error fetching defect details:', err);
        setError(`Error loading defect details: ${err.message}`);
        setLoading(false);
      }
    };

    if (imageId) {
      fetchDefectDetail();
    }
  }, [imageId]);

  const toggleImageType = () => {
    setImageType(prev => prev === 'original' ? 'processed' : 'original');
  };

  const getDefectTypeLabel = (type) => {
    switch (type) {
      case 'pothole':
        return <Badge bg="danger">Pothole</Badge>;
      case 'crack':
        return <Badge bg="warning" text="dark">Crack</Badge>;
      case 'kerb':
        return <Badge bg="primary">Kerb</Badge>;
      default:
        return <Badge bg="secondary">{type}</Badge>;
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  return (
    <Container className="py-4">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2>Defect Detail</h2>
        <Link to="/dashboard" className="btn btn-outline-primary">
          Back to Dashboard
        </Link>
      </div>

      {loading ? (
        <div className="text-center py-5">
          <Spinner animation="border" role="status" variant="primary">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
        </div>
      ) : error ? (
        <Alert variant="danger">{error}</Alert>
      ) : defectData ? (
        <>
          <Card className="shadow-sm mb-4">
            <Card.Header className="bg-primary text-white">
              <div className="d-flex justify-content-between align-items-center">
                <h5 className="mb-0">
                  {getDefectTypeLabel(defectData.type)} - ID: {imageId}
                </h5>
                {/* Only show Original/Processed buttons for non-video entries */}
                {defectData.image.media_type !== 'video' && (
                  <div>
                    <Button
                      variant={imageType === 'original' ? 'light' : 'outline-light'}
                      size="sm"
                      className="me-2"
                      onClick={toggleImageType}
                    >
                      Original
                    </Button>
                    <Button
                      variant={imageType === 'processed' ? 'light' : 'outline-light'}
                      size="sm"
                      onClick={toggleImageType}
                    >
                      Processed
                    </Button>
                  </div>
                )}
              </div>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={6} className="text-center mb-4">
                  {/* Toggle Buttons - Same as Dashboard */}
                  <div className="d-flex justify-content-center mb-3">
                    <div className="btn-group btn-group-sm" role="group">
                      <button
                        type="button"
                        className={`btn ${imageType === 'processed' ? 'btn-primary' : 'btn-outline-primary'}`}
                        onClick={() => setImageType('processed')}
                      >
                        Processed
                      </button>
                      <button
                        type="button"
                        className={`btn ${imageType === 'original' ? 'btn-primary' : 'btn-outline-primary'}`}
                        onClick={() => setImageType('original')}
                      >
                        Original
                      </button>
                    </div>
                  </div>

                  <EnhancedDefectImageDisplay
                    imageData={defectData.image}
                    imageType={imageType}
                    defectType={defectData.type}
                  />
                </Col>
                <Col md={6}>
                  <h5>Basic Information</h5>
                  <table className="table table-bordered">
                    <tbody>
                      <tr>
                        <th width="40%">Type</th>
                        <td>{defectData.type}</td>
                      </tr>
                      <tr>
                        <th>Defect Count</th>
                        <td>
                          {defectData.image.pothole_count || 
                           defectData.image.crack_count || 
                           defectData.image.kerb_count || 'N/A'}
                        </td>
                      </tr>
                      <tr>
                        <th>Date Detected</th>
                        <td>{formatDate(defectData.image.timestamp)}</td>
                      </tr>
                      <tr>
                        <th>Reported By</th>
                        <td>{defectData.image.username || 'N/A'}</td>
                      </tr>
                      <tr>
                        <th>Role</th>
                        <td>{defectData.image.role || 'N/A'}</td>
                      </tr>
                      <tr>
                        <th>Coordinates</th>
                        <td>{defectData.image.coordinates || 'Not Available'}</td>
                      </tr>
                      <tr>
                        <th>Media Type</th>
                        <td>
                          {defectData.image.media_type === 'video' ? (
                            <span className="text-info">📹 Video</span>
                          ) : (
                            <span className="text-primary">📷 Image</span>
                          )}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </Col>
              </Row>

              {/* EXIF and Metadata Information */}
              {(defectData.image.exif_data || defectData.image.metadata) && (
                <Row className="mt-4">
                  <Col>
                    <Card>
                      <Card.Header>
                        <h5 className="mb-0">📊 Media Information</h5>
                      </Card.Header>
                      <Card.Body>
                        <Row>
                          {/* Camera Information */}
                          {defectData.image.exif_data?.camera_info && Object.keys(defectData.image.exif_data.camera_info).length > 0 && (
                            <Col md={6} className="mb-3">
                              <h6 className="text-primary">📷 Camera Information</h6>
                              <table className="table table-sm">
                                <tbody>
                                  {defectData.image.exif_data.camera_info.camera_make && (
                                    <tr>
                                      <th width="40%">Make:</th>
                                      <td>{defectData.image.exif_data.camera_info.camera_make}</td>
                                    </tr>
                                  )}
                                  {defectData.image.exif_data.camera_info.camera_model && (
                                    <tr>
                                      <th>Model:</th>
                                      <td>{defectData.image.exif_data.camera_info.camera_model}</td>
                                    </tr>
                                  )}
                                  {defectData.image.exif_data.camera_info.software && (
                                    <tr>
                                      <th>Software:</th>
                                      <td>{defectData.image.exif_data.camera_info.software}</td>
                                    </tr>
                                  )}
                                </tbody>
                              </table>
                            </Col>
                          )}

                          {/* Technical Information */}
                          {defectData.image.exif_data?.technical_info && Object.keys(defectData.image.exif_data.technical_info).length > 0 && (
                            <Col md={6} className="mb-3">
                              <h6 className="text-success">⚙️ Technical Details</h6>
                              <table className="table table-sm">
                                <tbody>
                                  {defectData.image.exif_data.technical_info.iso && (
                                    <tr>
                                      <th width="40%">ISO:</th>
                                      <td>{defectData.image.exif_data.technical_info.iso}</td>
                                    </tr>
                                  )}
                                  {defectData.image.exif_data.technical_info.exposure_time && (
                                    <tr>
                                      <th>Exposure:</th>
                                      <td>{defectData.image.exif_data.technical_info.exposure_time}</td>
                                    </tr>
                                  )}
                                  {defectData.image.exif_data.technical_info.focal_length && (
                                    <tr>
                                      <th>Focal Length:</th>
                                      <td>{defectData.image.exif_data.technical_info.focal_length}</td>
                                    </tr>
                                  )}
                                </tbody>
                              </table>
                            </Col>
                          )}

                          {/* Basic Media Info */}
                          {defectData.image.exif_data?.basic_info && (
                            <Col md={6} className="mb-3">
                              <h6 className="text-info">📐 Media Properties</h6>
                              <table className="table table-sm">
                                <tbody>
                                  <tr>
                                    <th width="40%">Dimensions:</th>
                                    <td>{defectData.image.exif_data.basic_info.width} × {defectData.image.exif_data.basic_info.height}</td>
                                  </tr>
                                  {defectData.image.exif_data.basic_info.format && (
                                    <tr>
                                      <th>Format:</th>
                                      <td>{defectData.image.exif_data.basic_info.format}</td>
                                    </tr>
                                  )}
                                </tbody>
                              </table>
                            </Col>
                          )}

                          {/* GPS Information */}
                          {defectData.image.exif_data?.gps_coordinates && (
                            <Col md={6} className="mb-3">
                              <h6 className="text-warning">🌍 GPS Information</h6>
                              <table className="table table-sm">
                                <tbody>
                                  <tr>
                                    <th width="40%">Latitude:</th>
                                    <td>{defectData.image.exif_data.gps_coordinates.latitude?.toFixed(6)}</td>
                                  </tr>
                                  <tr>
                                    <th>Longitude:</th>
                                    <td>{defectData.image.exif_data.gps_coordinates.longitude?.toFixed(6)}</td>
                                  </tr>
                                </tbody>
                              </table>
                            </Col>
                          )}

                          {/* Video-specific Information */}
                          {defectData.image.media_type === 'video' && (
                            <Col md={12} className="mb-3">
                              <h6 className="text-danger">🎬 Video Information</h6>
                              <table className="table table-sm">
                                <tbody>
                                  {defectData.image.video_id && (
                                    <tr>
                                      <th width="20%">Video ID:</th>
                                      <td>{defectData.image.video_id}</td>
                                    </tr>
                                  )}
                                  {defectData.image.metadata?.format_info?.duration && (
                                    <tr>
                                      <th>Duration:</th>
                                      <td>{Math.round(defectData.image.metadata.format_info.duration)}s</td>
                                    </tr>
                                  )}
                                  {defectData.image.metadata?.basic_info?.width && defectData.image.metadata?.basic_info?.height && (
                                    <tr>
                                      <th>Resolution:</th>
                                      <td>{defectData.image.metadata.basic_info.width}x{defectData.image.metadata.basic_info.height}</td>
                                    </tr>
                                  )}
                                  {defectData.image.metadata?.format_info?.format_name && (
                                    <tr>
                                      <th>Format:</th>
                                      <td>{defectData.image.metadata.format_info.format_name.toUpperCase()}</td>
                                    </tr>
                                  )}
                                  {/* Show detection counts for videos */}
                                  {defectData.image.model_outputs && (
                                    <>
                                      {defectData.image.model_outputs.potholes && defectData.image.model_outputs.potholes.length > 0 && (
                                        <tr>
                                          <th>Potholes Detected:</th>
                                          <td>{defectData.image.model_outputs.potholes.length}</td>
                                        </tr>
                                      )}
                                      {defectData.image.model_outputs.cracks && defectData.image.model_outputs.cracks.length > 0 && (
                                        <tr>
                                          <th>Cracks Detected:</th>
                                          <td>{defectData.image.model_outputs.cracks.length}</td>
                                        </tr>
                                      )}
                                      {defectData.image.model_outputs.kerbs && defectData.image.model_outputs.kerbs.length > 0 && (
                                        <tr>
                                          <th>Kerbs Detected:</th>
                                          <td>{defectData.image.model_outputs.kerbs.length}</td>
                                        </tr>
                                      )}
                                    </>
                                  )}
                                </tbody>
                              </table>
                            </Col>
                          )}
                        </Row>
                      </Card.Body>
                    </Card>
                  </Col>
                </Row>
              )}

              {defectData.type === 'pothole' && defectData.image.potholes && (
                <div className="mt-4">
                  <h5>Pothole Details</h5>
                  <div className="table-responsive">
                    <table className="table table-striped table-bordered">
                      <thead className="table-primary">
                        <tr>
                          <th>ID</th>
                          <th>Area (cm²)</th>
                          <th>Depth (cm)</th>
                          <th>Volume (cm³)</th>
                          <th>Severity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {defectData.image.potholes.map((pothole, index) => (
                          <tr key={index}>
                            <td>{pothole.pothole_id}</td>
                            <td>{pothole.area_cm2?.toFixed(2) || 'N/A'}</td>
                            <td>{pothole.depth_cm?.toFixed(2) || 'N/A'}</td>
                            <td>{pothole.volume?.toFixed(2) || 'N/A'}</td>
                            <td>
                              {pothole.area_cm2 > 1000 ? (
                                <Badge bg="danger">High</Badge>
                              ) : pothole.area_cm2 > 500 ? (
                                <Badge bg="warning" text="dark">Medium</Badge>
                              ) : (
                                <Badge bg="success">Low</Badge>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {defectData.type === 'crack' && defectData.image.cracks && (
                <div className="mt-4">
                  <h5>Crack Details</h5>
                  <div className="table-responsive">
                    <table className="table table-striped table-bordered">
                      <thead className="table-primary">
                        <tr>
                          <th>ID</th>
                          <th>Type</th>
                          <th>Area (cm²)</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {defectData.image.cracks.map((crack, index) => (
                          <tr key={index}>
                            <td>{crack.crack_id}</td>
                            <td>{crack.crack_type}</td>
                            <td>{crack.area_cm2?.toFixed(2) || 'N/A'}</td>
                            <td>{(crack.confidence * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  {defectData.image.type_counts && (
                    <div className="mt-3">
                      <h6>Crack Type Distribution</h6>
                      <div className="d-flex flex-wrap">
                        {Object.entries(defectData.image.type_counts).map(([type, count]) => (
                          <div key={type} className="me-3 mb-2">
                            <Badge bg="info" className="me-1">{count}</Badge> {type}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {defectData.type === 'kerb' && defectData.image.kerbs && (
                <div className="mt-4">
                  <h5>Kerb Details</h5>
                  <div className="table-responsive">
                    <table className="table table-striped table-bordered">
                      <thead className="table-primary">
                        <tr>
                          <th>ID</th>
                          <th>Type</th>
                          <th>Condition</th>
                          <th>Length (m)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {defectData.image.kerbs.map((kerb, index) => (
                          <tr key={index}>
                            <td>{kerb.kerb_id}</td>
                            <td>{kerb.kerb_type}</td>
                            <td>
                              <Badge 
                                bg={
                                  kerb.condition === 'Good' ? 'success' :
                                  kerb.condition === 'Fair' ? 'warning' : 'danger'
                                }
                                text={kerb.condition === 'Fair' ? 'dark' : undefined}
                              >
                                {kerb.condition}
                              </Badge>
                            </td>
                            <td>{kerb.length_m?.toFixed(2) || 'N/A'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  {defectData.image.condition_counts && (
                    <div className="mt-3">
                      <h6>Condition Distribution</h6>
                      <div className="d-flex flex-wrap">
                        {Object.entries(defectData.image.condition_counts).map(([condition, count]) => (
                          <div key={condition} className="me-3 mb-2">
                            <Badge 
                              bg={
                                condition === 'Good' ? 'success' :
                                condition === 'Fair' ? 'warning' : 'danger'
                              }
                              text={condition === 'Fair' ? 'dark' : undefined}
                              className="me-1"
                            >
                              {count}
                            </Badge> {condition}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Recommendation section - if available */}
              <div className="mt-4">
                <Card className="bg-light">
                  <Card.Header>
                    <h5 className="mb-0">Recommended Action</h5>
                  </Card.Header>
                  <Card.Body>
                    <p>Based on the defect analysis, the following action is recommended:</p>
                    {defectData.type === 'pothole' && (
                      <div>
                        <ul>
                          <li>Clean out loose material</li>
                          <li>Apply tack coat</li>
                          <li>Fill with hot mix asphalt</li>
                          <li>Compact thoroughly</li>
                        </ul>
                        <p><strong>Priority:</strong> {
                          defectData.image.potholes && defectData.image.potholes.length > 0 && 
                          defectData.image.potholes.some(p => p.area_cm2 > 1000) ? 
                          'High' : defectData.image.potholes && defectData.image.potholes.length > 0 && 
                          defectData.image.potholes.some(p => p.area_cm2 > 500) ? 
                          'Medium' : 'Low'
                        }</p>
                      </div>
                    )}
                    
                    {defectData.type === 'crack' && (
                      <div>
                        <ul>
                          <li>Clean cracks with compressed air</li>
                          <li>Apply appropriate crack sealant</li>
                          {defectData.image.type_counts && 
                           defectData.image.type_counts['Alligator Crack'] > 0 && (
                            <li>Consider section replacement for alligator crack areas</li>
                          )}
                        </ul>
                        <p><strong>Priority:</strong> {
                          defectData.image.type_counts && 
                          defectData.image.type_counts['Alligator Crack'] > 0 ? 
                          'High' : 'Medium'
                        }</p>
                      </div>
                    )}
                    
                    {defectData.type === 'kerb' && (
                      <div>
                        <ul>
                          <li>Repair damaged sections</li>
                          <li>Realign displaced kerbs</li>
                          <li>Replace severely damaged kerbs</li>
                        </ul>
                        <p><strong>Priority:</strong> {
                          defectData.image.condition_counts && 
                          defectData.image.condition_counts['Poor'] > 0 ? 
                          'High' : defectData.image.condition_counts['Fair'] > 0 ? 
                          'Medium' : 'Low'
                        }</p>
                      </div>
                    )}
                  </Card.Body>
                </Card>
              </div>
            </Card.Body>
          </Card>

          <div className="d-flex justify-content-end mt-4">
            <Button variant="primary">
              Generate Report
            </Button>
          </div>
        </>
      ) : (
        <Alert variant="warning">No defect data found for ID: {imageId}</Alert>
      )}
    </Container>
  );
}

export default DefectDetail; 