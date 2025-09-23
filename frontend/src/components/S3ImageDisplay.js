import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Tabs, Tab, Button, Alert, Spinner } from 'react-bootstrap';

/**
 * S3 Image Display Component
 * Demonstrates how to retrieve and display images from S3 using URLs stored in MongoDB
 * Similar to the "All uploaded images" section of the dashboard
 */

/**
 * Enhanced Image URL Resolution Logic
 * Handles multiple URL types with comprehensive fallback system
 */
const getImageUrlForDisplay = (imageData, imageType = 'original') => {
  console.log('🔍 Resolving image URL:', { imageData, imageType });

  if (!imageData) {
    console.log('❌ No imageData provided');
    return null;
  }

  // Handle video representative frames
  if (imageData.media_type === 'video' && imageData.representative_frame) {
    console.log('📹 Using video representative frame');
    return `data:image/jpeg;base64,${imageData.representative_frame}`;
  }

  // Priority 1: Pre-signed URL (most secure and reliable)
  const presignedUrlField = `${imageType}_image_presigned_url`;
  if (imageData[presignedUrlField]) {
    console.log('🔗 Using pre-signed URL:', presignedUrlField);
    return imageData[presignedUrlField];
  }

  // Priority 2: S3 Proxy endpoint (recommended for private buckets)
  const s3KeyField = `${imageType}_image_s3_url`;
  if (imageData[s3KeyField]) {
    const proxyUrl = `/api/pavement/get-s3-image/${encodeURIComponent(imageData[s3KeyField])}`;
    console.log('🔄 Using S3 proxy URL:', proxyUrl);
    return proxyUrl;
  }

  // Priority 3: Direct S3 URL (if bucket is public)
  const fullUrlField = `${imageType}_image_full_url`;
  if (imageData[fullUrlField]) {
    console.log('🌐 Using direct S3 URL:', fullUrlField);
    return imageData[fullUrlField];
  }

  // Priority 4: GridFS endpoint (legacy fallback)
  const gridfsIdField = `${imageType}_image_id`;
  if (imageData[gridfsIdField]) {
    const gridfsUrl = `/api/pavement/get-image/${imageData[gridfsIdField]}`;
    console.log('📁 Using GridFS URL:', gridfsUrl);
    return gridfsUrl;
  }

  // Priority 5: Use display_url if available
  if (imageData.display_url) {
    console.log('🎯 Using display URL:', imageData.display_url);
    return imageData.display_url;
  }

  console.log('❌ No valid image URL found');
  return null;
};

/**
 * Enhanced Image Component with comprehensive error handling and fallback system
 */
const EnhancedS3ImageDisplay = ({ imageData, imageType = 'original', alt, className, style, onError }) => {
  const [currentImageUrl, setCurrentImageUrl] = useState(null);
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [fallbackAttempts, setFallbackAttempts] = useState(0);

  useEffect(() => {
    // Reset state when imageData changes
    setHasError(false);
    setIsLoading(true);
    setFallbackAttempts(0);

    // Get initial image URL
    const imageUrl = getImageUrlForDisplay(imageData, imageType);

    console.log('🖼️ Enhanced S3 Image Display Debug:', {
      imageType,
      imageId: imageData?.image_id,
      generatedUrl: imageUrl,
      s3Key: imageData?.[`${imageType}_image_s3_url`],
      presignedUrl: imageData?.[`${imageType}_image_presigned_url`],
      fullUrl: imageData?.[`${imageType}_image_full_url`],
      gridfsId: imageData?.[`${imageType}_image_id`]
    });

    setCurrentImageUrl(imageUrl);
  }, [imageData, imageType]);

  const handleImageError = () => {
    console.error('❌ Image loading failed:', currentImageUrl);
    setIsLoading(false);

    // Try fallback URLs
    if (fallbackAttempts === 0) {
      // First fallback: try alternative image type
      const alternativeType = imageType === 'original' ? 'processed' : 'original';
      const fallbackUrl = getImageUrlForDisplay(imageData, alternativeType);
      
      if (fallbackUrl && fallbackUrl !== currentImageUrl) {
        console.log('🔄 Trying alternative image type:', alternativeType);
        setCurrentImageUrl(fallbackUrl);
        setFallbackAttempts(1);
        return;
      }
    }

    if (fallbackAttempts === 1) {
      // Second fallback: try direct S3 URL
      const directUrl = imageData?.[`${imageType}_image_full_url`];
      if (directUrl && directUrl !== currentImageUrl) {
        console.log('🔄 Trying direct S3 URL:', directUrl);
        setCurrentImageUrl(directUrl);
        setFallbackAttempts(2);
        return;
      }
    }

    // All fallbacks failed
    console.error('❌ All image loading attempts failed');
    setHasError(true);
    if (onError) onError();
  };

  const handleImageLoad = () => {
    console.log('✅ Image loaded successfully:', currentImageUrl);
    setIsLoading(false);
    setHasError(false);
  };

  if (hasError || !currentImageUrl) {
    return (
      <div className={`text-muted d-flex align-items-center justify-content-center ${className}`} style={style}>
        <div className="text-center">
          <i className="fas fa-image-slash fa-2x mb-2"></i>
          <div>Image not available</div>
          {fallbackAttempts > 0 && (
            <small className="text-warning">
              ({fallbackAttempts} fallback attempts made)
            </small>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="position-relative">
      {isLoading && (
        <div className="position-absolute top-50 start-50 translate-middle">
          <Spinner animation="border" size="sm" />
        </div>
      )}
      <img
        src={currentImageUrl}
        alt={alt}
        className={className}
        style={style}
        onError={handleImageError}
        onLoad={handleImageLoad}
        loading="lazy"
      />
      {fallbackAttempts > 0 && (
        <div className="position-absolute bottom-0 end-0 bg-warning text-dark px-1 rounded-start" style={{ fontSize: '0.7rem' }}>
          Fallback {fallbackAttempts}
        </div>
      )}
    </div>
  );
};

/**
 * Image Card Component for individual image display
 */
const S3ImageCard = ({ imageData, defectType }) => {
  const [isOriginal, setIsOriginal] = useState(false);

  if (!imageData) {
    return null;
  }

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getDefectDisplayName = (type) => {
    const names = {
      'potholes': 'Pothole',
      'cracks': 'Crack',
      'kerbs': 'Kerb'
    };
    return names[type] || type;
  };

  return (
    <Col md={4} lg={3} className="mb-3">
      <Card className="h-100 shadow-sm">
        <Card.Header className="py-2">
          <div className="d-flex justify-content-between align-items-center">
            <small className="fw-bold text-primary">
              {getDefectDisplayName(defectType)} {imageData.image_id}
            </small>
            <div>
              <Button
                variant={isOriginal ? "outline-secondary" : "primary"}
                size="sm"
                onClick={() => setIsOriginal(false)}
                className="me-1"
              >
                Processed
              </Button>
              <Button
                variant={isOriginal ? "primary" : "outline-secondary"}
                size="sm"
                onClick={() => setIsOriginal(true)}
              >
                Original
              </Button>
            </div>
          </div>
        </Card.Header>
        <Card.Body className="p-2">
          <div className="text-center mb-2">
            <EnhancedS3ImageDisplay
              imageData={imageData}
              imageType={isOriginal ? 'original' : 'processed'}
              alt={`${getDefectDisplayName(defectType)} ${imageData.image_id}`}
              className="img-fluid border rounded"
              style={{ maxHeight: "200px", width: "100%", objectFit: "cover" }}
              onError={() => {
                console.warn(`Failed to load ${isOriginal ? 'original' : 'processed'} image for ${defectType} ${imageData.image_id}`);
              }}
            />
          </div>
          
          <div className="small">
            <div className="mb-1">
              <strong>User:</strong> {imageData.username || 'N/A'}
            </div>
            <div className="mb-1">
              <strong>Role:</strong> {imageData.role || 'N/A'}
            </div>
            <div className="mb-1">
              <strong>Date:</strong> {formatDate(imageData.timestamp)}
            </div>
            {imageData.coordinates && (
              <div className="mb-1">
                <strong>Location:</strong> {imageData.coordinates}
              </div>
            )}
            
            {/* Debug Information */}
            <details className="mt-2">
              <summary className="text-muted" style={{ fontSize: '0.7rem', cursor: 'pointer' }}>
                Debug Info
              </summary>
              <div className="mt-1 p-1 bg-light rounded" style={{ fontSize: '0.6rem' }}>
                <div>S3 Key: {imageData[`${isOriginal ? 'original' : 'processed'}_image_s3_url`] || 'None'}</div>
                <div>Pre-signed: {imageData[`${isOriginal ? 'original' : 'processed'}_image_presigned_url`] ? 'Yes' : 'No'}</div>
                <div>Full URL: {imageData[`${isOriginal ? 'original' : 'processed'}_image_full_url`] ? 'Yes' : 'No'}</div>
                <div>GridFS ID: {imageData[`${isOriginal ? 'original' : 'processed'}_image_id`] || 'None'}</div>
                <div>Media Type: {imageData.media_type || 'image'}</div>
              </div>
            </details>
          </div>
        </Card.Body>
      </Card>
    </Col>
  );
};

/**
 * Main S3 Image Display Component
 * Demonstrates complete S3 image retrieval and display functionality
 */
const S3ImageDisplay = () => {
  const [imagesData, setImagesData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchImagesFromS3 = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('🔄 Fetching images from S3...');
      
      const response = await fetch('/api/images/retrieve-from-s3?types=pothole,crack,kerb&limit=20');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        console.log('✅ Images fetched successfully:', result.data);
        setImagesData(result.data);
      } else {
        throw new Error(result.message || 'Failed to fetch images');
      }
      
    } catch (err) {
      console.error('❌ Error fetching images:', err);
      setError(err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    fetchImagesFromS3();
  };

  useEffect(() => {
    fetchImagesFromS3();
  }, []);

  if (loading) {
    return (
      <div className="text-center py-5">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading images...</span>
        </Spinner>
        <div className="mt-2">Loading images from S3...</div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="danger">
        <Alert.Heading>Error Loading Images</Alert.Heading>
        <p>{error}</p>
        <Button variant="outline-danger" onClick={handleRefresh}>
          Try Again
        </Button>
      </Alert>
    );
  }

  if (!imagesData) {
    return (
      <Alert variant="info">
        <Alert.Heading>No Images Available</Alert.Heading>
        <p>No images found in the database.</p>
        <Button variant="outline-info" onClick={handleRefresh}>
          Refresh
        </Button>
      </Alert>
    );
  }

  return (
    <div className="container-fluid">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h4>S3 Image Retrieval & Display</h4>
        <Button 
          variant="outline-primary" 
          onClick={handleRefresh}
          disabled={refreshing}
        >
          {refreshing ? (
            <>
              <Spinner animation="border" size="sm" className="me-2" />
              Refreshing...
            </>
          ) : (
            <>
              <i className="fas fa-sync-alt me-2"></i>
              Refresh
            </>
          )}
        </Button>
      </div>

      <Card className="shadow-sm">
        <Card.Header className="bg-primary text-white">
          <h6 className="mb-0">All Images Retrieved from S3</h6>
        </Card.Header>
        <Card.Body>
          <Tabs defaultActiveKey="potholes" className="mb-3">
            <Tab 
              eventKey="potholes" 
              title={`Potholes (${imagesData.potholes?.count || 0})`}
            >
              <div style={{ maxHeight: '700px', overflowY: 'auto' }}>
                <Row>
                  {imagesData.potholes?.latest?.map((pothole, index) => (
                    <S3ImageCard
                      key={`pothole-${pothole.image_id || index}`}
                      imageData={pothole}
                      defectType="potholes"
                    />
                  ))}
                  {(!imagesData.potholes?.latest || imagesData.potholes.latest.length === 0) && (
                    <Col>
                      <Alert variant="info">No pothole images available</Alert>
                    </Col>
                  )}
                </Row>
              </div>
            </Tab>
            
            <Tab 
              eventKey="cracks" 
              title={`Cracks (${imagesData.cracks?.count || 0})`}
            >
              <div style={{ maxHeight: '700px', overflowY: 'auto' }}>
                <Row>
                  {imagesData.cracks?.latest?.map((crack, index) => (
                    <S3ImageCard
                      key={`crack-${crack.image_id || index}`}
                      imageData={crack}
                      defectType="cracks"
                    />
                  ))}
                  {(!imagesData.cracks?.latest || imagesData.cracks.latest.length === 0) && (
                    <Col>
                      <Alert variant="info">No crack images available</Alert>
                    </Col>
                  )}
                </Row>
              </div>
            </Tab>
            
            <Tab 
              eventKey="kerbs" 
              title={`Kerbs (${imagesData.kerbs?.count || 0})`}
            >
              <div style={{ maxHeight: '700px', overflowY: 'auto' }}>
                <Row>
                  {imagesData.kerbs?.latest?.map((kerb, index) => (
                    <S3ImageCard
                      key={`kerb-${kerb.image_id || index}`}
                      imageData={kerb}
                      defectType="kerbs"
                    />
                  ))}
                  {(!imagesData.kerbs?.latest || imagesData.kerbs.latest.length === 0) && (
                    <Col>
                      <Alert variant="info">No kerb images available</Alert>
                    </Col>
                  )}
                </Row>
              </div>
            </Tab>
          </Tabs>
        </Card.Body>
      </Card>
    </div>
  );
};

export default S3ImageDisplay;
