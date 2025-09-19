import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Container, Row, Col, Card, Button, Spinner, Alert, Badge } from 'react-bootstrap';
import axios from 'axios';
import './dashboard.css';

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
              </div>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={6} className="text-center mb-4">
                  {defectData.image && (() => {
                    // Check if this is video data with representative frame
                    if (defectData.image.media_type === 'video' && defectData.image.representative_frame) {
                      return (
                        <div className="defect-image-container">
                          <img
                            src={`data:image/jpeg;base64,${defectData.image.representative_frame}`}
                            alt={`${defectData.type} video thumbnail`}
                            className="img-fluid border rounded shadow-sm"
                            style={{ maxHeight: '400px' }}
                            onError={(e) => {
                              console.warn(`Failed to load representative frame for video ${defectData.image.image_id}`);
                              e.target.style.display = 'none';
                            }}
                          />
                          <div className="mt-2">
                            <small className="text-info fw-bold">
                              📹 Video Thumbnail
                            </small>
                          </div>
                        </div>
                      );
                    }

                    // Handle regular image data
                    // Check if S3 URLs are available (new format)
                    const s3Url = imageType === 'original'
                      ? (defectData.image.original_image_full_url || defectData.image.original_image_s3_url)
                      : (defectData.image.processed_image_full_url || defectData.image.processed_image_s3_url);

                    const gridfsId = imageType === 'original'
                      ? defectData.image.original_image_id
                      : defectData.image.processed_image_id;

                    // Use S3 URL if available, otherwise fall back to GridFS
                    const imageSrc = s3Url || (gridfsId ? `/api/pavement/get-image/${gridfsId}` : null);

                    return imageSrc ? (
                      <div className="defect-image-container">
                        <img
                          src={imageSrc}
                          alt={`${defectData.type} defect`}
                          className="img-fluid border rounded shadow-sm"
                          style={{ maxHeight: '400px' }}
                          onError={(e) => {
                            // If S3 image fails to load and we have GridFS ID, try GridFS as fallback
                            if (s3Url && gridfsId) {
                              e.target.src = `/api/pavement/get-image/${gridfsId}`;
                            }
                          }}
                        />
                      </div>
                    ) : (
                      <div className="text-muted">No image available</div>
                    );
                  })()}
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
                    </tbody>
                  </table>
                </Col>
              </Row>

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