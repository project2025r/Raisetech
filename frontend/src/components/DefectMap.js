import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { Card, Row, Col, Form, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';

// Fix for the Leaflet default icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom marker icons for different defect types using location icons
const createCustomIcon = (color, isVideo = false) => {
  const iconSize = isVideo ? [36, 36] : [32, 32];

  // Create different SVG content for video vs image markers
  const svgContent = isVideo ?
    // Video marker with camera icon made from SVG shapes (no Unicode)
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}" width="${iconSize[0]}px" height="${iconSize[1]}px">
      <path d="M0 0h24v24H0z" fill="none"/>
      <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
      <rect x="8" y="6" width="8" height="6" rx="1" fill="white"/>
      <circle cx="9.5" cy="7.5" r="1" fill="${color}"/>
      <rect x="11" y="7" width="4" height="2" fill="${color}"/>
    </svg>` :
    // Image marker with standard location pin (no Unicode)
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}" width="${iconSize[0]}px" height="${iconSize[1]}px">
      <path d="M0 0h24v24H0z" fill="none"/>
      <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
    </svg>`;

  return L.icon({
    iconUrl: `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svgContent)}`,
    iconSize: iconSize,
    iconAnchor: [iconSize[0]/2, iconSize[1]],
    popupAnchor: [0, -iconSize[1]],
  });
};

const icons = {
  pothole: createCustomIcon('#FF0000'), // Red for potholes
  crack: createCustomIcon('#FFCC00'),   // Yellow for cracks (alligator cracks)
  kerb: createCustomIcon('#0066FF'),    // Blue for kerb defects
  // Video variants
  'pothole-video': createCustomIcon('#FF0000', true), // Red for pothole videos
  'crack-video': createCustomIcon('#FFCC00', true),   // Yellow for crack videos
  'kerb-video': createCustomIcon('#0066FF', true),    // Blue for kerb videos
};

function DefectMap({ user }) {
  const [defects, setDefects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [center] = useState([20.5937, 78.9629]); // India center
  const [zoom] = useState(6); // Country-wide zoom for India
  
  // Filters for the map
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [selectedUser, setSelectedUser] = useState('');
  const [usersList, setUsersList] = useState([]);
  const [defectTypeFilters, setDefectTypeFilters] = useState({
    pothole: true,
    crack: true,
    kerb: true,
  });

  // Fetch all images with defects and coordinates
  const fetchDefectData = async (forceRefresh = false) => {
    try {
      setLoading(true);
      setError(null); // Clear any previous errors

      // Prepare query parameters
      let params = {};
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;
      if (selectedUser) params.username = selectedUser;
      if (user?.role) params.user_role = user.role;

      // Add cache-busting parameter for force refresh
      if (forceRefresh) {
        params._t = Date.now();
      }

      console.log('Fetching defect data with params:', params);
      const response = await axios.get('/api/dashboard/image-stats', { params });
      console.log('API response:', response.data);

      if (response.data.success) {
        // Process the data to extract defect locations with type information
        const processedDefects = [];
        
        response.data.images.forEach(image => {
          try {
            // Only process images with valid coordinates
            if (image.coordinates && image.coordinates !== 'Not Available') {
              let lat, lng;

              // First, try to use EXIF GPS coordinates if available (most accurate)
              if (image.exif_data?.gps_coordinates) {
                lat = image.exif_data.gps_coordinates.latitude;
                lng = image.exif_data.gps_coordinates.longitude;
                console.log(`🎯 Using EXIF GPS coordinates for ${image.image_id}: [${lat}, ${lng}]`);
              } else {
                // Fallback to stored coordinates
                // Handle different coordinate formats
                if (typeof image.coordinates === 'string') {
                  // Parse coordinates (expected format: "lat,lng")
                  const coords = image.coordinates.split(',');
                  if (coords.length === 2) {
                    lat = parseFloat(coords[0].trim());
                    lng = parseFloat(coords[1].trim());
                  }
                } else if (Array.isArray(image.coordinates) && image.coordinates.length === 2) {
                  // Handle array format [lat, lng]
                  lat = parseFloat(image.coordinates[0]);
                  lng = parseFloat(image.coordinates[1]);
                } else if (typeof image.coordinates === 'object' && image.coordinates.lat && image.coordinates.lng) {
                  // Handle object format {lat: x, lng: y}
                  lat = parseFloat(image.coordinates.lat);
                  lng = parseFloat(image.coordinates.lng);
                }
                console.log(`📍 Using stored coordinates for ${image.image_id}: [${lat}, ${lng}]`);
              }

            // Validate coordinates are within reasonable bounds
            if (!isNaN(lat) && !isNaN(lng) &&
                lat >= -90 && lat <= 90 &&
                lng >= -180 && lng <= 180) {
              console.log(`✅ Valid coordinates for image ${image.id}: [${lat}, ${lng}]`);
              processedDefects.push({
                id: image.id,
                image_id: image.image_id,
                type: image.type,
                position: [lat, lng],
                defect_count: image.defect_count,
                timestamp: new Date(image.timestamp).toLocaleString(),
                username: image.username,
                original_image_id: image.original_image_id,
                // For cracks, include type information if available
                type_counts: image.type_counts,
                // For kerbs, include condition information if available
                condition_counts: image.condition_counts,
                // EXIF and metadata information
                exif_data: image.exif_data || {},
                metadata: image.metadata || {},
                media_type: image.media_type || 'image',
                original_image_full_url: image.original_image_full_url
              });
            } else {
              console.warn(`❌ Invalid coordinates for image ${image.id}:`, image.coordinates, `parsed: lat=${lat}, lng=${lng}`);
            }
          } else {
            console.log(`⚠️ Skipping image ${image.id}: coordinates=${image.coordinates}`);
          }
          } catch (coordError) {
            console.error(`❌ Error processing coordinates for image ${image.id}:`, coordError, image.coordinates);
          }
        });

        console.log('Processed defects:', processedDefects.length);
        setDefects(processedDefects);

        // If no defects were processed, show a helpful message
        if (processedDefects.length === 0) {
          setError('No defects found with valid coordinates for the selected date range');
        }
      } else {
        console.error('API returned success: false', response.data);
        setError('Error fetching defect data: ' + (response.data.message || 'Unknown error'));
      }

      setLoading(false);
    } catch (err) {
      console.error('Error fetching defect data:', err);
      setError('Failed to load defect data: ' + (err.response?.data?.message || err.message));
      setLoading(false);
    }
  };
  
  // Fetch the list of users for the filter dropdown
  const fetchUsers = async () => {
    try {
      const params = {};
      if (user?.role) params.user_role = user.role;
      
      const response = await axios.get('/api/users/all', { params });
      if (response.data.success) {
        setUsersList(response.data.users);
      }
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  // Initialize default dates for the last 30 days
  useEffect(() => {
    const currentDate = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    setEndDate(currentDate.toISOString().split('T')[0]);
    setStartDate(thirtyDaysAgo.toISOString().split('T')[0]);

    console.log('DefectMap component initialized');
  }, []);

  // Fetch data when component mounts and when filters change
  useEffect(() => {
    fetchDefectData();
    fetchUsers();
  }, [user]);

  // Auto-refresh every 30 seconds to catch new uploads
  useEffect(() => {
    const interval = setInterval(() => {
      fetchDefectData(true); // Force refresh to get latest data
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [startDate, endDate, selectedUser, user?.role]);

  // Manual refresh function
  const handleRefresh = () => {
    fetchDefectData(true);
  };

  // Handle filter application
  const handleApplyFilters = () => {
    fetchDefectData();
  };

  // Handle resetting filters
  const handleResetFilters = () => {
    // Reset date filters to last 30 days
    const currentDate = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    
    setEndDate(currentDate.toISOString().split('T')[0]);
    setStartDate(thirtyDaysAgo.toISOString().split('T')[0]);
    setSelectedUser('');
    setDefectTypeFilters({
      pothole: true,
      crack: true,
      kerb: true,
    });
    
    // Refetch data with reset filters
    fetchDefectData();
  };

  // Handle defect type filter changes
  const handleDefectTypeFilterChange = (type) => {
    setDefectTypeFilters(prevFilters => ({
      ...prevFilters,
      [type]: !prevFilters[type]
    }));
  };

  // Filter defects based on applied filters
  const filteredDefects = defects.filter(defect => 
    defectTypeFilters[defect.type]
  );

  return (
    <Card className="shadow-sm dashboard-card mb-4">
      <Card.Header className="bg-primary text-white">
        <h5 className="mb-0">Defect Map View</h5>
      </Card.Header>
      <Card.Body>
        <Row className="mb-3">
          <Col lg={4} className="filter-section">
            <h6>Date Filter</h6>
            <div className="filter-controls">
              <div className="filter-field-container">
                <div className="filter-field">
                  <Form.Group>
                    <Form.Label className="small">Start Date</Form.Label>
                    <Form.Control
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      size="sm"
                    />
                  </Form.Group>
                </div>
                <div className="filter-field">
                  <Form.Group>
                    <Form.Label className="small">End Date</Form.Label>
                    <Form.Control
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      size="sm"
                    />
                  </Form.Group>
                </div>
              </div>
            </div>
          </Col>
          <Col lg={4} className="filter-section">
            <h6>User Filter</h6>
            <div className="filter-controls">
              <div className="filter-field-container">
                <div className="filter-field">
                  <Form.Group>
                    <Form.Label className="small">Select User</Form.Label>
                    <Form.Select
                      value={selectedUser}
                      onChange={(e) => setSelectedUser(e.target.value)}
                      size="sm"
                    >
                      <option value="">All Users</option>
                      {usersList.map((user, index) => (
                        <option key={index} value={user.username}>
                          {user.username} ({user.role})
                        </option>
                      ))}
                    </Form.Select>
                  </Form.Group>
                </div>
              </div>
            </div>
          </Col>
          <Col lg={4} className="filter-section">
            <h6>Defect Type Filter</h6>
            <div className="filter-controls defect-type-filters">
              <div className="defect-type-filter-options" style={{ marginTop: "0", paddingLeft: "0", width: "100%", minWidth: "150px" }}>
                <Form.Check
                  type="checkbox"
                  id="pothole-filter"
                  label="Potholes"
                  checked={defectTypeFilters.pothole}
                  onChange={() => handleDefectTypeFilterChange('pothole')}
                  className="mb-2 filter-checkbox"
                />
                <Form.Check
                  type="checkbox"
                  id="crack-filter"
                  label="Cracks"
                  checked={defectTypeFilters.crack}
                  onChange={() => handleDefectTypeFilterChange('crack')}
                  className="mb-2 filter-checkbox"
                />
                <Form.Check
                  type="checkbox"
                  id="kerb-filter"
                  label="Kerbs"
                  checked={defectTypeFilters.kerb}
                  onChange={() => handleDefectTypeFilterChange('kerb')}
                  className="mb-2 filter-checkbox"
                />
              </div>
            </div>
          </Col>
        </Row>
        
        <Row className="mb-3">
          <Col className="text-center">
            <div className="filter-actions-container">
              <Button 
                variant="primary" 
                size="sm" 
                onClick={handleApplyFilters}
                className="me-3 filter-btn"
              >
                Apply Filters
              </Button>
              <Button
                variant="outline-secondary"
                size="sm"
                onClick={handleResetFilters}
                className="me-3 filter-btn"
              >
                Reset Filters
              </Button>
              <Button
                variant="success"
                size="sm"
                onClick={handleRefresh}
                disabled={loading}
                className="filter-btn"
              >
                {loading ? '🔄 Refreshing...' : '🔄 Refresh Map'}
              </Button>
            </div>
          </Col>
        </Row>

        <Row>
          <Col>
            <div className="map-legend mb-3">
              <div className="legend-section">
                <h6 className="legend-title">📷 Images</h6>
                <div className="legend-item">
                  <div className="legend-marker" style={{ backgroundColor: '#FF0000' }}></div>
                  <span>Potholes</span>
                </div>
                <div className="legend-item">
                  <div className="legend-marker" style={{ backgroundColor: '#FFCC00' }}></div>
                  <span>Cracks</span>
                </div>
                <div className="legend-item">
                  <div className="legend-marker" style={{ backgroundColor: '#0066FF' }}></div>
                  <span>Kerb Defects</span>
                </div>
              </div>
              <div className="legend-section">
                <h6 className="legend-title">📹 Videos</h6>
                <div className="legend-item">
                  <div className="legend-marker video-marker" style={{ backgroundColor: '#FF0000' }}>📹</div>
                  <span>Pothole Videos</span>
                </div>
                <div className="legend-item">
                  <div className="legend-marker video-marker" style={{ backgroundColor: '#FFCC00' }}>📹</div>
                  <span>Crack Videos</span>
                </div>
                <div className="legend-item">
                  <div className="legend-marker video-marker" style={{ backgroundColor: '#0066FF' }}>📹</div>
                  <span>Kerb Videos</span>
                </div>
              </div>
            </div>
          </Col>
        </Row>

        {loading ? (
          <div className="text-center p-5">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : error ? (
          <div className="alert alert-danger">{error}</div>
        ) : (
          <div className="map-container" style={{ height: '500px', width: '100%' }}>
            <MapContainer 
              center={center} 
              zoom={zoom} 
              style={{ height: '100%', width: '100%' }}
              scrollWheelZoom={true}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              
              {filteredDefects.map((defect) => {
                // Determine icon based on media type and defect type
                const iconKey = defect.media_type === 'video' ? `${defect.type}-video` : defect.type;
                const selectedIcon = icons[iconKey] || icons[defect.type];

                return (
                <Marker
                  key={defect.id}
                  position={defect.position}
                  icon={selectedIcon}
                >
                  <Popup maxWidth={400}>
                    <div className="defect-popup">
                      <h6>{defect.type.charAt(0).toUpperCase() + defect.type.slice(1)} Defect</h6>

                      {/* Video Thumbnail */}
                      {defect.media_type === 'video' && defect.representative_frame && (
                        <div className="mb-3 text-center">
                          <img
                            src={`data:image/jpeg;base64,${defect.representative_frame}`}
                            alt="Video thumbnail"
                            className="img-fluid border rounded"
                            style={{ maxHeight: '150px', maxWidth: '100%' }}
                            onError={(e) => {
                              console.warn(`Failed to load representative frame for video ${defect.image_id}`);
                              e.target.style.display = 'none';
                            }}
                          />
                          <div className="mt-1">
                            <small className="text-info fw-bold">📹 Video Representative Frame</small>
                            {defect.video_id && (
                              <div>
                                <small className="text-muted d-block">Video ID: {defect.video_id}</small>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Regular Image Display */}
                      {defect.media_type !== 'video' && defect.original_image_full_url && (
                        <div className="mb-3 text-center">
                          <img
                            src={defect.original_image_full_url}
                            alt="Defect image"
                            className="img-fluid border rounded"
                            style={{ maxHeight: '150px', maxWidth: '100%' }}
                            onError={(e) => {
                              console.warn(`Failed to load image for defect ${defect.image_id}`);
                              e.target.style.display = 'none';
                            }}
                          />
                          <div className="mt-1">
                            <small className="text-primary fw-bold">📷 Original Image</small>
                          </div>
                        </div>
                      )}

                      {/* Basic Information */}
                      <div className="mb-3">
                        <h6 className="text-primary">Basic Information</h6>
                        <ul className="list-unstyled small">
                          <li><strong>Count:</strong> {defect.defect_count}</li>
                          <li><strong>Date:</strong> {defect.timestamp}</li>
                          <li><strong>Reported by:</strong> {defect.username}</li>
                          <li><strong>Media Type:</strong> {defect.media_type}</li>
                          <li><strong>GPS:</strong> {defect.position[0].toFixed(6)}, {defect.position[1].toFixed(6)}</li>
                        </ul>
                      </div>

                      {/* Defect-specific Information */}
                      {defect.type === 'crack' && defect.type_counts && (
                        <div className="mb-3">
                          <h6 className="text-primary">Crack Types</h6>
                          <ul className="list-unstyled small">
                            {Object.entries(defect.type_counts).map(([type, count]) => (
                              <li key={type}><strong>{type}:</strong> {count}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {defect.type === 'kerb' && defect.condition_counts && (
                        <div className="mb-3">
                          <h6 className="text-primary">Kerb Conditions</h6>
                          <ul className="list-unstyled small">
                            {Object.entries(defect.condition_counts).map(([condition, count]) => (
                              <li key={condition}><strong>{condition}:</strong> {count}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* EXIF/Metadata Information */}
                      {(defect.exif_data || defect.metadata) && (
                        <div className="mb-3">
                          <h6 className="text-primary">📊 Media Information</h6>
                          <div className="small">
                            {/* GPS Information - Show first as it's most important for mapping */}
                            {defect.exif_data?.gps_coordinates && (
                              <div className="mb-2 p-2 bg-light rounded">
                                <strong className="text-success">🌍 GPS (EXIF):</strong>
                                <div>Lat: {defect.exif_data.gps_coordinates.latitude?.toFixed(6)}</div>
                                <div>Lng: {defect.exif_data.gps_coordinates.longitude?.toFixed(6)}</div>
                              </div>
                            )}

                            {/* Camera Information */}
                            {defect.exif_data?.camera_info && Object.keys(defect.exif_data.camera_info).length > 0 && (
                              <div className="mb-2">
                                <strong>📷 Camera:</strong>
                                <ul className="list-unstyled ms-2">
                                  {defect.exif_data.camera_info.camera_make && (
                                    <li>Make: {defect.exif_data.camera_info.camera_make}</li>
                                  )}
                                  {defect.exif_data.camera_info.camera_model && (
                                    <li>Model: {defect.exif_data.camera_info.camera_model}</li>
                                  )}
                                </ul>
                              </div>
                            )}

                            {/* Technical Information */}
                            {defect.exif_data?.technical_info && Object.keys(defect.exif_data.technical_info).length > 0 && (
                              <div className="mb-2">
                                <strong>⚙️ Technical:</strong>
                                <ul className="list-unstyled ms-2">
                                  {defect.exif_data.technical_info.iso && (
                                    <li>ISO: {defect.exif_data.technical_info.iso}</li>
                                  )}
                                  {defect.exif_data.technical_info.exposure_time && (
                                    <li>Exposure: {defect.exif_data.technical_info.exposure_time}</li>
                                  )}
                                </ul>
                              </div>
                            )}

                            {/* Basic Media Info */}
                            {defect.exif_data?.basic_info && (
                              <div className="mb-2">
                                <strong>📐 Dimensions:</strong> {defect.exif_data.basic_info.width} × {defect.exif_data.basic_info.height}
                                {defect.exif_data.basic_info.format && (
                                  <span> ({defect.exif_data.basic_info.format})</span>
                                )}
                              </div>
                            )}

                            {/* Video-specific metadata */}
                            {defect.media_type === 'video' && (
                              <div className="mb-3">
                                <h6 className="text-info">📹 Video Information</h6>
                                <ul className="list-unstyled small">
                                  {defect.metadata?.format_info?.duration && (
                                    <li><strong>Duration:</strong> {Math.round(defect.metadata.format_info.duration)}s</li>
                                  )}
                                  {defect.metadata?.basic_info?.width && defect.metadata?.basic_info?.height && (
                                    <li><strong>Resolution:</strong> {defect.metadata.basic_info.width}x{defect.metadata.basic_info.height}</li>
                                  )}
                                  {defect.metadata?.format_info?.format_name && (
                                    <li><strong>Format:</strong> {defect.metadata.format_info.format_name.toUpperCase()}</li>
                                  )}
                                  {defect.video_id && (
                                    <li><strong>Video ID:</strong> {defect.video_id}</li>
                                  )}
                                  {defect.original_video_url && (
                                    <li><strong>Original Video:</strong> Available</li>
                                  )}
                                  {defect.processed_video_url && (
                                    <li><strong>Processed Video:</strong> Available</li>
                                  )}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Action Buttons */}
                      <div className="d-flex gap-2">
                        <Link
                          to={`/view/${defect.image_id}`}
                          className="btn btn-sm btn-primary"
                          onClick={(e) => e.stopPropagation()}
                        >
                          View Details
                        </Link>
                        {defect.original_image_full_url && (
                          <a
                            href={defect.original_image_full_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn btn-sm btn-outline-secondary"
                            onClick={(e) => e.stopPropagation()}
                          >
                            View Original
                          </a>
                        )}
                      </div>
                    </div>
                  </Popup>
                </Marker>
                );
              })}
            </MapContainer>
          </div>
        )}
      </Card.Body>
    </Card>
  );
}

export default DefectMap; 