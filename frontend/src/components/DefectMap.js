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
const createCustomIcon = (color) => {
  return L.icon({
    iconUrl: `data:image/svg+xml;base64,${btoa(`
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="${color}" width="32px" height="32px">
        <path d="M0 0h24v24H0z" fill="none"/>
        <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
      </svg>
    `)}`,
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32],
  });
};

const icons = {
  pothole: createCustomIcon('#FF0000'), // Red for potholes
  crack: createCustomIcon('#FFCC00'),   // Yellow for cracks (alligator cracks)
  kerb: createCustomIcon('#0066FF'),    // Blue for kerb defects
};

function DefectMap({ user }) {
  const [defects, setDefects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [center, setCenter] = useState([20.5937, 78.9629]); // Default center (India)
  const [zoom, setZoom] = useState(5);
  
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
  const fetchDefectData = async () => {
    try {
      setLoading(true);
      
      // Prepare query parameters
      let params = {};
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;
      if (selectedUser) params.username = selectedUser;
      if (user?.role) params.user_role = user.role;
      
      const response = await axios.get('/api/dashboard/image-stats', { params });
      
      if (response.data.success) {
        // Process the data to extract defect locations with type information
        const processedDefects = [];
        
        response.data.images.forEach(image => {
          // Only process images with valid coordinates
          if (image.coordinates && image.coordinates !== 'Not Available') {
            // Parse coordinates (expected format: "lat,lng")
            const [lat, lng] = image.coordinates.split(',').map(coord => parseFloat(coord.trim()));
            
            if (!isNaN(lat) && !isNaN(lng)) {
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
              });
            }
          }
        });
        
        setDefects(processedDefects);
      } else {
        setError('Error fetching defect data');
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching defect data:', err);
      setError('Failed to load defect data');
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
  }, []);

  // Fetch data when component mounts and when filters change
  useEffect(() => {
    fetchDefectData();
    fetchUsers();
  }, [user]);

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
                className="filter-btn"
              >
                Reset Filters
              </Button>
            </div>
          </Col>
        </Row>

        <Row>
          <Col>
            <div className="map-legend mb-3">
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
              
              {filteredDefects.map((defect) => (
                <Marker
                  key={defect.id}
                  position={defect.position}
                  icon={icons[defect.type]}
                >
                  <Popup>
                    <div className="defect-popup">
                      <h6>{defect.type.charAt(0).toUpperCase() + defect.type.slice(1)} Defect</h6>
                      <ul className="list-unstyled">
                        <li><strong>Count:</strong> {defect.defect_count}</li>
                        <li><strong>Date:</strong> {defect.timestamp}</li>
                        <li><strong>Reported by:</strong> {defect.username}</li>
                        {defect.type === 'crack' && defect.type_counts && (
                          <li>
                            <strong>Crack Types:</strong>
                            <ul>
                              {Object.entries(defect.type_counts).map(([type, count]) => (
                                <li key={type}>{type}: {count}</li>
                              ))}
                            </ul>
                          </li>
                        )}
                        {defect.type === 'kerb' && defect.condition_counts && (
                          <li>
                            <strong>Conditions:</strong>
                            <ul>
                              {Object.entries(defect.condition_counts).map(([condition, count]) => (
                                <li key={condition}>{condition}: {count}</li>
                              ))}
                            </ul>
                          </li>
                        )}
                      </ul>
                      <Link 
                        to={`/view/${defect.image_id}`} 
                        className="btn btn-sm btn-primary"
                        onClick={(e) => e.stopPropagation()}
                      >
                        View Details
                      </Link>
                    </div>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          </div>
        )}
      </Card.Body>
    </Card>
  );
}

export default DefectMap; 