import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card, Tabs, Tab, Form, Button } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import ChartContainer from '../components/ChartContainer';
import DefectMap from '../components/DefectMap';
import './dashboard.css';

// ImageCard component to isolate state for each image
const ImageCard = ({ defect, defectType, defectIdKey }) => {
  const [isOriginal, setIsOriginal] = useState(false);
  
  const toggleView = (showOriginal) => {
    setIsOriginal(showOriginal);
  };
  
  // Check if this is a multi-defect image
  const isMultiDefect = defect.detected_defects && defect.detected_defects.length > 1;
  const detectedDefects = defect.detected_defects || [];
  
  return (
    <Col md={4} className="mb-4" key={`${defectType}-${defect[defectIdKey]}`}>
      <Card className={`h-100 shadow-sm ${isMultiDefect ? 'border-warning' : ''}`}>
        <Card.Header className={isMultiDefect ? 'bg-warning bg-opacity-10' : ''}>
          <div className="d-flex justify-content-between align-items-center">
            <h6 className="mb-0">
              {defectType === 'cracks' ? `${defect.crack_type} #${defect.crack_id}` : 
               defectType === 'kerbs' ? `${defect.condition} #${defect.kerb_id}` : 
               `Pothole #${defect.pothole_id}`}
            </h6>
            {isMultiDefect && (
              <small className="text-warning fw-bold">
                🔀 Multi-Defect
              </small>
            )}
          </div>
          {isMultiDefect && (
            <div className="mt-1">
              <small className="text-muted">
                Also contains: {detectedDefects.filter(d => d !== defectType).join(', ')}
              </small>
            </div>
          )}
        </Card.Header>
        <Card.Body>
          <div className="mb-2 text-center">
            {defect.processed_image_id ? (
              <img 
                src={`/api/pavement/get-image/${isOriginal 
                  ? defect.original_image_id 
                  : defect.processed_image_id
                }`}
                alt={`${defectType === 'cracks' ? 'Crack' : defectType === 'kerbs' ? 'Kerb' : 'Pothole'} ${defect[defectIdKey]}`}
                className="img-fluid mb-2 border"
                style={{ maxHeight: "200px" }}
              />
            ) : (
              <div className="text-muted">No image available</div>
            )}
          </div>
          <div className="small">
            {defectType === 'potholes' && (
              <>
                <p className="mb-1"><strong>Area:</strong> {defect.area_cm2.toFixed(2)} cm²</p>
                <p className="mb-1"><strong>Depth:</strong> {defect.depth_cm.toFixed(2)} cm</p>
                <p className="mb-1"><strong>Volume:</strong> {defect.volume.toFixed(2)}</p>
              </>
            )}
            {defectType === 'cracks' && (
              <>
                <p className="mb-1"><strong>Type:</strong> {defect.crack_type}</p>
                <p className="mb-1"><strong>Area:</strong> {defect.area_cm2.toFixed(2)} cm²</p>
                <p className="mb-1"><strong>Range:</strong> {defect.area_range}</p>
              </>
            )}
            {defectType === 'kerbs' && (
              <>
                <p className="mb-1"><strong>Type:</strong> {defect.kerb_type}</p>
                <p className="mb-1"><strong>Length:</strong> {defect.length_m && defect.length_m.toFixed(2)} m</p>
                <p className="mb-1"><strong>Condition:</strong> {defect.condition}</p>
              </>
            )}
            <p className="mb-1"><strong>Uploaded by:</strong> {defect.username}</p>
            <p className="mb-1"><strong>Timestamp:</strong> {new Date(defect.timestamp).toLocaleString()}</p>
            <div className="mt-2">
              <Button
                variant={isOriginal ? 'primary' : 'outline-primary'}
                size="sm"
                className="me-2"
                onClick={() => toggleView(true)}
              >
                Original
              </Button>
              <Button
                variant={!isOriginal ? 'success' : 'outline-success'}
                size="sm"
                onClick={() => toggleView(false)}
              >
                Processed
              </Button>
            </div>
          </div>
        </Card.Body>
      </Card>
    </Col>
  );
};

function Dashboard({ user }) {
  const [statistics, setStatistics] = useState({
    potholesDetected: 0,
    cracksDetected: 0,
    kerbsDetected: 0,
    totalUsers: 0
  });
  const [weeklyData, setWeeklyData] = useState({
    days: [],
    issues: []
  });
  const [issuesByType, setIssuesByType] = useState({
    types: [],
    counts: []
  });
  const [dashboardData, setDashboardData] = useState({
    potholes: {
      count: 0,
      by_size: {},
      avg_volume: 0,
      latest: []
    },
    cracks: {
      count: 0,
      by_type: {},
      by_size: {},
      latest: []
    },
    kerbs: {
      count: 0,
      by_condition: {},
      latest: []
    },
    users: {
      count: 0,
      by_role: {},
      latest: []
    }
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Date filter state
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [dateFilterApplied, setDateFilterApplied] = useState(false);

  // User filter state
  const [usersList, setUsersList] = useState([]);
  const [selectedUser, setSelectedUser] = useState('');
  const [userFilterApplied, setUserFilterApplied] = useState(false);

  // Defect type filter state
  const [defectFilters, setDefectFilters] = useState({
    potholes: true,
    cracks: true,
    kerbs: true
  });
  
  // Filtered issues state
  const [filteredIssuesByType, setFilteredIssuesByType] = useState({
    types: [],
    counts: []
  });

  // Get current date and 30 days ago for default date range
  useEffect(() => {
    const currentDate = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    
    setEndDate(currentDate.toISOString().split('T')[0]);
    setStartDate(thirtyDaysAgo.toISOString().split('T')[0]);
  }, []);

  // Fetch dashboard data from backend
  const fetchData = async (filters = {}) => {
    try {
      setLoading(true);
      
      // Add filters to requests if provided
      const params = {};
      if (filters.startDate) params.start_date = filters.startDate;
      if (filters.endDate) params.end_date = filters.endDate;
      if (filters.username) params.username = filters.username;
      if (user?.role) params.user_role = user.role;
      
      // Get overview statistics
      const statsResponse = await axios.get('/api/dashboard/statistics', { params });
      if (statsResponse.data.success) {
        setStatistics({
          potholesDetected: statsResponse.data.data.issues_by_type.potholes,
          cracksDetected: statsResponse.data.data.issues_by_type.cracks,
          kerbsDetected: statsResponse.data.data.issues_by_type.kerbs,
          totalUsers: statistics.totalUsers // Preserve the existing user count
        });
      }
      
      // Get weekly trend data
      const weeklyResponse = await axios.get('/api/dashboard/weekly-trend', { params });
      setWeeklyData({
        days: weeklyResponse.data.days,
        issues: weeklyResponse.data.issues
      });
      
      // Get issues by type
      const typesResponse = await axios.get('/api/dashboard/issues-by-type', { params });
      setIssuesByType({
        types: typesResponse.data.types,
        counts: typesResponse.data.counts
      });
      
      // Get detailed dashboard data including latest images
      const dashboardResponse = await axios.get('/api/dashboard/summary', { params });
      if (dashboardResponse.data.success) {
        const dashboardData = dashboardResponse.data.data;
        
        // Calculate multi-defect statistics
        const multiDefectStats = {
          totalImages: 0,
          multiDefectImages: 0,
          singleDefectImages: 0,
          categoryBreakdown: {
            potholes: 0,
            cracks: 0,
            kerbs: 0
          }
        };
        
        // Count multi-defect images from each category
        ['potholes', 'cracks', 'kerbs'].forEach(category => {
          const categoryImages = dashboardData[category].latest || [];
          categoryImages.forEach(item => {
            if (item.multi_defect_image) {
              multiDefectStats.multiDefectImages++;
            }
            multiDefectStats.totalImages++;
            multiDefectStats.categoryBreakdown[category]++;
          });
        });
        
        multiDefectStats.singleDefectImages = multiDefectStats.totalImages - multiDefectStats.multiDefectImages;
        
        // Add multi-defect statistics to dashboard data
        dashboardData.multiDefectStats = multiDefectStats;
        
        setDashboardData(dashboardData);
      }
      
      // Get users data
      try {
        const usersResponse = await axios.get('/api/users/summary', { params });
        if (usersResponse.data.success) {
          setStatistics(prevStats => ({
            ...prevStats,
            totalUsers: usersResponse.data.total_users || 0
          }));
          
          // Ensure users data is properly set in dashboardData
          setDashboardData(prevData => ({
            ...prevData,
            users: {
              count: usersResponse.data.total_users || 0,
              by_role: usersResponse.data.roles_distribution || {},
              latest: usersResponse.data.recent_users || []
            }
          }));
        }
      } catch (userErr) {
        console.error('Error fetching user data:', userErr);
        // Non-critical error, continue with other data
      }
      
      setLoading(false);
    } catch (err) {
      setError('Error fetching dashboard data');
      setLoading(false);
      console.error('Error fetching data:', err);
    }
  };

  // Initial data fetch
  useEffect(() => {
    if (startDate && endDate) {
      fetchData({ startDate, endDate });
    }
  }, []);

  // Fetch users list for filter dropdown
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const params = {};
        if (user?.role) params.user_role = user.role;
        
        const response = await axios.get('/api/users/all', { params });
        if (response.data.success) {
          // Users are already filtered by the backend based on RBAC
          setUsersList(response.data.users);
        }
      } catch (error) {
        console.error('Error fetching users:', error);
      }
    };
    
    fetchUsers();
  }, [user]);

  // Handle date filter application
  const handleApplyDateFilter = () => {
    if (startDate && endDate) {
      fetchData({ 
        startDate, 
        endDate,
        username: selectedUser || undefined
      });
      setDateFilterApplied(true);
    }
  };

  // Handle date filter reset
  const handleResetDateFilter = () => {
    const currentDate = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    
    const newEndDate = currentDate.toISOString().split('T')[0];
    const newStartDate = thirtyDaysAgo.toISOString().split('T')[0];
    
    setEndDate(newEndDate);
    setStartDate(newStartDate);
    fetchData({ 
      startDate: newStartDate, 
      endDate: newEndDate,
      username: selectedUser || undefined
    });
    setDateFilterApplied(false);
  };

  // Handle user filter application
  const handleApplyUserFilter = () => {
    fetchData({
      startDate,
      endDate,
      username: selectedUser || undefined
    });
    setUserFilterApplied(!!selectedUser);
  };

  // Handle user filter reset
  const handleResetUserFilter = () => {
    setSelectedUser('');
    fetchData({
      startDate,
      endDate
    });
    setUserFilterApplied(false);
  };

  // Handle user selection
  const handleUserChange = (e) => {
    setSelectedUser(e.target.value);
  };

  // Filter the issues by type whenever the filters or data changes
  useEffect(() => {
    if (issuesByType.types.length > 0) {
      const filteredTypes = [];
      const filteredCounts = [];
      
      issuesByType.types.forEach((type, index) => {
        if (
          (type.includes('Pothole') && defectFilters.potholes) ||
          (type.includes('Crack') && defectFilters.cracks) ||
          (type.includes('Kerb') && defectFilters.kerbs)
        ) {
          filteredTypes.push(type);
          filteredCounts.push(issuesByType.counts[index]);
        }
      });
      
      setFilteredIssuesByType({
        types: filteredTypes,
        counts: filteredCounts
      });
    }
  }, [issuesByType, defectFilters]);

  // Handle defect filter change
  const handleDefectFilterChange = (defectType) => {
    setDefectFilters(prev => ({
      ...prev,
      [defectType]: !prev[defectType]
    }));
  };

  return (
    <Container fluid className="dashboard-container">
      <h2 className="my-4">Dashboard</h2>
      
      {/* Filters Card */}
      <Card className="mb-4 shadow-sm dashboard-card filters-card">
        <Card.Header className="bg-primary text-white">
          <h5 className="mb-0">Filters</h5>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col lg={6} className="filter-section">
              <h6>Date Range</h6>
              <div className="filter-controls">
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
                <div className="filter-actions">
                  <Button
                    size="sm"
                    variant="primary"
                    onClick={handleApplyDateFilter}
                    disabled={!startDate || !endDate}
                  >
                    Apply
                  </Button>
                  <Button
                    size="sm"
                    variant="outline-secondary"
                    onClick={handleResetDateFilter}
                    disabled={!dateFilterApplied}
                  >
                    Reset
                  </Button>
                </div>
              </div>
              {dateFilterApplied && (
                <div className="filter-status text-success">
                  Showing data from {new Date(startDate).toLocaleDateString()} to {new Date(endDate).toLocaleDateString()}
                </div>
              )}
            </Col>
            <Col lg={6} className="filter-section">
              <h6>User Filter</h6>
              <div className="filter-controls">
                <div className="filter-field">
                  <Form.Group>
                    <Form.Label className="small">Select User</Form.Label>
                    <Form.Select
                      value={selectedUser}
                      onChange={handleUserChange}
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
                <div className="filter-actions">
                  <Button
                    size="sm"
                    variant="primary"
                    onClick={handleApplyUserFilter}
                  >
                    Apply
                  </Button>
                  <Button
                    size="sm"
                    variant="outline-secondary"
                    onClick={handleResetUserFilter}
                    disabled={!userFilterApplied}
                  >
                    Reset
                  </Button>
                </div>
              </div>
              {userFilterApplied && (
                <div className="filter-status text-success">
                  Showing data for user: {selectedUser}
                </div>
              )}
            </Col>
          </Row>
        </Card.Body>
      </Card>
      
      {loading ? (
        <div className="text-center">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      ) : error ? (
        <div className="alert alert-danger">{error}</div>
      ) : (
        <>
          {/* Top Stats Cards */}
          <Row className="mb-4">
            <Col md={3}>
              <Card className="h-100 shadow-sm dashboard-card">
                <Card.Body className="text-center">
                  <h5 className="card-title">Potholes</h5>
                  <h2 className="text-primary">{statistics.potholesDetected}</h2>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="h-100 shadow-sm dashboard-card">
                <Card.Body className="text-center">
                  <h5 className="card-title">Cracks</h5>
                  <h2 className="text-primary">{statistics.cracksDetected}</h2>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="h-100 shadow-sm dashboard-card">
                <Card.Body className="text-center">
                  <h5 className="card-title">Kerbs</h5>
                  <h2 className="text-primary">{statistics.kerbsDetected}</h2>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="h-100 shadow-sm dashboard-card">
                <Card.Body className="text-center">
                  <h5 className="card-title">Users</h5>
                  <h2 className="text-success">{statistics.totalUsers}</h2>
                </Card.Body>
              </Card>
            </Col>
          </Row>


          
          {/* Users Overview Section */}
          <Row className="mb-4">
            <Col md={12}>
              <Card className="shadow-sm dashboard-card">
                <Card.Header className="bg-primary text-white">
                  <h5 className="mb-0">Users Overview</h5>
                </Card.Header>
                <Card.Body>
                  <Row>
                    <Col md={6}>
                      <h6 className="mb-3">User Distribution by Role</h6>
                      <ChartContainer
                        data={[
                          {
                            type: 'pie',
                            labels: Object.keys(dashboardData.users?.by_role || {}),
                            values: Object.values(dashboardData.users?.by_role || {}),
                            marker: {
                              colors: ['#007bff', '#28a745', '#dc3545', '#6c757d']
                            },
                            textinfo: "label+percent",
                            insidetextorientation: "radial"
                          }
                        ]}
                        isPieChart={true}
                      />
                    </Col>
                    <Col md={6}>
                      <h6 className="mb-3">Recent Users</h6>
                      <div className="table-responsive">
                        <table className="table table-sm table-hover">
                          <thead>
                            <tr>
                              <th>Username</th>
                              <th>Role</th>
                              <th>Last Login</th>
                            </tr>
                          </thead>
                          <tbody>
                            {dashboardData.users?.latest && dashboardData.users.latest.length > 0 ? (
                              dashboardData.users.latest.map((user, index) => (
                                <tr key={`user-${index}`}>
                                  <td>{user.username}</td>
                                  <td>
                                    <span className={`badge bg-${
                                      user.role === 'admin' ? 'danger' : 
                                      user.role === 'manager' ? 'warning' : 
                                      'primary'
                                    }`}>
                                      {user.role}
                                    </span>
                                  </td>
                                  <td>{new Date(user.last_login).toLocaleString()}</td>
                                </tr>
                              ))
                            ) : (
                              <tr>
                                <td colSpan="3" className="text-center">No recent user activity</td>
                              </tr>
                            )}
                          </tbody>
                        </table>
                      </div>
                    </Col>
                  </Row>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          {/* Charts Row */}
          <Row>
            <Col md={6}>
              <Card className="mb-4 shadow-sm dashboard-card">
                <Card.Header className="bg-primary text-white">
                  <h5 className="mb-0">Weekly Detection Trend</h5>
                </Card.Header>
                <Card.Body>
                  <ChartContainer
                    data={[
                      {
                        x: weeklyData.days,
                        y: weeklyData.issues,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: { color: '#007bff' }
                      }
                    ]}
                    layout={{
                      xaxis: { title: 'Day' },
                      yaxis: { title: 'Issues Detected' }
                    }}
                  />
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card className="mb-4 shadow-sm dashboard-card">
                <Card.Header className="bg-primary text-white">
                  <h5 className="mb-0">Issues by Type</h5>
                </Card.Header>
                <Card.Body>
                  <ChartContainer
                    data={[
                      {
                        type: 'bar',
                        x: filteredIssuesByType.types,
                        y: filteredIssuesByType.counts,
                        marker: {
                          color: filteredIssuesByType.types.map(type => {
                            if (type.includes('Pothole')) return '#007bff';
                            if (type.includes('Crack')) return '#28a745';
                            if (type.includes('Kerb')) return '#dc3545';
                            return '#6c757d';
                          })
                        }
                      }
                    ]}
                    layout={{
                      xaxis: { 
                        title: 'Issue Type',
                        tickangle: -45,
                        automargin: true
                      },
                      yaxis: { title: 'Count' },
                      margin: { t: 10, b: 80, l: 50, r: 10 } // More compact margins
                    }}
                    showLegend={true}
                    legendItems={[
                      {
                        label: 'Potholes',
                        color: '#007bff',
                        checked: defectFilters.potholes,
                        onChange: () => handleDefectFilterChange('potholes')
                      },
                      {
                        label: 'Cracks',
                        color: '#28a745',
                        checked: defectFilters.cracks,
                        onChange: () => handleDefectFilterChange('cracks')
                      },
                      {
                        label: 'Kerbs',
                        color: '#dc3545',
                        checked: defectFilters.kerbs,
                        onChange: () => handleDefectFilterChange('kerbs')
                      }
                    ]}
                    className="compact-legend"
                  />
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* Infrastructure Health Row */}
          <Row className="mb-4">
            <Col md={12}>
              <Card className="shadow-sm dashboard-card">
                <Card.Header className="bg-primary text-white">
                  <h5 className="mb-0">Infrastructure Distribution</h5>
                </Card.Header>
                <Card.Body>
                  <ChartContainer
                    data={[
                      {
                        type: 'pie',
                        labels: ['Potholes', 'Cracks', 'Kerbs'],
                        values: [
                          statistics.potholesDetected,
                          statistics.cracksDetected,
                          statistics.kerbsDetected
                        ],
                        marker: {
                          colors: ['#007bff', '#28a745', '#dc3545']
                        },
                        textinfo: "label+percent",
                        insidetextorientation: "radial"
                      }
                    ]}
                    layout={{
                      height: 350
                    }}
                    isPieChart={true}
                  />
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          {/* Defect Map Section */}
                          <DefectMap user={user} />
          
          {/* Recently Uploaded Images Section */}
          <Row className="mb-4">
            <Col md={12}>
              <Card className="shadow-sm dashboard-card">
                <Card.Header className="bg-primary text-white">
                  <h5 className="mb-0">All Uploaded Images</h5>
                </Card.Header>
                <Card.Body>
                  <Tabs defaultActiveKey="potholes" className="mb-3">
                    <Tab eventKey="potholes" title={`Potholes (${dashboardData.potholes.latest.length})`}>
                      <div style={{ maxHeight: '700px', overflowY: 'auto', paddingRight: '10px' }}>
                        <Row>
                          {dashboardData.potholes.latest.map((pothole) => (
                            <ImageCard 
                              key={`pothole-${pothole.pothole_id}`}
                              defect={pothole} 
                              defectType="potholes" 
                              defectIdKey="pothole_id" 
                            />
                          ))}
                        </Row>
                      </div>
                    </Tab>
                    <Tab eventKey="cracks" title={`Cracks (${dashboardData.cracks.latest.length})`}>
                      <div style={{ maxHeight: '700px', overflowY: 'auto', paddingRight: '10px' }}>
                        <Row>
                          {dashboardData.cracks.latest.map((crack) => (
                            <ImageCard 
                              key={`crack-${crack.crack_id}`}
                              defect={crack} 
                              defectType="cracks" 
                              defectIdKey="crack_id" 
                            />
                          ))}
                        </Row>
                      </div>
                    </Tab>
                    <Tab eventKey="kerbs" title={`Kerbs (${dashboardData.kerbs.latest.length})`}>
                      <div style={{ maxHeight: '700px', overflowY: 'auto', paddingRight: '10px' }}>
                        <Row>
                          {dashboardData.kerbs && dashboardData.kerbs.latest && dashboardData.kerbs.latest.length > 0 ? (
                            dashboardData.kerbs.latest.map((kerb) => (
                              <ImageCard 
                                key={`kerb-${kerb.kerb_id}`}
                                defect={kerb} 
                                defectType="kerbs" 
                                defectIdKey="kerb_id" 
                              />
                            ))
                          ) : (
                            <Col>
                              <div className="alert alert-info">
                                No kerb images available yet. Upload some kerb images using the Pavement Analysis tool.
                              </div>
                            </Col>
                          )}
                        </Row>
                      </div>
                    </Tab>
                  </Tabs>
                </Card.Body>
              </Card>
            </Col>
          </Row>

         
        </>
      )}
    </Container>
  );
}

export default Dashboard; 