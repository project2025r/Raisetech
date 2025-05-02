import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card, Tabs, Tab, Form, Button } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import ChartContainer from '../components/ChartContainer';
import './dashboard.css';

function Dashboard() {
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

  // Image toggle state
  const [displayedImages, setDisplayedImages] = useState({
    potholes: {}, // Will store pothole_id -> 'original' or 'processed'
    cracks: {},   // Will store crack_id -> 'original' or 'processed' 
    kerbs: {}     // Will store kerb_id -> 'original' or 'processed'
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
  const fetchData = async (dateFilters = {}) => {
    try {
      setLoading(true);
      
      // Add date filters to requests if provided
      const params = {};
      if (dateFilters.startDate) params.start_date = dateFilters.startDate;
      if (dateFilters.endDate) params.end_date = dateFilters.endDate;
      
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
        setDashboardData(dashboardResponse.data.data);
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

  // Handle date filter application
  const handleApplyDateFilter = () => {
    if (startDate && endDate) {
      fetchData({ startDate, endDate });
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
    fetchData({ startDate: newStartDate, endDate: newEndDate });
    setDateFilterApplied(false);
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

  // Initialize images to processed by default
  useEffect(() => {
    if (dashboardData.potholes.latest.length > 0 || 
        dashboardData.cracks.latest.length > 0 || 
        dashboardData.kerbs.latest.length > 0) {
      
      // Create initialized objects for each defect type
      const potholeImages = {};
      const crackImages = {};
      const kerbImages = {};
      
      // Initialize pothole images to 'processed' by default
      dashboardData.potholes.latest.forEach(pothole => {
        potholeImages[pothole.pothole_id] = 'processed';
      });
      
      // Initialize crack images to 'processed' by default
      dashboardData.cracks.latest.forEach(crack => {
        crackImages[crack.crack_id] = 'processed';
      });
      
      // Initialize kerb images to 'processed' by default
      dashboardData.kerbs.latest.forEach(kerb => {
        kerbImages[kerb.kerb_id] = 'processed';
      });
      
      // Update the state with initialized images
      setDisplayedImages({
        potholes: potholeImages,
        cracks: crackImages,
        kerbs: kerbImages
      });
    }
  }, [dashboardData]);

  // Updated toggle image function with console logging for debugging
  const toggleImage = (type, id, imageType) => {
    console.log(`Toggling ${type} image for ID ${id} to ${imageType}`);
    
    setDisplayedImages(prev => {
      // Create a copy of the specific defect type object
      const updatedTypeImages = { ...prev[type] };
      
      // Update only the specific ID
      updatedTypeImages[id] = imageType;
      
      // Return the complete updated state
      return {
        ...prev,
        [type]: updatedTypeImages
      };
    });
  };

  return (
    <Container fluid className="mt-4">
      <h1 className="text-center mb-4">Dashboard</h1>
      
      {/* Date Filter */}
      <Card className="mb-4 shadow-sm">
        <Card.Body>
          <Row className="date-filter-container">
            <Col md={6}>
              <Form.Group className="mb-3">
                <Form.Label>Start Date</Form.Label>
                <Form.Control
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </Form.Group>
              <Form.Group className="mb-3">
                <Form.Label>End Date</Form.Label>
                <Form.Control
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </Form.Group>
            </Col>
            <Col md={6} className="d-flex align-items-end date-filter-buttons">
              <Button 
                variant="primary" 
                onClick={handleApplyDateFilter}
                className="me-2"
                disabled={!startDate || !endDate}
              >
                Apply Filter
              </Button>
              <Button 
                variant="outline-secondary" 
                onClick={handleResetDateFilter}
                disabled={!dateFilterApplied}
              >
                Reset
              </Button>
              {dateFilterApplied && (
                <span className="ms-3 text-success">
                  Showing data from {new Date(startDate).toLocaleDateString()} to {new Date(endDate).toLocaleDateString()}
                </span>
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
          
          {/* Recently Uploaded Images Section */}
          <Row className="mb-4">
            <Col md={12}>
              <Card className="shadow-sm">
                <Card.Header className="bg-primary text-white">
                  <h5 className="mb-0">All Uploaded Images</h5>
                </Card.Header>
                <Card.Body>
                  <Tabs defaultActiveKey="potholes" className="mb-3">
                    <Tab eventKey="potholes" title={`Potholes (${dashboardData.potholes.latest.length})`}>
                      <div style={{ maxHeight: '700px', overflowY: 'auto', paddingRight: '10px' }}>
                        <Row>
                          {dashboardData.potholes.latest.map((pothole, index) => (
                            <Col md={4} className="mb-4" key={`pothole-${index}`}>
                              <Card>
                                <Card.Header>
                                  <h6 className="mb-0">Pothole #{pothole.pothole_id}</h6>
                                </Card.Header>
                                <Card.Body>
                                  <div className="mb-2 text-center">
                                    {pothole.processed_image_id ? (
                                      <img 
                                        src={`/api/pavement/get-image/${
                                          displayedImages.potholes && displayedImages.potholes[pothole.pothole_id] === 'original'
                                            ? pothole.original_image_id 
                                            : pothole.processed_image_id
                                        }`}
                                        alt={`Pothole ${pothole.pothole_id}`}
                                        className="img-fluid mb-2 border"
                                        style={{ maxHeight: "200px" }}
                                      />
                                    ) : (
                                      <div className="text-muted">No image available</div>
                                    )}
                                  </div>
                                  <div className="small">
                                    <p className="mb-1"><strong>Area:</strong> {pothole.area_cm2.toFixed(2)} cm²</p>
                                    <p className="mb-1"><strong>Depth:</strong> {pothole.depth_cm.toFixed(2)} cm</p>
                                    <p className="mb-1"><strong>Volume:</strong> {pothole.volume.toFixed(2)}</p>
                                    <p className="mb-1"><strong>Uploaded by:</strong> {pothole.username}</p>
                                    <p className="mb-1"><strong>Timestamp:</strong> {new Date(pothole.timestamp).toLocaleString()}</p>
                                    <div className="mt-2">
                                      <Button
                                        variant={displayedImages.potholes && displayedImages.potholes[pothole.pothole_id] === 'original' 
                                          ? 'primary' 
                                          : 'outline-primary'}
                                        size="sm"
                                        className="me-2"
                                        onClick={() => toggleImage('potholes', pothole.pothole_id, 'original')}
                                      >
                                        Original
                                      </Button>
                                      <Button
                                        variant={displayedImages.potholes && displayedImages.potholes[pothole.pothole_id] !== 'original' 
                                          ? 'success' 
                                          : 'outline-success'}
                                        size="sm"
                                        onClick={() => toggleImage('potholes', pothole.pothole_id, 'processed')}
                                      >
                                        Processed
                                      </Button>
                                    </div>
                                  </div>
                                </Card.Body>
                              </Card>
                            </Col>
                          ))}
                        </Row>
                      </div>
                    </Tab>
                    <Tab eventKey="cracks" title={`Cracks (${dashboardData.cracks.latest.length})`}>
                      <div style={{ maxHeight: '700px', overflowY: 'auto', paddingRight: '10px' }}>
                        <Row>
                          {dashboardData.cracks.latest.map((crack, index) => (
                            <Col md={4} className="mb-4" key={`crack-${index}`}>
                              <Card>
                                <Card.Header>
                                  <h6 className="mb-0">{crack.crack_type} #{crack.crack_id}</h6>
                                </Card.Header>
                                <Card.Body>
                                  <div className="mb-2 text-center">
                                    {crack.processed_image_id ? (
                                      <img 
                                        src={`/api/pavement/get-image/${
                                          displayedImages.cracks && displayedImages.cracks[crack.crack_id] === 'original' 
                                            ? crack.original_image_id 
                                            : crack.processed_image_id
                                        }`}
                                        alt={`Crack ${crack.crack_id}`}
                                        className="img-fluid mb-2 border"
                                        style={{ maxHeight: "200px" }}
                                      />
                                    ) : (
                                      <div className="text-muted">No image available</div>
                                    )}
                                  </div>
                                  <div className="small">
                                    <p className="mb-1"><strong>Type:</strong> {crack.crack_type}</p>
                                    <p className="mb-1"><strong>Area:</strong> {crack.area_cm2.toFixed(2)} cm²</p>
                                    <p className="mb-1"><strong>Range:</strong> {crack.area_range}</p>
                                    <p className="mb-1"><strong>Uploaded by:</strong> {crack.username}</p>
                                    <p className="mb-1"><strong>Timestamp:</strong> {new Date(crack.timestamp).toLocaleString()}</p>
                                    <div className="mt-2">
                                      <Button
                                        variant={displayedImages.cracks && displayedImages.cracks[crack.crack_id] === 'original' 
                                          ? 'primary' 
                                          : 'outline-primary'}
                                        size="sm"
                                        className="me-2"
                                        onClick={() => toggleImage('cracks', crack.crack_id, 'original')}
                                      >
                                        Original
                                      </Button>
                                      <Button
                                        variant={displayedImages.cracks && displayedImages.cracks[crack.crack_id] !== 'original' 
                                          ? 'success' 
                                          : 'outline-success'}
                                        size="sm"
                                        onClick={() => toggleImage('cracks', crack.crack_id, 'processed')}
                                      >
                                        Processed
                                      </Button>
                                    </div>
                                  </div>
                                </Card.Body>
                              </Card>
                            </Col>
                          ))}
                        </Row>
                      </div>
                    </Tab>
                    <Tab eventKey="kerbs" title={`Kerbs (${dashboardData.kerbs.latest.length})`}>
                      <div style={{ maxHeight: '700px', overflowY: 'auto', paddingRight: '10px' }}>
                        <Row>
                          {dashboardData.kerbs && dashboardData.kerbs.latest && dashboardData.kerbs.latest.length > 0 ? (
                            dashboardData.kerbs.latest.map((kerb, index) => (
                              <Col md={4} className="mb-4" key={`kerb-${index}`}>
                                <Card>
                                  <Card.Header>
                                    <h6 className="mb-0">{kerb.condition} #{kerb.kerb_id}</h6>
                                  </Card.Header>
                                  <Card.Body>
                                    <div className="mb-2 text-center">
                                      {kerb.processed_image_id ? (
                                        <img 
                                          src={`/api/pavement/get-image/${
                                            displayedImages.kerbs && displayedImages.kerbs[kerb.kerb_id] === 'original' 
                                              ? kerb.original_image_id 
                                              : kerb.processed_image_id
                                          }`}
                                          alt={`Kerb ${kerb.kerb_id}`}
                                          className="img-fluid mb-2 border"
                                          style={{ maxHeight: "200px" }}
                                        />
                                      ) : (
                                        <div className="text-muted">No image available</div>
                                      )}
                                    </div>
                                    <div className="small">
                                      <p className="mb-1"><strong>Type:</strong> {kerb.kerb_type}</p>
                                      <p className="mb-1"><strong>Length:</strong> {kerb.length_m && kerb.length_m.toFixed(2)} m</p>
                                      <p className="mb-1"><strong>Condition:</strong> {kerb.condition}</p>
                                      <p className="mb-1"><strong>Uploaded by:</strong> {kerb.username}</p>
                                      <p className="mb-1"><strong>Timestamp:</strong> {new Date(kerb.timestamp).toLocaleString()}</p>
                                      <div className="mt-2">
                                        <Button
                                          variant={displayedImages.kerbs && displayedImages.kerbs[kerb.kerb_id] === 'original' 
                                            ? 'primary' 
                                            : 'outline-primary'}
                                          size="sm"
                                          className="me-2"
                                          onClick={() => toggleImage('kerbs', kerb.kerb_id, 'original')}
                                        >
                                          Original
                                        </Button>
                                        <Button
                                          variant={displayedImages.kerbs && displayedImages.kerbs[kerb.kerb_id] !== 'original' 
                                            ? 'success' 
                                            : 'outline-success'}
                                          size="sm"
                                          onClick={() => toggleImage('kerbs', kerb.kerb_id, 'processed')}
                                        >
                                          Processed
                                        </Button>
                                      </div>
                                    </div>
                                  </Card.Body>
                                </Card>
                              </Col>
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