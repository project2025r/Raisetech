import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card, Button, Alert, Form, Badge, Tabs, Tab } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import ResponsiveCheckboxList from '../components/ResponsiveCheckboxList';
import './recommendation.css';

const MATERIAL_EQUIPMENT_MAP = {
  "Mixes (cold mixed/hot mixed) for immediate use": [
    "Material truck (with hand tools)",
    "Equipment truck",
    "Asphalt mix carrying and laying equipment",
    "Compaction device (vibratory walk-behind roller or plate compactor)",
    "Mechanical brooms (for highways and urban roads)"
  ],
  "Storable cold mixes (cutback IRC:116/emulsion IRC:100)": [
    "Pug mill or concrete mixer for emulsion-aggregate mixes",
    "Material truck (with hand tools)",
    "Mechanical tool/equipment for pavement cutting and dressing",
    "Compaction device (vibratory walk-behind roller or plate compactor)"
  ],
  "Readymade mixes": [
    "Material truck (with hand tools)",
    "Mechanical tool/equipment for pavement cutting and dressing",
    "Compaction device (vibratory walk-behind roller or plate compactor)"
  ],
  "Cold mixes by patching machines": [
    "Machine Mixed Spot Cold Mix and Patching Equipment",
    "Mobile Mechanized Maintenance Units",
    "Jet Patching Velocity Spray Injection Technology",
    "Mechanical tool/equipment for pavement cutting and dressing"
  ],
  "Open-graded or dense-graded premix": [
    "Material truck (with hand tools)",
    "Compaction device (vibratory walk-behind roller or plate compactor)"
  ],
  "Prime coat": [
    "Material truck (with hand tools)"
  ],
  "Tack coat": [
    "Material truck (with hand tools)",
    "Bitumen Sprayer"
  ]
};

const EQUIPMENT_OPTIONS = [
  "Material truck (with hand tools)",
  "Equipment truck",
  "Mechanical tool/equipment for pavement cutting and dressing",
  "Compaction device (vibratory walk-behind roller or plate compactor)",
  "Air compressor with pavement cutter",
  "Asphalt mix carrying and laying equipment",
  "Traffic control devices and equipment",
  "Mechanical brooms (for highways and urban roads)",
  "Pug mill or concrete mixer for emulsion-aggregate mixes",
  "Hand rammer for compaction",
  "Small roller for compaction",
  "Drag spreader for smoothening surfaces",
  "Mechanical grit spreader for uniform aggregate spreading",
  "Machine Mixed Spot Cold Mix and Patching Equipment",
  "Mobile Mechanized Maintenance Units",
  "Jet Patching Velocity Spray Injection Technology",
  "Infrared road patching technology",
  "Stiff wire brush for cleaning potholes",
  "Compressed air jetting for removing loose materials",
  "Bitumen Sprayer"
];

/**
 * Recommendation component handles pothole repair recommendations
 * based on all potholes detected from the most recent image upload.
 * It provides both automatic recommendations based on detected potholes
 * and allows for manual selection of repair parameters.
 */
function Recommendation() {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState({
    labels: [],
    values: [],
    colors: []
  });

  // New state for pothole recommendation functionality
  const [activeTab, setActiveTab] = useState("manual");
  const [potholeData, setPotholeData] = useState([]);
  const [potholeStats, setPotholeStats] = useState({
    potholeCount: 0,
    avgPotholeVolume: 0,
    totalPotholeVolume: 0,
    roadLength: 500 // Default value
  });
  const [selectedMaterials, setSelectedMaterials] = useState([]);
  const [selectedEquipment, setSelectedEquipment] = useState([]);
  const [laborCounts, setLaborCounts] = useState({
    Unskilled: 0,
    Skilled: 0,
    Supervisors: 0
  });
  const [repairResults, setRepairResults] = useState(null);
  const [autoRecommendations, setAutoRecommendations] = useState(null);

  useEffect(() => {
    // Fetch recommendations data from backend
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Get existing recommendations
        const responseRecommendations = await axios.get('/api/recommendation/list');
        setRecommendations(responseRecommendations.data);
        
        // Get all potholes from the most recent image upload
        const responsePotholes = await axios.get('/api/potholes/recent');
        console.log('Pothole data from API:', responsePotholes.data);
        
        // Handle both the new and old data structure
        let potholeData = [];
        
        if (responsePotholes.data.potholes) {
          potholeData = responsePotholes.data.potholes;
          // Set image data if available in the new structure
          if (responsePotholes.data.image_data) {
            console.log('Image data from API:', responsePotholes.data.image_data);
          }
        }
        
        console.log('Processed pothole data:', potholeData);
        setPotholeData(potholeData);
        
        // Calculate aggregate statistics for all potholes in the most recent image
        if (potholeData.length > 0) {
          const count = potholeData.length;
          console.log('Number of potholes:', count);
          
          // Handle field naming inconsistencies if any (some may use volume, some Volume)
          const totalVolume = potholeData.reduce((sum, pothole) => {
            const volume = pothole.volume || pothole.Volume || 0;
            console.log('Pothole volume:', volume);
            return sum + volume;
          }, 0);
          
          console.log('Total volume:', totalVolume);
          const avgVolume = totalVolume / count;
          console.log('Average volume:', avgVolume);
          
          setPotholeStats({
            potholeCount: count,
            avgPotholeVolume: avgVolume,
            totalPotholeVolume: totalVolume,
            roadLength: 500 // This is a default value, could be calculated or fetched
          });
          
          // Transform data if needed to match what the API expects
          const formattedPotholeData = potholeData.map(pothole => ({
            volume: pothole.volume || pothole.Volume || 0,
            area_cm2: pothole.area_cm2 || pothole["area_cm2"] || 0,
            depth_cm: pothole.depth_cm || pothole["depth_cm"] || 0
          }));
          
          // Generate automatic recommendations based on all potholes in the most recent image
          getAutoRecommendations(formattedPotholeData);
        } else {
          // If no recent potholes, display a message
          setError('No potholes detected in the most recent image. Please upload and process an image in the Pavement section first.');
        }
        
        // Chart data processing (unchanged)
        const priorities = ['High', 'Medium', 'Low'];
        const counts = {
          'High': 0,
          'Medium': 0,
          'Low': 0
        };
        
        responseRecommendations.data.forEach(item => {
          counts[item.priority]++;
        });
        
        setChartData({
          labels: priorities,
          values: priorities.map(p => counts[p]),
          colors: ['#dc3545', '#ffc107', '#28a745']
        });
        
        setLoading(false);
      } catch (err) {
        setError('Error fetching data');
        setLoading(false);
        console.error('Error fetching data:', err);
      }
    };

    fetchData();
  }, []);

  // Handle automatic recommendations
  const getAutoRecommendations = async (potholeData) => {
    try {
      console.log('Sending pothole data to auto recommendations API:', potholeData);
      // Ensure we're passing all pothole data, not just the first entry
      const response = await axios.post('/api/recommendation/auto', { 
        potholeData: potholeData 
      });
      
      console.log('Auto recommendations response:', response.data);
      
      if (response.data && response.data.success && response.data.recommendations) {
        setAutoRecommendations(response.data.recommendations);
      } else {
        console.error('Auto recommendations response error:', response.data);
        setError('Failed to get automatic recommendations');
      }
    } catch (err) {
      console.error('Error getting auto recommendations:', err);
      setError('Error fetching automatic recommendations');
    }
  };

  // Handle material selection
  const handleMaterialChange = (e) => {
    const material = e.target.value;
    const isChecked = e.target.checked;
    
    if (isChecked) {
      setSelectedMaterials([...selectedMaterials, material]);
      
      // Auto-select equipment
      const requiredEquipment = MATERIAL_EQUIPMENT_MAP[material] || [];
      const newEquipment = [...new Set([...selectedEquipment, ...requiredEquipment])];
      setSelectedEquipment(newEquipment);
    } else {
      setSelectedMaterials(selectedMaterials.filter(item => item !== material));
    }
  };

  // Handle equipment selection
  const handleEquipmentChange = (e) => {
    const equipment = e.target.value;
    const isChecked = e.target.checked;
    
    if (isChecked) {
      setSelectedEquipment([...selectedEquipment, equipment]);
    } else {
      setSelectedEquipment(selectedEquipment.filter(item => item !== equipment));
    }
  };

  // Handle labor count changes
  const handleLaborChange = (type, value) => {
    setLaborCounts({
      ...laborCounts,
      [type]: parseInt(value) || 0
    });
  };

  // Calculate repair estimates
  const calculateRepairEstimates = async () => {
    try {
      const response = await axios.post('/api/recommendation/analyze', {
        selectedMaterials,
        selectedEquipment,
        laborCounts,
        potholeCount: potholeStats.potholeCount,
        avgPotholeVolume: potholeStats.avgPotholeVolume,
        roadLength: potholeStats.roadLength
      });
      
      setRepairResults(response.data.results);
    } catch (err) {
      console.error('Error calculating repair estimates:', err);
      setError('Error calculating repair estimates');
    }
  };

  const handleApprove = async (id) => {
    try {
      await axios.post(`/api/recommendation/approve/${id}`);
      // Update local state to reflect the change
      setRecommendations(prevRecs => 
        prevRecs.map(rec => 
          rec.id === id ? { ...rec, status: 'Approved' } : rec
        )
      );
    } catch (err) {
      console.error('Error approving recommendation:', err);
    }
  };

  const handleReject = async (id) => {
    try {
      await axios.post(`/api/recommendation/reject/${id}`);
      // Update local state to reflect the change
      setRecommendations(prevRecs => 
        prevRecs.map(rec => 
          rec.id === id ? { ...rec, status: 'Rejected' } : rec
        )
      );
    } catch (err) {
      console.error('Error rejecting recommendation:', err);
    }
  };

  const getPriorityBadgeClass = (priority) => {
    switch (priority) {
      case 'High':
        return 'bg-danger';
      case 'Medium':
        return 'bg-warning';
      case 'Low':
        return 'bg-success';
      default:
        return 'bg-secondary';
    }
  };

  const getStatusBadgeClass = (status) => {
    switch (status) {
      case 'Pending':
        return 'bg-secondary';
      case 'Approved':
        return 'bg-success';
      case 'Rejected':
        return 'bg-danger';
      default:
        return 'bg-secondary';
    }
  };

  return (
    <Container fluid className="mt-4">
      <Row>
        <Col md={12}>
          <Card className="mb-4 shadow-sm recommendation-card">
            <Card.Header className="recommendation-header">
              <Row>
                <Col>
                  <h5 className="mb-0">Pothole Repair Recommendations</h5>
                </Col>
                <Col xs="auto">
                  <div className="recommendation-type-buttons">
                    <Button
                      className={`recommendation-btn ${activeTab === "manual" ? "active" : "inactive"}`}
                      onClick={() => setActiveTab("manual")}
                      variant=""
                    >
                      Manual Recommendations
                    </Button>
                    <Button
                      className={`recommendation-btn ${activeTab === "auto" ? "active" : "inactive"}`}
                      onClick={() => setActiveTab("auto")}
                      variant=""
                    >
                      Automatic Recommendations
                    </Button>
                  </div>
                </Col>
              </Row>
            </Card.Header>
            <Card.Body>
              {/* Display pothole statistics at the top in a single row */}
              <Row className="mb-3">
                <Col md={12} className="mb-2">
                  <h6 className="text-center">Statistics from Most Recent Image Upload</h6>
                </Col>
              </Row>
              <Row className="stats-row mb-3 g-3">
                <Col md={3} sm={6}>
                  <Card className="recommendation-card h-100">
                    <Card.Body className="py-3">
                      <h6 className="mb-2">Detected Potholes</h6>
                      <h4 className="mb-0">{potholeStats.potholeCount}</h4>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={3} sm={6}>
                  <Card className="recommendation-card h-100">
                    <Card.Body className="py-3">
                      <h6 className="mb-2">Avg Pothole Volume</h6>
                      <h4 className="mb-0">{potholeStats.avgPotholeVolume.toFixed(2)} cm<sup>3</sup></h4>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={3} sm={6}>
                  <Card className="recommendation-card h-100">
                    <Card.Body className="py-3">
                      <h6 className="mb-2">Total Volume</h6>
                      <h4 className="mb-0">{potholeStats.totalPotholeVolume.toFixed(2)} cm<sup>3</sup></h4>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={3} sm={6}>
                  <Card className="recommendation-card h-100">
                    <Card.Body className="py-3">
                      <h6 className="mb-2">Road Length</h6>
                      <h4 className="mb-0">{potholeStats.roadLength} m</h4>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>

              {loading ? (
                <div className="text-center">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                </div>
              ) : error ? (
                <Alert variant="danger" className="p-3">{error}</Alert>
              ) : potholeData.length === 0 ? (
                <Alert variant="warning" className="p-3">
                  No pothole data available. Please process an image in the Pavement section first.
                </Alert>
              ) : (
                <Row>
                  <Col md={12}>
                    {activeTab === "manual" ? (
                      /* Manual Recommendations */
                      <div className="repair-parameters">
                        <h5 className="mb-3">Select Repair Parameters</h5>
                        <p className="text-muted mb-3">Based on all {potholeStats.potholeCount} potholes from the latest image</p>
                        
                        <Row className="lists-container">
                          <Col md={6}>
                            <h6 className="mt-4 mb-2">Materials</h6>
                            <ResponsiveCheckboxList
                              options={Object.keys(MATERIAL_EQUIPMENT_MAP)}
                              selectedValues={selectedMaterials}
                              onChange={(e) => handleMaterialChange(e)}
                              idPrefix="material"
                              className="repair-param-item"
                              containerClassName="repair-param-container"
                            />
                          </Col>
                          
                          <Col md={6}>
                            <h6 className="mt-4 mb-2">Equipment</h6>
                            <ResponsiveCheckboxList
                              options={EQUIPMENT_OPTIONS}
                              selectedValues={selectedEquipment}
                              onChange={(e) => handleEquipmentChange(e)}
                              idPrefix="equipment"
                              className="repair-param-item"
                              containerClassName="repair-param-container"
                            />
                          </Col>
                        </Row>
                        
                        <h6 className="mt-4 mb-2">Labor Requirements</h6>
                        <Row className="mb-4">
                          <Col xs={12} sm={4} className="mb-3">
                            <Form.Group>
                              <Form.Label>Unskilled Labor</Form.Label>
                              <Form.Control
                                type="number"
                                min="0"
                                value={laborCounts.Unskilled}
                                onChange={(e) => handleLaborChange('Unskilled', e.target.value)}
                              />
                            </Form.Group>
                          </Col>
                          <Col xs={12} sm={4} className="mb-3">
                            <Form.Group>
                              <Form.Label>Skilled Labor</Form.Label>
                              <Form.Control
                                type="number"
                                min="0"
                                value={laborCounts.Skilled}
                                onChange={(e) => handleLaborChange('Skilled', e.target.value)}
                              />
                            </Form.Group>
                          </Col>
                          <Col xs={12} sm={4} className="mb-3">
                            <Form.Group>
                              <Form.Label>Supervisors</Form.Label>
                              <Form.Control
                                type="number"
                                min="0"
                                value={laborCounts.Supervisors}
                                onChange={(e) => handleLaborChange('Supervisors', e.target.value)}
                              />
                            </Form.Group>
                          </Col>
                        </Row>
                        
                        <div className="d-grid gap-2 col-md-6 mx-auto mb-4">
                          <Button 
                            variant="primary" 
                            size="lg" 
                            onClick={calculateRepairEstimates}
                            disabled={selectedMaterials.length === 0}
                          >
                            Calculate Repair Estimates
                          </Button>
                        </div>

                        {repairResults && (
                          <div>
                            <h5 className="mb-3">Repair Analysis Summary</h5>
                            <Row>
                              <Col md={4}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Body>
                                    <h6>Pothole Category</h6>
                                    <h4>{repairResults.category}</h4>
                                  </Card.Body>
                                </Card>
                              </Col>
                              <Col md={4}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Body>
                                    <h6>Base Repair Cost</h6>
                                    <h4>₹{repairResults.base_cost_min.toFixed(2)} – ₹{repairResults.base_cost_max.toFixed(2)}</h4>
                                  </Card.Body>
                                </Card>
                                {repairResults.extra_equip_cost > 0 && (
                                  <Card className="mb-3 recommendation-card">
                                    <Card.Body>
                                      <h6>Additional Equipment Cost</h6>
                                      <h4>₹{repairResults.extra_equip_cost.toFixed(2)}</h4>
                                    </Card.Body>
                                  </Card>
                                )}
                              </Col>
                              <Col md={4}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Body>
                                    <h6>Estimated Repair Time</h6>
                                    <h4>{repairResults.total_time_minutes} minutes (~{repairResults.repair_days} day(s))</h4>
                                  </Card.Body>
                                </Card>
                              </Col>
                            </Row>
                            <Row>
                              <Col md={6}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Header>Total Repair Cost Estimate</Card.Header>
                                  <Card.Body>
                                    <h3>₹{repairResults.total_cost_min.toFixed(2)} – ₹{repairResults.total_cost_max.toFixed(2)}</h3>
                                  </Card.Body>
                                </Card>
                              </Col>
                              <Col md={6}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Header>Additional Considerations</Card.Header>
                                  <Card.Body>
                                    <p>The estimates adjust based on your input selections and the cost rubric.</p>
                                  </Card.Body>
                                </Card>
                              </Col>
                            </Row>
                          </div>
                        )}
                      </div>
                    ) : (
                      /* Automatic Recommendations */
                      <div>
                        {autoRecommendations ? (
                          <div>
                            <h4 className="mb-3">Pothole Type: {autoRecommendations.potholeType}</h4>
                            <p className="text-muted mb-3">Comprehensive analysis based on all {autoRecommendations.totalPotholes} potholes from the latest image upload</p>
                            <Row>
                              <Col md={6}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Body>
                                    <p><strong>Total Potholes Detected:</strong> {autoRecommendations.totalPotholes}</p>
                                    <p><strong>Average Pothole Volume:</strong> {autoRecommendations.avgVolume ? autoRecommendations.avgVolume.toFixed(2) : '0'} cm³</p>
                                    <p><strong>Total Volume:</strong> {autoRecommendations.totalVolume ? autoRecommendations.totalVolume.toFixed(2) : '0'} cm³</p>
                                    <p><strong>Manpower Required:</strong> {autoRecommendations.manpowerRequired}</p>
                                  </Card.Body>
                                </Card>
                              </Col>
                              <Col md={6}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Body>
                                    <p><strong>Cost Estimate (per Pothole):</strong> {autoRecommendations.costPerPothole}</p>
                                    <p><strong>Total Cost:</strong> {autoRecommendations.totalCost}</p>
                                    <p><strong>Traffic Disruption:</strong> {autoRecommendations.trafficDisruption}</p>
                                  </Card.Body>
                                </Card>
                              </Col>
                            </Row>
                            <Row>
                              <Col md={6}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Header>Materials & Equipment</Card.Header>
                                  <Card.Body>
                                    <p><strong>Materials Required:</strong> {autoRecommendations.materialsRequired}</p>
                                    <p><strong>Equipment Used:</strong> {autoRecommendations.equipmentUsed}</p>
                                    <p><strong>Time Taken (Per Pothole):</strong> {autoRecommendations.timePerPothole}</p>
                                    <p><strong>Durability:</strong> {autoRecommendations.durability}</p>
                                  </Card.Body>
                                </Card>
                              </Col>
                              <Col md={6}>
                                <Card className="mb-3 recommendation-card">
                                  <Card.Header>Risk Assessment</Card.Header>
                                  <Card.Body>
                                    <p><strong>What If Not Fixed?:</strong> {autoRecommendations.ifNotFixed}</p>
                                  </Card.Body>
                                </Card>
                              </Col>
                            </Row>
                          </div>
                        ) : (
                          <div className="text-center">
                            <div className="spinner-border text-primary" role="status">
                              <span className="visually-hidden">Loading automatic recommendations...</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </Col>
                </Row>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default Recommendation; 