import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { FaMap, FaColumns, FaLightbulb, FaChartBar } from 'react-icons/fa';
import './Home.css';

const Home = ({ user }) => {
  const modules = [
    {
      title: 'Pavement',
      description: 'Detect and analyze potholes and cracks in road pavement',
      icon: <FaMap size={48} className="module-icon-svg" />,
      path: '/pavement',
      color: '#3498db'
    },
    {
      title: 'Infrastructure',
      description: 'Identify and monitor road infrastructure elements such as kerbs',
      icon: <FaColumns size={48} className="module-icon-svg" />,
      path: '/road-infrastructure',
      color: '#2ecc71'
    },
    {
      title: 'Recommendation',
      description: 'Get repair recommendations and cost estimates for detected issues',
      icon: <FaLightbulb size={48} className="module-icon-svg" />,
      path: '/recommendation',
      color: '#f39c12'
    },
    {
      title: 'Dashboard',
      description: 'View analytics and statistics from collected data',
      icon: <FaChartBar size={48} className="module-icon-svg" />,
      path: '/dashboard',
      color: '#9b59b6'
    }
  ];

  return (
    <Container fluid="md" className="home-page">
      <Row className="mb-4">
        <Col>
          <h2 className="big-font text-center text-md-start">Welcome, {user?.username || 'User'}</h2>
          <p className="welcome-message text-center text-md-start">
            Select a module below to start using the Road AI Safety Enhancement system.
          </p>
        </Col>
      </Row>

      <Row className="row-cols-1 row-cols-sm-2 row-cols-lg-4 g-3 g-md-4">
        {modules.map((module, index) => (
          <Col key={index}>
            <Link to={module.path} className="text-decoration-none">
              <Card className="home-card h-100">
                <Card.Body className="d-flex flex-column align-items-center">
                  <div className="module-icon" style={{ color: module.color }}>
                    {module.icon}
                  </div>
                  <Card.Title className="module-title">{module.title}</Card.Title>
                  <Card.Text className="module-description">
                    {module.description}
                  </Card.Text>
                </Card.Body>
              </Card>
            </Link>
          </Col>
        ))}
      </Row>

      <Row className="mt-4">
        <Col>
          <Card className="info-card">
            <Card.Body>
              <Card.Title className="text-center text-md-start">About Road AI Safety Enhancement</Card.Title>
              <Card.Text>
                This system uses advanced computer vision and artificial intelligence 
                to detect and analyze road conditions, helping maintain safer road
                infrastructure. The technology can identify potholes, various types of
                cracks, and road infrastructure elements to assist in timely maintenance
                and repair planning.
              </Card.Text>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Home; 