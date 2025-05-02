import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';

const Header = ({ user }) => {
  return (
    <div className="full-bar">
      <Container fluid>
        <Row className="align-items-center">
          <Col className="text-center">
            <span className="header-title">RAISE</span>
          </Col>
          <Col xs="auto" className="d-none d-md-block">
            <small className="header-user-info">
              {user?.role}: {user?.username}
            </small>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default Header; 