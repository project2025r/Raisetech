import React, { useState } from 'react';
import axios from 'axios';
import { Container, Row, Col, Form, Button, Alert } from 'react-bootstrap';
import './Login.css';

const Login = ({ onLogin }) => {
  const [role, setRole] = useState('Supervisor');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate form
    if (!username || !password) {
      setError('Please enter both username and password');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post('/api/auth/login', {
        role,
        username,
        password
      });

      if (response.data.success) {
        onLogin(response.data.user);
      } else {
        setError(response.data.message || 'Login failed');
      }
    } catch (error) {
      setError(
        error.response?.data?.message || 
        'Login failed. Please check your credentials and try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <Container>
        <Row className="justify-content-center">
          <Col xs={12} sm={10} md={8} lg={5} xl={4}>
            <div className="login-box">
              <div className="login-logo text-center mb-4">
                <h2 className="text-primary">Road AI Safety Enhancement</h2>
              </div>

              <div className="login-card">
                <h4 className="text-center mb-4">Sign In</h4>

                {error && <Alert variant="danger">{error}</Alert>}

                <Form onSubmit={handleSubmit} noValidate autoComplete="off">
                  <Form.Group className="mb-3">
                    <Form.Label>Role</Form.Label>
                    <Form.Select 
                      value={role}
                      onChange={(e) => setRole(e.target.value)}
                      aria-label="Select your role"
                    >
                      <option value="Supervisor">Supervisor</option>
                      <option value="Field Officer">Field Officer</option>
                      <option value="Admin">Admin</option>
                    </Form.Select>
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label>Username</Form.Label>
                    <Form.Control 
                      type="text" 
                      placeholder="Enter your username"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      autoCapitalize="none"
                      aria-label="Username"
                    />
                  </Form.Group>

                  <Form.Group className="mb-4">
                    <Form.Label>Password</Form.Label>
                    <Form.Control 
                      type="password" 
                      placeholder="Enter your password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      aria-label="Password"
                    />
                  </Form.Group>

                  <Button 
                    variant="primary" 
                    type="submit" 
                    className="w-100"
                    disabled={loading}
                    style={{ minHeight: "44px" }}
                  >
                    {loading ? 'Signing in...' : 'Login'}
                  </Button>
                </Form>
              </div>

              <div className="text-center mt-4 text-muted login-footer">
                © 2025 Road AI Safety Enhancement. All rights reserved.
              </div>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default Login; 