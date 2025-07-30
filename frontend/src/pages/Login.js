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

    // Set up retry logic for initial connection issues
    const maxRetries = 2;
    let retryCount = 0;
    let loginSuccessful = false;
    
    while (retryCount <= maxRetries && !loginSuccessful) {
      try {
        const response = await axios.post('/api/auth/login', {
          role,
          username,
          password
        });

        if (response.data.success) {
          onLogin(response.data.user);
          loginSuccessful = true;
          break;
        } else {
          setError(response.data.message || 'Login failed');
          break; // Don't retry if we got a response but authentication failed
        }
      } catch (error) {
        // Only retry if it's likely a connection error (no response)
        if (error.request && !error.response && retryCount < maxRetries) {
          retryCount++;
          setError(`Connection attempt ${retryCount} failed, retrying...`);
          // Wait before retrying
          await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
        } else {
          setError(
            error.response?.data?.message || 
            'Login failed. Please check your credentials and try again.'
          );
          break;
        }
      }
    }
    
    setLoading(false);
  };

  return (
    <div className="login-page">
      <Container>
        <Row className="justify-content-center">
          <Col xs={12} sm={10} md={8} lg={5} xl={4}>
            <div className="login-box">
              <div className="login-card">
                {error && <Alert variant="danger" className="p-3 mb-3">{error}</Alert>}

                <Form onSubmit={handleSubmit} noValidate autoComplete="off">
                  <Form.Group className="mb-3">
                    <Form.Label className="mb-1">Role</Form.Label>
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
                    <Form.Label className="mb-1">Username</Form.Label>
                    <Form.Control 
                      type="text" 
                      placeholder="Enter your username"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      autoCapitalize="none"
                      aria-label="Username"
                    />
                  </Form.Group>

                  <Form.Group className="mb-3">
                    <Form.Label className="mb-1">Password</Form.Label>
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

              <div className="text-center mt-3 text-muted login-footer">
                <small>© 2025 Road AI Safety Enhancement. All rights reserved.</small>
              </div>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default Login; 