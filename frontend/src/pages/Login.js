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
      <div className="login-box">
        <div className="login-card">
          <h1 className="login-title">Road AI Safety Enhancement</h1>
          {error && <Alert variant="danger" className="p-3 mb-3">{error}</Alert>}

                <Form onSubmit={handleSubmit} noValidate autoComplete="off">
                  <Form.Group className="mb-4">
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

                  <Form.Group className="mb-4">
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
                    style={{ minHeight: "48px" }}
                  >
                    {loading ? 'Signing in...' : 'Login'}
                  </Button>
                </Form>
              </div>
              <div className="login-footer">
                © 2025 Road AI Safety Enhancement. All rights reserved.
              </div>
            </div>
          </div>
  );
};

export default Login; 