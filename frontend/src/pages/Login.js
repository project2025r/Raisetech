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
      {/* Left Side - Animated Icons and Logo */}
      <div className="left-section">
        <div className="icon-circle icon-1">
          <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
        </div>
        <div className="icon-circle icon-2">
          <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
          </svg>
        </div>
        <div className="icon-circle icon-3">
          <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
            <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
          </svg>
        </div>
        <div className="icon-circle icon-4">
          <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
          </svg>
        </div>
        <div className="icon-circle icon-5">
          <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
            <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/>
          </svg>
        </div>
        <div className="icon-circle icon-6">
          <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
          </svg>
        </div>
        <div className="icon-circle icon-7">
          <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
            <path d="M9 11H7v2h2v-2zm4 0h-2v2h2v-2zm4 0h-2v2h2v-2zm2-7h-1V2h-2v2H8V2H6v2H5c-1.11 0-1.99.9-1.99 2L3 20c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 16H5V9h14v11z"/>
          </svg>
        </div>
        <div className="icon-circle icon-8">
          <svg viewBox="0 0 24 24" fill="white" width="32" height="32">
            <path d="M20 6h-2.18c.11-.31.18-.65.18-1 0-1.66-1.34-3-3-3-1.05 0-1.96.54-2.5 1.35l-.5.67-.5-.68C10.96 2.54 10.05 2 9 2 7.34 2 6 3.34 6 5c0 .35.07.69.18 1H4c-1.11 0-1.99.89-1.99 2L2 19c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2zm-5-2c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zM9 4c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm11 15H4v-2h16v2zm0-5H4V8h5.08L7 10.83 8.62 12 11 8.76l1-1.36 1 1.36L15.38 12 17 10.83 14.92 8H20v6z"/>
          </svg>
        </div>

        <div className="logo-container">
          <div className="logo-circle">
            <div className="logo-text">
              <span className="logo-raisse">RAISE</span>
              <span className="logo-tech">TECH</span>
              <span className="logo-ai">AI</span>
            </div>
          </div>
        </div>

        {/* Background decorative circles */}
        <div className="bg-circle bg-circle-1"></div>
        <div className="bg-circle bg-circle-2"></div>
      </div>

      {/* Right Side - Login Form */}
      <div className="right-section">
        <div className="login-box">
          <h1 className="login-title">Login</h1>
          <h2 className="login-subtitle">Road Ai Safety Enhancement</h2>

          {error && <Alert variant="danger" className="mb-3">{error}</Alert>}

          {/* Role Selection Tabs */}
          <div className="role-tabs">
            <button
              type="button"
              className={`role-tab ${role === 'Field Officer' ? 'active' : ''}`}
              onClick={() => setRole('Field Officer')}
            >
              Field Workers
            </button>
            <button
              type="button"
              className={`role-tab ${role === 'Admin' ? 'active' : ''}`}
              onClick={() => setRole('Admin')}
            >
              Admin
            </button>
            <button
              type="button"
              className={`role-tab ${role === 'Supervisor' ? 'active' : ''}`}
              onClick={() => setRole('Supervisor')}
            >
              Supervisor
            </button>
          </div>

          <Form onSubmit={handleSubmit} noValidate autoComplete="off">
            <Form.Group className="mb-4">
              <Form.Label>User Name</Form.Label>
              <Form.Control
                type="text"
                placeholder="username"
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
                placeholder="***"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                aria-label="Password"
              />
            </Form.Group>

            <Button
              // variant="primary"
              type="submit"
              className="w-100 login-button"
              disabled={loading}
            >
              {loading ? 'Signing in...' : 'Login'}
            </Button>
            <div className="login-footer-btn">
                © 2025 Road AI Safety Enhancement. All rights reserved.
              </div>
          </Form>
        </div>
      </div>
    </div>
  );
};

export default Login; 