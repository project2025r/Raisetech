import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

// Import components
import Login from './pages/Login';
import Home from './pages/Home';
import Pavement from './pages/Pavement';
import RoadInfrastructure from './pages/RoadInfrastructure';
import Recommendation from './pages/Recommendation';
import Dashboard from './pages/Dashboard';
import DefectDetail from './pages/DefectDetail';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

function App() {
  const [authenticated, setAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);

  // Check if user is already authenticated (stored in session storage)
  useEffect(() => {
    const storedUser = sessionStorage.getItem('user');
    if (storedUser) {
      setCurrentUser(JSON.parse(storedUser));
      setAuthenticated(true);
    }
  }, []);

  // Handle login
  const handleLogin = (user) => {
    setCurrentUser(user);
    setAuthenticated(true);
    sessionStorage.setItem('user', JSON.stringify(user));
  };

  // Handle logout
  const handleLogout = () => {
    setCurrentUser(null);
    setAuthenticated(false);
    sessionStorage.removeItem('user');
  };

  return (
    <Router>
      <div className="app-container">
        {authenticated && <Sidebar onLogout={handleLogout} />}
        <div className="content-container">
          {authenticated && <Header user={currentUser} />}
          <Routes>
            <Route 
              path="/login" 
              element={authenticated ? <Navigate to="/" /> : <Login onLogin={handleLogin} />} 
            />
            <Route 
              path="/" 
              element={authenticated ? <Home user={currentUser} /> : <Navigate to="/login" />} 
            />
            <Route 
              path="/pavement" 
              element={authenticated ? <Pavement /> : <Navigate to="/login" />} 
            />
            <Route 
              path="/road-infrastructure" 
              element={authenticated ? <RoadInfrastructure /> : <Navigate to="/login" />} 
            />
            <Route 
              path="/recommendation" 
              element={authenticated ? <Recommendation /> : <Navigate to="/login" />} 
            />
            <Route 
              path="/dashboard" 
              element={authenticated ? <Dashboard /> : <Navigate to="/login" />} 
            />
            <Route 
              path="/view/:imageId" 
              element={authenticated ? <DefectDetail /> : <Navigate to="/login" />} 
            />
            <Route 
              path="*" 
              element={<Navigate to={authenticated ? "/" : "/login"} />} 
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App; 