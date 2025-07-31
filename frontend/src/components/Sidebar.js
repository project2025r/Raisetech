import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { 
  FaHome, 
  FaMap, 
  FaColumns, 
  FaLightbulb, 
  FaChartBar, 
  FaSignOutAlt,
  FaBars,
  FaTimes
} from 'react-icons/fa';
import axios from 'axios';
import './Sidebar.css';

const Sidebar = ({ onLogout }) => {
  const navigate = useNavigate();
  const [activePage, setActivePage] = useState(window.location.pathname);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [isOpen, setIsOpen] = useState(!isMobile);

  // Handle window resize events
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth <= 768;
      setIsMobile(mobile);
      if (!mobile) {
        setIsOpen(true);
      } else {
        setIsOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleLogout = async () => {
    try {
      // Call logout API
      await axios.post('/api/auth/logout');
      // Call the onLogout prop function to update app state
      onLogout();
      // Redirect to login page
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
      // Even if there's an error, still log out on the client side
      onLogout();
      navigate('/login');
    }
  };

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  const handleNavClick = () => {
    if (isMobile) {
      setIsOpen(false);
    }
  };

  const menuItems = [
    { path: '/', name: 'Home', icon: <FaHome size={20} /> },
    { path: '/pavement', name: 'Pavement', icon: <FaMap size={20} /> },
    { path: '/road-infrastructure', name: 'Infrastructure', icon: <FaColumns size={20} /> },
    { path: '/recommendation', name: 'Recommendation', icon: <FaLightbulb size={20} /> },
    { path: '/dashboard', name: 'Dashboard', icon: <FaChartBar size={20} /> }
  ];

  return (
    <>
      {/* Mobile sidebar toggle button */}
      {isMobile && (
        <button 
          className="sidebar-toggle" 
          onClick={toggleSidebar}
          aria-label={isOpen ? "Close menu" : "Open menu"}
        >
          {isOpen ? <FaTimes size={24} /> : <FaBars size={24} />}
        </button>
      )}

      {/* Sidebar overlay backdrop for mobile */}
      {isMobile && isOpen && (
        <div className="sidebar-backdrop" onClick={toggleSidebar} />
      )}

      {/* Main sidebar */}
      <div className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h3></h3>
          {isMobile && (
            <button 
              className="sidebar-close" 
              onClick={toggleSidebar}
              aria-label="Close menu"
            >
              <FaTimes size={20} />
            </button>
          )}
        </div>
        <div className="sidebar-menu">
          {menuItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => 
                isActive ? 'sidebar-item active' : 'sidebar-item'
              }
              onClick={() => {
                setActivePage(item.path);
                handleNavClick();
              }}
            >
              <div className="sidebar-icon">{item.icon}</div>
              <div className="sidebar-text">{item.name}</div>
            </NavLink>
          ))}
        </div>
        <div className="sidebar-footer">
          <button className="sidebar-logout" onClick={handleLogout}>
            <div className="sidebar-icon"><FaSignOutAlt size={20} /></div>
            <div className="sidebar-text">Logout</div>
          </button>
          <div className="sidebar-watermark">Powered by AiSPRY</div>
        </div>
      </div>

      {/* Mobile bottom navigation for quick access */}
      {isMobile && (
        <div className="mobile-nav">
          {menuItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => 
                isActive ? 'mobile-nav-item active' : 'mobile-nav-item'
              }
              onClick={() => setActivePage(item.path)}
            >
              <div className="mobile-nav-icon">{item.icon}</div>
              <div className="mobile-nav-text">{item.name}</div>
            </NavLink>
          ))}
        </div>
      )}
    </>
  );
};

export default Sidebar; 