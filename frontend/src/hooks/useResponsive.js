import { useState, useEffect } from 'react';

/**
 * Custom hook to detect screen size and provide responsive breakpoints
 * @returns {Object} Object containing responsive breakpoint booleans and current window dimensions
 */
const useResponsive = () => {
  const [windowDimensions, setWindowDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });
  
  useEffect(() => {
    const handleResize = () => {
      setWindowDimensions({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  const { width, height } = windowDimensions;
  
  return {
    isMobile: width <= 768,
    isSmallMobile: width <= 480,
    isTablet: width > 768 && width <= 992,
    isDesktop: width > 992,
    isLargeDesktop: width > 1200,
    width,
    height
  };
};

export default useResponsive; 