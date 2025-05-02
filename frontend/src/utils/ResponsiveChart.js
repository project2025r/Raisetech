import React, { useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';

/**
 * A wrapper component for Plotly charts that ensures they are responsive
 * on all devices including mobile.
 * 
 * @param {Object} props - Component props
 * @param {Array} props.data - Plotly data array
 * @param {Object} props.layout - Plotly layout object
 * @param {Object} props.config - Plotly config object
 * @param {String} props.className - Additional CSS class names
 * @param {Number} props.minHeight - Minimum height for the chart (default: 250px)
 * @param {Object} props.style - Additional inline styles
 * @returns {JSX.Element} Responsive Plotly chart
 */
const ResponsiveChart = ({ 
  data, 
  layout = {}, 
  config = {}, 
  className = '', 
  minHeight = 250,
  style = {},
  ...rest 
}) => {
  const containerRef = useRef(null);

  // Default configuration for better mobile display
  const defaultConfig = {
    responsive: true,
    displayModeBar: false, // Hide the modebar on mobile to save space
    ...config
  };

  // Enhance layout with better defaults for mobile
  const enhancedLayout = {
    autosize: true,
    font: {
      family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
      size: 12
    },
    margin: {
      l: 50,
      r: 25,
      t: 25,
      b: 50,
      pad: 4
    },
    ...layout
  };

  // Adjust the chart size on window resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        // Force Plotly to recalculate dimensions
        window.dispatchEvent(new Event('resize'));
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div 
      ref={containerRef} 
      className={`responsive-chart-container ${className}`} 
      style={{ 
        width: '100%', 
        minHeight: `${minHeight}px`,
        ...style
      }}
    >
      <Plot
        data={data}
        layout={enhancedLayout}
        config={defaultConfig}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
        {...rest}
      />
    </div>
  );
};

export default ResponsiveChart; 