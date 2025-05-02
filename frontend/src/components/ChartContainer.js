import React from 'react';
import PropTypes from 'prop-types';
import Plot from 'react-plotly.js';

/**
 * A responsive container for charts that ensures consistent styling and behavior
 * 
 * @param {Object} props Component props
 * @param {Array} props.data Plotly data array
 * @param {Object} props.layout Plotly layout object
 * @param {Object} props.config Plotly config object
 * @param {String} props.className Additional CSS class names
 * @param {Object} props.style Additional inline styles
 * @param {Boolean} props.showLegend Whether to show the legend or not
 * @param {Array} props.legendItems Array of legend items objects with label, color, checked, onChange props
 * @returns {JSX.Element}
 */
const ChartContainer = ({
  data,
  layout = {},
  config = {},
  className = '',
  style = {},
  showLegend = false,
  legendItems = [],
  isPieChart = false,
}) => {
  // Default configuration
  const defaultConfig = {
    responsive: true,
    displayModeBar: false,
    ...config
  };

  // Set appropriate height based on chart type and screen size
  const getDefaultHeight = () => {
    if (window.innerWidth <= 768) {
      return isPieChart ? 300 : 250;
    }
    return isPieChart ? 350 : 300;
  };

  // Enhanced layout with better defaults
  const enhancedLayout = {
    autosize: true,
    font: {
      family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      size: 12
    },
    margin: isPieChart 
      ? { t: 20, b: 20, l: 20, r: 20 } 
      : { t: 10, b: 80, l: 60, r: 10 },
    height: layout.height || getDefaultHeight(),
    ...layout
  };

  return (
    <div className={`chart-wrapper ${className}`}>
      {showLegend && legendItems.length > 0 && (
        <div className="legend-container">
          {legendItems.map((item, index) => (
            <div className="legend-item" key={index}>
              <input
                className="form-check-input legend-checkbox"
                type="checkbox"
                id={`legend-item-${index}`}
                checked={item.checked}
                onChange={item.onChange}
              />
              <label className="legend-label" htmlFor={`legend-item-${index}`}>
                <span 
                  className="legend-color" 
                  style={{ backgroundColor: item.color }}
                ></span>
                {item.label}
              </label>
            </div>
          ))}
        </div>
      )}
      <div className={`chart-container ${isPieChart ? 'pie-chart-container' : ''}`}>
        <Plot
          data={data}
          layout={enhancedLayout}
          config={defaultConfig}
          style={{ width: '100%', height: '100%', ...style }}
          useResizeHandler={true}
        />
      </div>
    </div>
  );
};

ChartContainer.propTypes = {
  data: PropTypes.array.isRequired,
  layout: PropTypes.object,
  config: PropTypes.object,
  className: PropTypes.string,
  style: PropTypes.object,
  showLegend: PropTypes.bool,
  legendItems: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.string.isRequired,
      color: PropTypes.string.isRequired,
      checked: PropTypes.bool.isRequired,
      onChange: PropTypes.func.isRequired
    })
  ),
  isPieChart: PropTypes.bool
};

export default ChartContainer; 