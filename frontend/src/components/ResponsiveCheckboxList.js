import React from 'react';
import PropTypes from 'prop-types';

/**
 * A responsive checkbox list component that displays options in a scrollable container
 * with proper mobile styling for touch interactions.
 * 
 * @param {Object} props Component props
 * @param {Array} props.options Array of option objects or strings
 * @param {Array} props.selectedValues Array of currently selected values
 * @param {Function} props.onChange Function called when an option is selected/unselected
 * @param {String} props.idPrefix Prefix for checkbox IDs
 * @param {String} props.className Additional CSS classes
 * @param {String} props.optionLabelKey Key for option label when using objects
 * @param {String} props.optionValueKey Key for option value when using objects
 * @param {Boolean} props.isLoading Whether the options are loading
 * @param {String} props.keyField Key field for option objects
 * @param {String} props.labelField Label field for option objects
 * @param {String} props.containerClassName Additional CSS classes for the container
 * @returns {JSX.Element}
 */
const ResponsiveCheckboxList = ({
  options,
  selectedValues,
  onChange,
  isLoading = false,
  keyField = 'value',
  labelField = 'label',
  containerClassName = '',
  idPrefix = '',
  className = '',
}) => {
  // Function to handle checkbox changes
  const handleCheckboxChange = (e) => {
    const value = e.target.value;
    onChange(e);
  };

  // Helper function to clean values for ID generation
  const cleanValueForId = (value) => {
    if (value === undefined || value === null) return 'undefined';
    return String(value)
      .replace(/[^a-zA-Z0-9-_]/g, '_')
      .toLowerCase();
  };

  // If options are loading, show a loading state
  if (isLoading) {
    return (
      <div className={`checkbox-container ${containerClassName}`}>
        <div className="text-center py-3">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      </div>
    );
  }

  // If no options, show an empty state
  if (!options || options.length === 0) {
    return (
      <div className={`checkbox-container ${containerClassName}`}>
        <div className="text-center py-3">No options available</div>
      </div>
    );
  }

  return (
    <div className={`checkbox-container ${containerClassName}`}>
      {options.map((option, index) => {
        const value = typeof option === 'object' ? option[keyField] : option;
        const label = typeof option === 'object' ? option[labelField] : option;
        const cleanValue = cleanValueForId(value);
        const id = `${idPrefix}-${cleanValue}-${index}`;
        const isChecked = selectedValues.includes(value);

        return (
          <div 
            key={id} 
            className={`form-check ${className}`}
          >
            <input
              id={id}
              type="checkbox"
              className="form-check-input"
              checked={isChecked}
              value={value}
              onChange={handleCheckboxChange}
            />
            <label 
              className="form-check-label" 
              htmlFor={id}
              title={label}
            >
              {label}
            </label>
          </div>
        );
      })}
    </div>
  );
};

ResponsiveCheckboxList.propTypes = {
  options: PropTypes.oneOfType([
    PropTypes.arrayOf(PropTypes.string),
    PropTypes.arrayOf(PropTypes.shape({
      [PropTypes.string]: PropTypes.any,
    })),
  ]).isRequired,
  selectedValues: PropTypes.array.isRequired,
  onChange: PropTypes.func.isRequired,
  isLoading: PropTypes.bool,
  keyField: PropTypes.string,
  labelField: PropTypes.string,
  containerClassName: PropTypes.string,
  idPrefix: PropTypes.string,
  className: PropTypes.string,
};

export default ResponsiveCheckboxList; 