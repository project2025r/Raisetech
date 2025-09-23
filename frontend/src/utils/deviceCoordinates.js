/**
 * Device Coordinate Integration Utility
 * Handles coordinate capture from any device or mobile phone
 */

/**
 * Get device coordinates using multiple methods with fallbacks
 * @param {Object} options - Configuration options
 * @returns {Promise<Object>} Coordinate data with accuracy and source info
 */
export const getDeviceCoordinates = (options = {}) => {
  const {
    enableHighAccuracy = true,
    timeout = 15000,
    maximumAge = 300000, // 5 minutes
    fallbackToIP = true
  } = options;

  return new Promise((resolve, reject) => {
    console.log('🌍 Starting device coordinate capture...');

    // Check if geolocation is supported
    if (!navigator.geolocation) {
      console.warn('❌ Geolocation not supported by this browser');
      if (fallbackToIP) {
        return getIPBasedLocation().then(resolve).catch(reject);
      }
      return reject(new Error('Geolocation not supported'));
    }

    const geoOptions = {
      enableHighAccuracy,
      timeout,
      maximumAge
    };

    console.log('📍 Requesting GPS coordinates with options:', geoOptions);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const coords = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          altitude: position.coords.altitude,
          altitudeAccuracy: position.coords.altitudeAccuracy,
          heading: position.coords.heading,
          speed: position.coords.speed,
          timestamp: position.timestamp,
          source: 'GPS',
          formatted: `${position.coords.latitude.toFixed(6)},${position.coords.longitude.toFixed(6)}`
        };

        console.log('✅ GPS coordinates obtained:', coords);
        resolve(coords);
      },
      (error) => {
        console.warn('❌ GPS coordinate error:', error.message);
        
        // Handle different error types
        let errorMessage = 'Failed to get location';
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage = 'Location access denied by user';
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage = 'Location information unavailable';
            break;
          case error.TIMEOUT:
            errorMessage = 'Location request timed out';
            break;
        }

        console.log(`🔄 GPS failed (${errorMessage}), trying fallback methods...`);

        // Try fallback methods
        if (fallbackToIP) {
          getIPBasedLocation()
            .then(resolve)
            .catch(() => reject(new Error(`GPS failed: ${errorMessage}, IP fallback also failed`)));
        } else {
          reject(new Error(errorMessage));
        }
      },
      geoOptions
    );
  });
};

/**
 * Get approximate location based on IP address
 * @returns {Promise<Object>} IP-based coordinate data
 */
const getIPBasedLocation = async () => {
  console.log('🌐 Attempting IP-based location...');
  
  try {
    // Try multiple IP geolocation services
    const services = [
      'https://ipapi.co/json/',
      'https://ip-api.com/json/',
      'https://ipinfo.io/json'
    ];

    for (const service of services) {
      try {
        console.log(`🔄 Trying IP service: ${service}`);
        const response = await fetch(service, { timeout: 5000 });
        const data = await response.json();

        let coords = null;

        // Handle different service response formats
        if (service.includes('ipapi.co')) {
          coords = {
            latitude: data.latitude,
            longitude: data.longitude,
            city: data.city,
            region: data.region,
            country: data.country_name
          };
        } else if (service.includes('ip-api.com')) {
          coords = {
            latitude: data.lat,
            longitude: data.lon,
            city: data.city,
            region: data.regionName,
            country: data.country
          };
        } else if (service.includes('ipinfo.io')) {
          const [lat, lng] = (data.loc || '0,0').split(',');
          coords = {
            latitude: parseFloat(lat),
            longitude: parseFloat(lng),
            city: data.city,
            region: data.region,
            country: data.country
          };
        }

        if (coords && coords.latitude && coords.longitude) {
          const result = {
            ...coords,
            accuracy: 10000, // IP-based location is less accurate
            source: 'IP',
            service: service,
            formatted: `${coords.latitude.toFixed(6)},${coords.longitude.toFixed(6)}`,
            timestamp: Date.now()
          };

          console.log('✅ IP-based coordinates obtained:', result);
          return result;
        }
      } catch (serviceError) {
        console.warn(`❌ IP service ${service} failed:`, serviceError.message);
        continue;
      }
    }

    throw new Error('All IP geolocation services failed');
  } catch (error) {
    console.error('❌ IP-based location failed:', error);
    throw error;
  }
};

/**
 * Watch device position for continuous updates
 * @param {Function} callback - Called with new position data
 * @param {Object} options - Configuration options
 * @returns {number} Watch ID for clearing the watch
 */
export const watchDeviceCoordinates = (callback, options = {}) => {
  const {
    enableHighAccuracy = true,
    timeout = 30000,
    maximumAge = 60000 // 1 minute
  } = options;

  if (!navigator.geolocation) {
    console.warn('❌ Geolocation not supported for watching');
    return null;
  }

  console.log('👀 Starting coordinate watching...');

  const watchId = navigator.geolocation.watchPosition(
    (position) => {
      const coords = {
        latitude: position.coords.latitude,
        longitude: position.coords.longitude,
        accuracy: position.coords.accuracy,
        timestamp: position.timestamp,
        source: 'GPS_WATCH',
        formatted: `${position.coords.latitude.toFixed(6)},${position.coords.longitude.toFixed(6)}`
      };

      console.log('📍 Position update:', coords);
      callback(coords);
    },
    (error) => {
      console.warn('❌ Position watch error:', error.message);
      callback({ error: error.message, source: 'GPS_WATCH_ERROR' });
    },
    {
      enableHighAccuracy,
      timeout,
      maximumAge
    }
  );

  return watchId;
};

/**
 * Clear position watching
 * @param {number} watchId - Watch ID returned by watchDeviceCoordinates
 */
export const clearCoordinateWatch = (watchId) => {
  if (watchId && navigator.geolocation) {
    navigator.geolocation.clearWatch(watchId);
    console.log('🛑 Coordinate watching cleared');
  }
};

/**
 * Get coordinate accuracy description
 * @param {number} accuracy - Accuracy in meters
 * @returns {string} Human-readable accuracy description
 */
export const getAccuracyDescription = (accuracy) => {
  if (!accuracy) return 'Unknown accuracy';
  
  if (accuracy <= 5) return 'Very High (±5m)';
  if (accuracy <= 20) return 'High (±20m)';
  if (accuracy <= 100) return 'Medium (±100m)';
  if (accuracy <= 1000) return 'Low (±1km)';
  return 'Very Low (>1km)';
};

/**
 * Validate coordinates
 * @param {Object} coords - Coordinate object
 * @returns {boolean} True if coordinates are valid
 */
export const validateCoordinates = (coords) => {
  if (!coords || typeof coords !== 'object') return false;
  
  const { latitude, longitude } = coords;
  
  return (
    typeof latitude === 'number' &&
    typeof longitude === 'number' &&
    latitude >= -90 && latitude <= 90 &&
    longitude >= -180 && longitude <= 180 &&
    !isNaN(latitude) && !isNaN(longitude)
  );
};

/**
 * Format coordinates for display
 * @param {Object} coords - Coordinate object
 * @param {number} precision - Decimal places (default: 6)
 * @returns {string} Formatted coordinate string
 */
export const formatCoordinates = (coords, precision = 6) => {
  if (!validateCoordinates(coords)) return 'Invalid coordinates';
  
  return `${coords.latitude.toFixed(precision)}, ${coords.longitude.toFixed(precision)}`;
};

/**
 * Calculate distance between two coordinate points
 * @param {Object} coord1 - First coordinate
 * @param {Object} coord2 - Second coordinate
 * @returns {number} Distance in meters
 */
export const calculateDistance = (coord1, coord2) => {
  if (!validateCoordinates(coord1) || !validateCoordinates(coord2)) {
    return null;
  }

  const R = 6371e3; // Earth's radius in meters
  const φ1 = coord1.latitude * Math.PI / 180;
  const φ2 = coord2.latitude * Math.PI / 180;
  const Δφ = (coord2.latitude - coord1.latitude) * Math.PI / 180;
  const Δλ = (coord2.longitude - coord1.longitude) * Math.PI / 180;

  const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
            Math.cos(φ1) * Math.cos(φ2) *
            Math.sin(Δλ/2) * Math.sin(Δλ/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

  return R * c; // Distance in meters
};
