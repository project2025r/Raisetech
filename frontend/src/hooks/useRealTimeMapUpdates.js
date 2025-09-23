import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';

/**
 * Custom hook for real-time map updates
 * Ensures newly uploaded images with coordinates are immediately reflected on the map
 */
export const useRealTimeMapUpdates = (initialFilters = {}) => {
  const [defects, setDefects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const intervalRef = useRef(null);
  const lastFetchRef = useRef(0);

  // Fetch defect data with enhanced caching and real-time updates
  const fetchDefectData = useCallback(async (forceRefresh = false) => {
    try {
      setLoading(true);
      setError(null);

      // Prevent too frequent requests (minimum 2 seconds between requests)
      const now = Date.now();
      if (!forceRefresh && (now - lastFetchRef.current) < 2000) {
        console.log('🚫 Skipping fetch - too frequent');
        setLoading(false);
        return defects;
      }
      lastFetchRef.current = now;

      // Prepare query parameters with enhanced cache-busting
      const params = {
        ...initialFilters,
        _t: now,
        _refresh: forceRefresh ? Math.random().toString(36).substring(7) : undefined
      };

      console.log('🔄 Fetching real-time defect data...', params);

      const response = await axios.get('/api/dashboard/image-stats', {
        params,
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        },
        timeout: 15000
      });

      if (response.data.success) {
        const images = response.data.images || [];
        console.log(`✅ Fetched ${images.length} images`);

        // Process images for map display with enhanced coordinate handling
        const processedDefects = [];

        for (const image of images) {
          let lat, lng;

          // Enhanced coordinate priority system
          // 1. EXIF GPS coordinates (most accurate)
          if (image.exif_data?.gps_coordinates) {
            lat = image.exif_data.gps_coordinates.latitude;
            lng = image.exif_data.gps_coordinates.longitude;
            console.log(`🎯 Using EXIF GPS for ${image.image_id}: [${lat}, ${lng}]`);
          }
          // 2. Device GPS coordinates
          else if (image.device_coordinates?.source === 'GPS') {
            lat = image.device_coordinates.latitude;
            lng = image.device_coordinates.longitude;
            console.log(`📱 Using device GPS for ${image.image_id}: [${lat}, ${lng}]`);
          }
          // 3. Device IP coordinates
          else if (image.device_coordinates?.source === 'IP') {
            lat = image.device_coordinates.latitude;
            lng = image.device_coordinates.longitude;
            console.log(`🌐 Using device IP for ${image.image_id}: [${lat}, ${lng}]`);
          }
          // 4. Stored coordinates (fallback)
          else if (image.coordinates) {
            if (typeof image.coordinates === 'string') {
              const coords = image.coordinates.split(',');
              if (coords.length === 2) {
                lat = parseFloat(coords[0].trim());
                lng = parseFloat(coords[1].trim());
              }
            } else if (typeof image.coordinates === 'object') {
              lat = image.coordinates.latitude;
              lng = image.coordinates.longitude;
            }
            console.log(`📍 Using stored coordinates for ${image.image_id}: [${lat}, ${lng}]`);
          }

          // Validate coordinates
          if (!isNaN(lat) && !isNaN(lng) &&
              lat >= -90 && lat <= 90 &&
              lng >= -180 && lng <= 180) {
            
            processedDefects.push({
              id: image.id,
              image_id: image.image_id,
              type: image.type,
              position: [lat, lng],
              defect_count: image.defect_count,
              timestamp: new Date(image.timestamp).toLocaleString(),
              username: image.username,
              original_image_id: image.original_image_id,
              original_image_s3_url: image.original_image_s3_url,
              original_image_full_url: image.original_image_full_url,
              // Enhanced metadata
              exif_data: image.exif_data || {},
              metadata: image.metadata || {},
              media_type: image.media_type || 'image',
              coordinate_source: image.coordinate_source || 'unknown',
              device_coordinates: image.device_coordinates || {},
              // Type-specific data
              type_counts: image.type_counts,
              condition_counts: image.condition_counts,
              representative_frame: image.representative_frame,
              video_id: image.video_id
            });
          } else {
            console.warn(`❌ Invalid coordinates for image ${image.image_id}: [${lat}, ${lng}]`);
          }
        }

        // Sort by timestamp (newest first) to prioritize recent uploads
        processedDefects.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        setDefects(processedDefects);
        setLastUpdate(new Date());
        
        console.log(`📊 Processed ${processedDefects.length} valid defects for map display`);
        
        // Log recent uploads (last 5 minutes)
        const recentThreshold = Date.now() - (5 * 60 * 1000);
        const recentDefects = processedDefects.filter(d => 
          new Date(d.timestamp).getTime() > recentThreshold
        );
        
        if (recentDefects.length > 0) {
          console.log(`🆕 Found ${recentDefects.length} recent defects (last 5 minutes)`);
        }

        return processedDefects;
      } else {
        throw new Error(response.data.message || 'Failed to fetch defect data');
      }
    } catch (err) {
      console.error('❌ Error fetching defect data:', err);
      setError('Failed to load defect data: ' + (err.response?.data?.message || err.message));
      return defects; // Return existing data on error
    } finally {
      setLoading(false);
    }
  }, [initialFilters, defects]);

  // Start real-time updates
  const startRealTimeUpdates = useCallback((interval = 30000) => {
    console.log(`🔄 Starting real-time updates (${interval}ms interval)`);
    
    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Set up new interval
    intervalRef.current = setInterval(() => {
      console.log('⏰ Auto-refresh triggered');
      fetchDefectData(true);
    }, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [fetchDefectData]);

  // Stop real-time updates
  const stopRealTimeUpdates = useCallback(() => {
    console.log('🛑 Stopping real-time updates');
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Force immediate refresh
  const forceRefresh = useCallback(() => {
    console.log('🔄 Force refresh triggered');
    return fetchDefectData(true);
  }, [fetchDefectData]);

  // Handle new upload notification
  const handleNewUpload = useCallback((uploadData) => {
    console.log('🆕 New upload detected, triggering immediate refresh');
    
    // Add the new upload to the list immediately for instant feedback
    if (uploadData && uploadData.coordinates) {
      const tempDefect = {
        id: `temp-${Date.now()}`,
        image_id: uploadData.image_id || `temp-${Date.now()}`,
        type: uploadData.type || 'unknown',
        position: uploadData.coordinates.split(',').map(c => parseFloat(c.trim())),
        defect_count: 1,
        timestamp: new Date().toLocaleString(),
        username: uploadData.username || 'Current User',
        coordinate_source: uploadData.coordinate_source || 'device',
        isTemporary: true
      };

      setDefects(prev => [tempDefect, ...prev]);
    }

    // Trigger refresh after a short delay to get the actual data
    setTimeout(() => {
      forceRefresh();
    }, 2000);
  }, [forceRefresh]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return {
    defects,
    loading,
    error,
    lastUpdate,
    fetchDefectData,
    startRealTimeUpdates,
    stopRealTimeUpdates,
    forceRefresh,
    handleNewUpload
  };
};
