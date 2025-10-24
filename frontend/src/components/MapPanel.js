import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Card, Button, Form, Alert, Spinner } from 'react-bootstrap';
import { MapContainer, TileLayer, Marker, Polyline, Tooltip, useMap } from 'react-leaflet';
import L from 'leaflet';

// Helper component to invalidate map size after mount and on tile load
function MapSizeFix({ mapRef }) {
  const map = useMap();

  useEffect(() => {
    // Store map reference for parent component
    if (mapRef) {
      mapRef.current = map;
    }

    // Invalidate size shortly after mount with better error handling
    const t1 = setTimeout(() => {
      try {
        if (map && map.invalidateSize) {
          map.invalidateSize(true);
        }
      } catch (error) {
        console.warn('Map invalidateSize error (t1):', error);
      }
    }, 100);

    const t2 = setTimeout(() => {
      try {
        if (map && map.invalidateSize) {
          map.invalidateSize(true);
        }
      } catch (error) {
        console.warn('Map invalidateSize error (t2):', error);
      }
    }, 500);

    // Also on window resize with error handling
    const onResize = () => {
      try {
        if (map && map.invalidateSize) {
          map.invalidateSize(true);
        }
      } catch (error) {
        console.warn('Map resize error:', error);
      }
    };

    window.addEventListener('resize', onResize);

    // Invalidate after zoom animations end with error handling
    const onZoomEnd = () => {
      try {
        if (map && map.invalidateSize) {
          map.invalidateSize(true);
        }
      } catch (error) {
        console.warn('Map zoom end error:', error);
      }
    };

    const onMoveEnd = () => {
      try {
        if (map && map.invalidateSize) {
          map.invalidateSize(true);
        }
      } catch (error) {
        console.warn('Map move end error:', error);
      }
    };

    if (map) {
      map.on('zoomend', onZoomEnd);
      map.on('moveend', onMoveEnd);
    }

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      window.removeEventListener('resize', onResize);
      if (map) {
        try {
          map.off('zoomend', onZoomEnd);
          map.off('moveend', onMoveEnd);
        } catch (error) {
          console.warn('Map cleanup error:', error);
        }
      }
    };
  }, [map, mapRef]);

  return null;
}


const defaultCenter = [1.3521, 103.8198];
const colors = ['#1e88e5', '#43a047', '#f4511e'];

const toLatLng = (coords) => coords.map(([lon, lat]) => [lat, lon]);

function lngLatToMeters(lat, lon) {
	const originShift = 2 * Math.PI * 6378137 / 2.0;
	const mx = lon * originShift / 180.0;
	let my = Math.log(Math.tan((90 + lat) * Math.PI / 360.0)) / (Math.PI / 180.0);
	my = my * originShift / 180.0;
	return { x: mx, y: my };
}

function pointToSegmentDistanceMeters(p, a, b) {
	const P = lngLatToMeters(p[0], p[1]);
	const A = lngLatToMeters(a[0], a[1]); 
	const B = lngLatToMeters(b[0], b[1]);
	const vx = B.x - A.x;
	const vy = B.y - A.y;
	const wx = P.x - A.x;
	const wy = P.y - A.y;
	const c1 = vx * wx + vy * wy;
	if (c1 <= 0) return Math.hypot(P.x - A.x, P.y - A.y);
	const c2 = vx * vx + vy * vy;
	if (c2 <= c1) return Math.hypot(P.x - B.x, P.y - B.y);
	const t = c1 / c2;
	const projx = A.x + t * vx;
	const projy = A.y + t * vy;
	return Math.hypot(P.x - projx, P.y - projy);
}

const MapPanel = ({
	panelTitle = 'Map',
	fromLocation,
	toLocation,
	onFromChange,
	onToChange,
	onRoutesLoaded,
	initialZoom = 12,
	gpsTrack,
	onOutOfRange,
}) => {
	const [fromLat, setFromLat] = useState(fromLocation?.lat || '');
	const [fromLon, setFromLon] = useState(fromLocation?.lon || '');
	const [toLat, setToLat] = useState(toLocation?.lat || '');
	const [toLon, setToLon] = useState(toLocation?.lon || '');
	const [routes, setRoutes] = useState([]);
	const [activeIdx, setActiveIdx] = useState(0);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState('');
	const clickToggleRef = useRef('from');
	const leafletMapRef = useRef(null);
	const lastFetchedCoordsRef = useRef(null);

	useEffect(() => {
		setFromLat(fromLocation?.lat ?? '');
		setFromLon(fromLocation?.lon ?? '');
	}, [fromLocation]);

	useEffect(() => {
		setToLat(toLocation?.lat ?? '');
		setToLon(toLocation?.lon ?? '');
	}, [toLocation]);

	const hasBoth = useMemo(() => (
		fromLat !== '' && fromLon !== '' && toLat !== '' && toLon !== ''
	), [fromLat, fromLon, toLat, toLon]);

	const coordsSignature = useMemo(() => {
		if (!hasBoth) return null;
		return `${Number(fromLat).toFixed(6)}:${Number(fromLon).toFixed(6)}|${Number(toLat).toFixed(6)}:${Number(toLon).toFixed(6)}`;
	}, [hasBoth, fromLat, fromLon, toLat, toLon]);

	const handleFetchRoutes = async ({ suppressErrors = false } = {}) => {
		if (!hasBoth) {
			if (!suppressErrors) {
				setError('Please set both From and To coordinates.');
			}
			return;
		}

		setError('');
		setLoading(true);
		try {
			lastFetchedCoordsRef.current = coordsSignature || null;
			const url = `https://router.project-osrm.org/route/v1/driving/${Number(fromLon)},${Number(fromLat)};${Number(toLon)},${Number(toLat)}?alternatives=true&overview=full&geometries=geojson&steps=true`;
			const res = await fetch(url);
			if (!res.ok) throw new Error(`OSRM request failed: ${res.status}`);
			const data = await res.json();
			if (data.code !== 'Ok') throw new Error(data.message || 'OSRM error');
			const limited = (data.routes || []).slice(0, 3).map((r, idx) => ({
				index: idx,
				polyline: toLatLng(r.geometry.coordinates),
				distance: r.distance,
				duration: r.duration,
				legs: r.legs,
			}));
			setRoutes(limited);
			setActiveIdx(0);
			onRoutesLoaded && onRoutesLoaded(limited);
			fitToRoute(limited[0]);
			if (gpsTrack && gpsTrack.length > 0 && limited.length > 0) {
				const activePath = limited[0].polyline;
				const halfWidthM = 50;
				const out = gpsTrack.some(([lat, lon]) => {
					let minD = Infinity;
					for (let i = 0; i < activePath.length - 1; i++) {
						const d = pointToSegmentDistanceMeters([lat, lon], activePath[i], activePath[i + 1]);
						if (d < minD) minD = d;
						if (minD <= halfWidthM) break;
					}
					return minD > halfWidthM;
				});
				if (out && onOutOfRange) onOutOfRange();
			}
		} catch (e) {
			if (!suppressErrors) {
				setError(e.message || 'Failed to fetch routes');
			}
		} finally {
			setLoading(false);
		}

		return true;
	};

	useEffect(() => {
		if (!coordsSignature) return;
		if (lastFetchedCoordsRef.current === coordsSignature) return;
		handleFetchRoutes({ suppressErrors: true });
	// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [coordsSignature]);

	const fitToRoute = (route) => {
		try {
			if (!leafletMapRef.current) return;
			if (!route || !route.polyline) return;
			const bounds = L.latLngBounds(route.polyline.map(([lat, lon]) => L.latLng(lat, lon)));
			leafletMapRef.current.fitBounds(bounds, { padding: [24, 24] });
		} catch (_) {}
	};



	useEffect(() => {
		if (routes.length > 0) {
			fitToRoute(routes[activeIdx] || routes[0]);
		} else if (gpsTrack && gpsTrack.length > 1 && leafletMapRef.current) {
			try {
				const bounds = L.latLngBounds(gpsTrack.map(([lat, lon]) => L.latLng(lat, lon)));
				leafletMapRef.current.fitBounds(bounds, { padding: [24, 24] });
			} catch (_) {}
		} else if (leafletMapRef.current && fromLat && fromLon && toLat && toLon) {
			// Fallback: fit to the straight line between From and To
			try {
				const bounds = L.latLngBounds([
					L.latLng(Number(fromLat), Number(fromLon)),
					L.latLng(Number(toLat), Number(toLon))
				]);
				leafletMapRef.current.fitBounds(bounds, { padding: [24, 24] });
			} catch (_) {}
		}
	// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [activeIdx, routes.length, gpsTrack && gpsTrack.length, fromLat, fromLon, toLat, toLon]);

	return (
		<Card className="mb-4">
			<Card.Header>
				<h6 className="mb-0">{panelTitle}</h6>
			</Card.Header>
			<Card.Body>
				{error && (
					<Alert variant="warning" className="mb-3">
						{error}
					</Alert>
				)}

				<div className="row g-2 mb-3">
					<div className="col-12 col-md-6">
						<Form.Label>From (lat, lon)</Form.Label>
						<div className="d-flex gap-2">
							<Form.Control value={fromLat} onChange={(e) => { setFromLat(e.target.value); onFromChange && onFromChange({ lat: e.target.value, lon: fromLon }); }} placeholder="Lat" />
							<Form.Control value={fromLon} onChange={(e) => { setFromLon(e.target.value); onFromChange && onFromChange({ lat: fromLat, lon: e.target.value }); }} placeholder="Lon" />
						</div>
					</div>
					<div className="col-12 col-md-6">
						<Form.Label>To (lat, lon)</Form.Label>
						<div className="d-flex gap-2">
							<Form.Control value={toLat} onChange={(e) => { setToLat(e.target.value); onToChange && onToChange({ lat: e.target.value, lon: toLon }); }} placeholder="Lat" />
							<Form.Control value={toLon} onChange={(e) => { setToLon(e.target.value); onToChange && onToChange({ lat: toLat, lon: e.target.value }); }} placeholder="Lon" />
						</div>
					</div>
				</div>

				<div className="d-flex justify-content-between align-items-center mb-2">
					<small className="text-muted">Tip: Click the map to set coordinates (click sets From, next click sets To, and so on)</small>
					<Button size="sm" onClick={handleFetchRoutes} disabled={loading || !hasBoth}>
						{loading ? (<><Spinner size="sm" className="me-1" />Getting Routes...</>) : 'Get Routes'}
					</Button>
				</div>

				<div style={{ height: 360, width: '100%' }}>
					<MapContainer
						center={(fromLat && fromLon) ? [Number(fromLat), Number(fromLon)] : ((toLat && toLon) ? [Number(toLat), Number(toLon)] : defaultCenter)}
						zoom={initialZoom}
						style={{ height: '100%', width: '100%' }}
						whenCreated={(map) => {
							leafletMapRef.current = map;
							map.on('click', (e) => {
								const lat = e.latlng.lat;
								const lon = e.latlng.lng;
								if (clickToggleRef.current === 'from' || !fromLat || !fromLon) {
									setFromLat(lat.toFixed(6));
									setFromLon(lon.toFixed(6));
									onFromChange && onFromChange({ lat: lat.toFixed(6), lon: lon.toFixed(6) });
									clickToggleRef.current = 'to';
								} else {
									setToLat(lat.toFixed(6));
									setToLon(lon.toFixed(6));
									onToChange && onToChange({ lat: lat.toFixed(6), lon: lon.toFixed(6) });
									clickToggleRef.current = 'from';
									if ((fromLat || fromLon) && (toLat || toLon)) {
										// Small delay to allow state to update
										setTimeout(() => handleFetchRoutes(), 0);
									}
								}
							});
						}}
					>
                        <MapSizeFix mapRef={leafletMapRef} />
						<TileLayer
							url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
							attribution="&copy; OpenStreetMap contributors"
						/>
						{fromLat && fromLon && (
							<Marker
								position={[Number(fromLat), Number(fromLon)]}
								draggable
								eventHandlers={{ dragend: (e) => {
									const ll = e.target.getLatLng();
									setFromLat(ll.lat.toFixed(6));
									setFromLon(ll.lng.toFixed(6));
									onFromChange && onFromChange({ lat: ll.lat.toFixed(6), lon: ll.lng.toFixed(6) });
									setTimeout(() => handleFetchRoutes(), 0);
								}}}
							>
								<Tooltip permanent>From</Tooltip>
							</Marker>
						)}
						{toLat && toLon && (
							<Marker
								position={[Number(toLat), Number(toLon)]}
								draggable
								eventHandlers={{ dragend: (e) => {
									const ll = e.target.getLatLng();
									setToLat(ll.lat.toFixed(6));
									setToLon(ll.lng.toFixed(6));
									onToChange && onToChange({ lat: ll.lat.toFixed(6), lon: ll.lng.toFixed(6) });
									setTimeout(() => handleFetchRoutes(), 0);
								}}}
							>
								<Tooltip permanent>To</Tooltip>
							</Marker>
						)}

						{routes.map((r, idx) => (
							<Polyline
								key={idx}
								positions={r.polyline}
								pathOptions={{ color: colors[idx % colors.length], weight: (activeIdx === idx ? 6 : 4), opacity: (activeIdx === idx ? 1.0 : 0.7) }}
								eventHandlers={{ click: () => setActiveIdx(idx) }}
							/>
						))}

						{/* Draw uploaded/recorded GPS track if provided */}
						{gpsTrack && gpsTrack.length > 1 && (
							<Polyline
								positions={gpsTrack}
								pathOptions={{ color: '#1565c0', weight: 5, opacity: 0.9 }}
							/>
						)}

						{/* Fallback: draw a straight line between From and To if no routes/gpsTrack yet */}
						{routes.length === 0 && fromLat && fromLon && toLat && toLon && (
							<Polyline
								positions={[[Number(fromLat), Number(fromLon)], [Number(toLat), Number(toLon)]]}
								pathOptions={{ color: '#1565c0', weight: 5, opacity: 0.9 }}
							/>
						)}
					</MapContainer>
				</div>

				{routes.length > 0 && (
					<div className="mt-3">
						<h6>Routes</h6>
						<ul className="mb-0" style={{ listStyle: 'none', paddingLeft: 0 }}>
							{routes.map((r, idx) => {
								const leg = r.legs && r.legs[0];
								const distance = leg?.distance ? `${(leg.distance / 1000).toFixed(1)} km` : `${(r.distance/1000).toFixed(1)} km`;
								const duration = leg?.duration ? `${Math.round(leg.duration/60)} min` : `${Math.round(r.duration/60)} min`;
								return (
									<li key={idx} className="mb-2">
										<span style={{ display: 'inline-block', width: 12, height: 12, backgroundColor: colors[idx % colors.length], marginRight: 8, borderRadius: 2 }}></span>
										<strong>{`Route ${idx + 1}`}</strong> — {distance}, {duration} {activeIdx === idx && <em>(active)</em>}
									</li>
								);
							})}
						</ul>
					</div>
				)}

				{routes[activeIdx]?.legs?.[0]?.steps && (
					<div className="mt-3">
						<h6>Directions</h6>
						<ol className="mb-0" style={{ paddingLeft: '1.2rem' }}>
							{routes[activeIdx].legs[0].steps.map((s, i) => (
								<li key={i} className="mb-1">
									{(s.maneuver && s.maneuver.modifier) ? `${s.maneuver.type || 'Go'} ${s.maneuver.modifier}` : (s.name || 'Continue')} — {s.distance ? `${Math.round(s.distance)} m` : ''}
								</li>
							))}
						</ol>
					</div>
				)}
			</Card.Body>
		</Card>
	);
};

export default MapPanel;