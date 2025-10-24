import math
from typing import List, Tuple, Optional


EARTH_RADIUS_M = 6371000.0


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""Compute great-circle distance between two points on Earth in meters."""
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
	c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
	return EARTH_RADIUS_M * c


def _bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""Initial bearing from point 1 to point 2 in radians."""
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dlambda = math.radians(lon2 - lon1)
	y = math.sin(dlambda) * math.cos(phi2)
	x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
	return math.atan2(y, x)


def _cross_track_distance_m(
	lat: float,
	lon: float,
	lat1: float,
	lon1: float,
	lat2: float,
	lon2: float,
) -> float:
	"""
	Cross-track distance from point to great-circle path between (lat1,lon1)-(lat2,lon2).
	Positive value means point is to the right of path, negative to the left; magnitude is meters.
	"""
	if (lat1 == lat2) and (lon1 == lon2):
		return _haversine_distance_m(lat, lon, lat1, lon1)
		d
	# Angular distance from start to point
	d13 = _haversine_distance_m(lat1, lon1, lat, lon) / EARTH_RADIUS_M
	# Bearings
	theta13 = _bearing_rad(lat1, lon1, lat, lon)
	theta12 = _bearing_rad(lat1, lon1, lat2, lon2)
	return math.asin(math.sin(d13) * math.sin(theta13 - theta12)) * EARTH_RADIUS_M


def is_point_within_corridor(
	point: Tuple[float, float],
	start: Tuple[float, float],
	end: Tuple[float, float],
	half_width_m: float = 50.0,
) -> bool:
	"""
	Check if a given (lat, lon) point lies within a corridor around the start-end path.
	The corridor is a great-circle path buffered by half_width_m meters.
	"""
	lat, lon = point
	lat1, lon1 = start
	lat2, lon2 = end

	# Quick bbox filter using ~111km per degree approximation
	deg_m = 111000.0
	margin_deg = half_width_m / deg_m
	min_lat = min(lat1, lat2) - margin_deg
	max_lat = max(lat1, lat2) + margin_deg
	min_lon = min(lon1, lon2) - margin_deg
	max_lon = max(lon1, lon2) + margin_deg
	if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
		return False

	# Cross-track distance from point to great-circle path
	ct_m = abs(_cross_track_distance_m(lat, lon, lat1, lon1, lat2, lon2))
	if ct_m > half_width_m:
		return False

	# Also ensure point projects between start and end along the path (within along-track)
	# Compare distances: if total < max(d1p, p2) by more than tolerance, it's outside segment
	start_to_end = _haversine_distance_m(lat1, lon1, lat2, lon2)
	start_to_point = _haversine_distance_m(lat1, lon1, lat, lon)
	point_to_end = _haversine_distance_m(lat, lon, lat2, lon2)
	# Triangle inequality tolerance: allow small tolerance due to curvature/rounding
	return start_to_point <= start_to_end + half_width_m and point_to_end <= start_to_end + half_width_m


def validate_path_within_from_to(
	track_points: List[Tuple[float, float]],
	from_location: Tuple[float, float],
	to_location: Tuple[float, float],
	half_width_m: float = 50.0,
	min_coverage_ratio: float = 0.7,
) -> Tuple[bool, Optional[str]]:
	"""
	Validate that a series of GPS points lies substantially within the corridor between from_location and to_location.

	- half_width_m: half width of the corridor around the path centerline
	- min_coverage_ratio: fraction of points that must fall inside the corridor
	Returns (True, None) if valid, else (False, reason)
	"""
	if not track_points:
		return False, "No GPS points captured during upload"

	inside = 0
	for lat, lon in track_points:
		if is_point_within_corridor((lat, lon), from_location, to_location, half_width_m):
			inside += 1

	coverage = inside / float(len(track_points))
	if coverage < min_coverage_ratio:
		return False, f"GPS path coverage {coverage:.0%} below required {min_coverage_ratio:.0%}"

	return True, None



