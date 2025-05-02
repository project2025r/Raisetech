from flask import Blueprint, request, jsonify
import json
import pandas as pd
import os
import math
from config.db import connect_to_db

recommendation_bp = Blueprint('recommendation', __name__)

# Load constants from the original code
MATERIAL_EQUIPMENT_MAP = {
    "Alligator Crack": {
        "materials": ["Hot Mix Asphalt", "Emulsified Asphalt", "Cold Patch"],
        "equipment": ["Pavement Saw", "Asphalt Milling Machine", "Vibratory Roller"]
    },
    "Edge Crack": {
        "materials": ["Hot Mix Asphalt", "Edge Joint Sealant", "Liquid Asphalt"],
        "equipment": ["Asphalt Roller", "Edge Sealing Machine", "Hand Tools"]
    },
    "Hairline Cracks": {
        "materials": ["Crack Sealant", "Liquid Asphalt", "Slurry Seal"],
        "equipment": ["Crack Sealing Machine", "Hand Tools", "Squeegees"]
    },
    "Longitudinal Cracking": {
        "materials": ["Hot Pour Crack Sealant", "Emulsified Asphalt", "Cold Patch"],
        "equipment": ["Routing Machine", "Crack Cleaning Equipment", "Sealing Equipment"]
    },
    "Transverse Cracking": {
        "materials": ["Hot Pour Crack Sealant", "Joint Sealant", "Backer Rod"],
        "equipment": ["Routing Machine", "Air Compressor", "Sealant Applicator"]
    },
    "Pothole": {
        "materials": ["Hot Mix Asphalt", "Cold Patch", "Tack Coat"],
        "equipment": ["Jackhammer", "Compactor", "Hand Tools"]
    }
}

# Add material equipment mapping for potholes from the Streamlit app
POTHOLE_MATERIAL_EQUIPMENT_MAP = {
    "Mixes (cold mixed/hot mixed) for immediate use": [
        "Material truck (with hand tools)",
        "Equipment truck",
        "Asphalt mix carrying and laying equipment",
        "Compaction device (vibratory walk-behind roller or plate compactor)",
        "Mechanical brooms (for highways and urban roads)"
    ],
    "Storable cold mixes (cutback IRC:116/emulsion IRC:100)": [
        "Pug mill or concrete mixer for emulsion-aggregate mixes",
        "Material truck (with hand tools)",
        "Mechanical tool/equipment for pavement cutting and dressing",
        "Compaction device (vibratory walk-behind roller or plate compactor)"
    ],
    "Readymade mixes": [
        "Material truck (with hand tools)",
        "Mechanical tool/equipment for pavement cutting and dressing",
        "Compaction device (vibratory walk-behind roller or plate compactor)"
    ],
    "Cold mixes by patching machines": [
        "Machine Mixed Spot Cold Mix and Patching Equipment",
        "Mobile Mechanized Maintenance Units",
        "Jet Patching Velocity Spray Injection Technology",
        "Mechanical tool/equipment for pavement cutting and dressing"
    ],
    "Open-graded or dense-graded premix": [
        "Material truck (with hand tools)",
        "Compaction device (vibratory walk-behind roller or plate compactor)"
    ],
    "Prime coat": [
        "Material truck (with hand tools)"
    ],
    "Tack coat": [
        "Material truck (with hand tools)",
        "Bitumen Sprayer"
    ]
}

EQUIPMENT_OPTIONS = {
    "Pavement Saw": "Used for cutting precise, straight lines in pavement.",
    "Asphalt Milling Machine": "Removes the top layer of asphalt to create a smooth, even surface.",
    "Vibratory Roller": "Compacts asphalt to improve strength and durability.",
    "Asphalt Roller": "Smooths and compacts new or repaired asphalt surfaces.",
    "Edge Sealing Machine": "Applies sealant to the edges of pavement to prevent water penetration.",
    "Hand Tools": "Include rakes, shovels, and tamps for manual asphalt work.",
    "Crack Sealing Machine": "Applies hot sealant into cracks to prevent water intrusion.",
    "Squeegees": "Used to spread and smooth sealant or other liquid materials.",
    "Routing Machine": "Creates a reservoir in cracks for better sealant adhesion.",
    "Crack Cleaning Equipment": "Removes debris from cracks before sealing.",
    "Sealing Equipment": "Applies sealant to cracks and joints.",
    "Air Compressor": "Blows debris out of cracks and provides power for pneumatic tools.",
    "Sealant Applicator": "Precisely applies sealant to cracks and joints.",
    "Jackhammer": "Breaks up damaged pavement for removal.",
    "Compactor": "Compacts new asphalt in repaired areas."
}

# Add pothole equipment options from the Streamlit app
POTHOLE_EQUIPMENT_OPTIONS = [
    "Material truck (with hand tools)",
    "Equipment truck",
    "Mechanical tool/equipment for pavement cutting and dressing",
    "Compaction device (vibratory walk-behind roller or plate compactor)",
    "Air compressor with pavement cutter",
    "Asphalt mix carrying and laying equipment",
    "Traffic control devices and equipment",
    "Mechanical brooms (for highways and urban roads)",
    "Pug mill or concrete mixer for emulsion-aggregate mixes",
    "Hand rammer for compaction",
    "Small roller for compaction",
    "Drag spreader for smoothening surfaces",
    "Mechanical grit spreader for uniform aggregate spreading",
    "Machine Mixed Spot Cold Mix and Patching Equipment",
    "Mobile Mechanized Maintenance Units",
    "Jet Patching Velocity Spray Injection Technology",
    "Infrared road patching technology",
    "Stiff wire brush for cleaning potholes",
    "Compressed air jetting for removing loose materials",
    "Bitumen Sprayer"
]

# Define the new pothole recommendations function
def get_pothole_recommendations(pothole_data):
    """
    Generate automatic recommendations for pothole repair based on the average volume
    of potholes detected and compute total cost as cost per pothole multiplied by the
    number of potholes.
    """
    print(f"Received pothole data: {pothole_data}")
    print(f"Number of potholes received: {len(pothole_data)}")
    
    if not pothole_data:
        return None

    # Calculate average pothole volume (in cm³) and number of potholes.
    # Handle potential different field names (volume vs Volume)
    avg_volume = 0
    total_volume = 0
    pothole_count = len(pothole_data)
    
    for pothole in pothole_data:
        # Try different possible field names
        volume = 0
        if isinstance(pothole, dict):
            volume = pothole.get("volume", pothole.get("Volume", 0))
            # Convert string values to float if necessary
            if isinstance(volume, str):
                try:
                    volume = float(volume)
                except ValueError:
                    volume = 0
        total_volume += volume
    
    if pothole_count > 0:
        avg_volume = total_volume / pothole_count
    
    print(f"Total volume: {total_volume}, Average volume: {avg_volume}, Pothole count: {pothole_count}")
    
    # Define recommendations based on pothole size.
    if avg_volume < 1000:
        pothole_type = "Small Pothole"
        cost_range = (1400, 1600)  # INR per pothole
        recommendations = {
            "potholeType": pothole_type,
            "materialsRequired": "Mixes (cold/hot) for immediate use, Tack Coat & Bitumen Emulsion",
            "equipmentUsed": "Material Truck (with hand tools), Equipment Truck, Mechanical pavement cutting tool, Compaction device, Mechanical brooms",
            "manpowerRequired": "3–4 workers",
            "timePerPothole": "20 minutes",
            "costPerPothole": f"₹{cost_range[0]}–₹{cost_range[1]}",
            "totalCost": f"₹{cost_range[0] * pothole_count}–₹{cost_range[1] * pothole_count}",
            "durability": "3–6 months",
            "trafficDisruption": "Low",
            "ifNotFixed": "Rapid expansion into a larger pothole",
            "avgVolume": avg_volume,
            "totalPotholes": pothole_count,
            "totalVolume": total_volume
        }
    elif avg_volume < 10000:
        pothole_type = "Medium Pothole"
        cost_range = (2700, 3000)  # INR per pothole
        recommendations = {
            "potholeType": pothole_type,
            "materialsRequired": "Hot Mix Asphalt, Bitumen Emulsion, Open-graded/dense-graded premix",
            "equipmentUsed": "Bitumen Sprayer, Plate Compactor, Material Truck",
            "manpowerRequired": "6–8 workers",
            "timePerPothole": "35 minutes",
            "costPerPothole": f"₹{cost_range[0]}–₹{cost_range[1]}",
            "totalCost": f"₹{cost_range[0] * pothole_count}–₹{cost_range[1] * pothole_count}",
            "durability": "2–3 years",
            "trafficDisruption": "Medium",
            "ifNotFixed": "Increases accident risks and further road damage",
            "avgVolume": avg_volume,
            "totalPotholes": pothole_count,
            "totalVolume": total_volume
        }
    else:
        pothole_type = "Large Pothole"
        cost_range = (3750, 4150)  # INR per pothole
        recommendations = {
            "potholeType": pothole_type,
            "materialsRequired": "Hot Mix Asphalt, Reinforced Patching, Prime Coat, Tack Coat",
            "equipmentUsed": "Spray Injection Patcher, Road Roller, Equipment Truck",
            "manpowerRequired": "3–5 workers",
            "timePerPothole": "50 minutes",
            "costPerPothole": f"₹{cost_range[0]}–₹{cost_range[1]}",
            "totalCost": f"₹{cost_range[0] * pothole_count}–₹{cost_range[1] * pothole_count}",
            "durability": "5+ years",
            "trafficDisruption": "High",
            "ifNotFixed": "Severe road failure leading to costly future repairs",
            "avgVolume": avg_volume,
            "totalPotholes": pothole_count,
            "totalVolume": total_volume
        }

    return recommendations

def analyze_repair(selected_materials, selected_equipment, labor_counts,
                  pothole_count, avg_pothole_volume, road_length):
    """
    Calculates repair estimates based on pothole volume and user selections,
    using the cost estimation rubric with realistic dummy values in INR.

    Pothole category determination:
      - Small: volume < 1000 cm³
      - Medium: 1000 cm³ ≤ volume < 10000 cm³
      - Large: volume ≥ 10000 cm³
    """

    if avg_pothole_volume < 1000:
        category = "Small"
        base_material_cost = 700  # INR per pothole (material)
        base_equipment_cost = 350  # INR per pothole (equipment)
        base_labor_cost = 450  # INR per pothole (labor)
        base_time_per_pothole = 20  # minutes per pothole
        daily_equip_cost = 50000  # INR per additional day for equipment rental
        recommended_materials = 2
        recommended_labor_total = 3.5
        recommended_equipment = 1
    elif avg_pothole_volume < 10000:
        category = "Medium"
        base_material_cost = 1350  # INR per pothole (material)
        base_equipment_cost = 700  # INR per pothole (equipment)
        base_labor_cost = 800  # INR per pothole (labor)
        base_time_per_pothole = 35  # minutes per pothole
        daily_equip_cost = 65000  # INR per additional day for equipment rental
        recommended_materials = 2
        recommended_labor_total = 7
        recommended_equipment = 1
    else:
        category = "Large"
        base_material_cost = 2000  # INR per pothole (material)
        base_equipment_cost = 1050  # INR per pothole (equipment)
        base_labor_cost = 900  # INR per pothole (labor)
        base_time_per_pothole = 50  # minutes per pothole
        daily_equip_cost = 80000  # INR per additional day for equipment rental
        recommended_materials = 2
        recommended_labor_total = 4
        recommended_equipment = 1

    material_factor = (len(selected_materials) / recommended_materials) if selected_materials else 1
    equipment_factor = len(selected_equipment) if selected_equipment else 1
    total_labor_input = (labor_counts.get("Unskilled", 0) +
                        labor_counts.get("Skilled", 0) +
                        labor_counts.get("Supervisors", 0))
    labor_factor = (total_labor_input / recommended_labor_total) if total_labor_input > 0 else 1

    adjusted_material_cost = base_material_cost * material_factor
    adjusted_equipment_cost = base_equipment_cost * equipment_factor
    adjusted_labor_cost = base_labor_cost * labor_factor

    adjusted_cost_per_pothole = adjusted_material_cost + adjusted_equipment_cost + adjusted_labor_cost

    cost_per_pothole_min = adjusted_cost_per_pothole * 0.95
    cost_per_pothole_max = adjusted_cost_per_pothole * 1.05

    base_cost_min = cost_per_pothole_min * pothole_count
    base_cost_max = cost_per_pothole_max * pothole_count

    total_time_minutes = pothole_count * base_time_per_pothole
    repair_days = math.ceil(total_time_minutes / 480)

    extra_days = max(0, repair_days - 1)
    extra_equip_cost = extra_days * daily_equip_cost

    total_cost_min = base_cost_min + extra_equip_cost
    total_cost_max = base_cost_max + extra_equip_cost

    return {
        "category": category,
        "total_time_minutes": total_time_minutes,
        "repair_days": repair_days,
        "base_cost_min": base_cost_min,
        "base_cost_max": base_cost_max,
        "extra_equip_cost": extra_equip_cost,
        "total_cost_min": total_cost_min,
        "total_cost_max": total_cost_max,
        "adjusted_material_cost": adjusted_material_cost,
        "adjusted_equipment_cost": adjusted_equipment_cost,
        "adjusted_labor_cost": adjusted_labor_cost,
        "adjusted_cost_per_pothole": adjusted_cost_per_pothole
    }

def get_pothole_recommendations_by_dimensions(area_cm2, depth_cm, volume):
    """
    Generate recommendations for pothole repair based on dimensions
    """
    # Base repair time and cost calculations
    base_repair_hours = 0.5  # Base time for small potholes
    base_cost_per_m2 = 50  # Base cost in dollars per square meter
    
    # Convert cm² to m²
    area_m2 = area_cm2 / 10000
    
    # Area-based scaling for repair time
    if area_cm2 < 1000:  # Small pothole
        repair_time = base_repair_hours
        repair_method = "Patch with cold mix asphalt"
        urgency = "Low"
    elif area_cm2 < 5000:  # Medium pothole
        repair_time = base_repair_hours * 2
        repair_method = "Remove damaged material and fill with hot mix asphalt"
        urgency = "Medium"
    else:  # Large pothole
        repair_time = base_repair_hours * 4
        repair_method = "Full-depth patching with hot mix asphalt"
        urgency = "High"
    
    # Depth-based scaling for materials needed and equipment
    if depth_cm < 3:
        material_amount = "Minimal"
        equipment = ["Hand Tools", "Compactor"]
    elif depth_cm < 7:
        material_amount = "Moderate"
        equipment = ["Hand Tools", "Compactor", "Asphalt Roller"]
    else:
        material_amount = "Substantial"
        equipment = ["Jackhammer", "Hand Tools", "Compactor", "Asphalt Roller"]
    
    # Cost estimation based on area and depth
    estimated_cost = area_m2 * base_cost_per_m2 * (1 + (depth_cm / 10))
    
    # Safety risk assessment
    if depth_cm > 5 and area_cm2 > 2000:
        safety_risk = "High - Risk to vehicles and cyclists"
    elif depth_cm > 3 or area_cm2 > 1000:
        safety_risk = "Medium - Potential hazard in wet conditions"
    else:
        safety_risk = "Low - Monitor for deterioration"
    
    return {
        "repair_method": repair_method,
        "estimated_time_hours": round(repair_time, 1),
        "estimated_cost_usd": round(estimated_cost, 2),
        "material_amount": material_amount,
        "recommended_equipment": equipment,
        "urgency": urgency,
        "safety_risk": safety_risk
    }

def estimate_cost_for_pothole(area_cm2, depth_cm):
    """
    Estimate repair cost for a pothole based on area and depth
    """
    # Convert cm² to m²
    area_m2 = area_cm2 / 10000
    
    # Base rates
    base_cost_per_m2 = 50  # Base cost in dollars per square meter
    labor_rate_per_hour = 35  # Labor cost per hour
    
    # Calculate material cost
    material_cost = area_m2 * base_cost_per_m2 * (1 + (depth_cm / 10))
    
    # Estimate labor hours
    labor_hours = 0.5  # Base time
    if area_cm2 > 1000:
        labor_hours = 1.0
    if area_cm2 > 5000:
        labor_hours = 2.0
    if depth_cm > 5:
        labor_hours *= 1.5
    
    # Calculate labor cost
    labor_cost = labor_hours * labor_rate_per_hour
    
    # Equipment cost (simplified)
    equipment_cost = labor_hours * 20
    
    # Total cost
    total_cost = material_cost + labor_cost + equipment_cost
    
    return {
        "material_cost": round(material_cost, 2),
        "labor_cost": round(labor_cost, 2),
        "equipment_cost": round(equipment_cost, 2),
        "total_cost": round(total_cost, 2),
        "labor_hours": labor_hours
    }

def analyze_repair_defect(defect_type, area_cm2=None, depth_cm=None, volume=None):
    """
    Generate repair recommendations based on defect type and dimensions
    """
    materials = MATERIAL_EQUIPMENT_MAP.get(defect_type, {}).get("materials", [])
    equipment = MATERIAL_EQUIPMENT_MAP.get(defect_type, {}).get("equipment", [])
    
    equipment_details = {equip: EQUIPMENT_OPTIONS.get(equip, "No description available") for equip in equipment}
    
    if defect_type == "Pothole" and area_cm2 and depth_cm:
        pothole_recs = get_pothole_recommendations_by_dimensions(area_cm2, depth_cm, volume)
        cost_estimate = estimate_cost_for_pothole(area_cm2, depth_cm)
        
        return {
            "defect_type": defect_type,
            "recommended_materials": materials,
            "recommended_equipment": equipment,
            "equipment_details": equipment_details,
            "repair_specifics": pothole_recs,
            "cost_breakdown": cost_estimate
        }
    else:
        # Generic recommendations for cracks
        repair_method = "Clean and seal cracks"
        if defect_type == "Alligator Crack":
            repair_method = "Remove and replace affected area with new asphalt"
            urgency = "High"
        elif defect_type == "Edge Crack":
            repair_method = "Clean, seal and reinforce edge"
            urgency = "Medium"
        else:
            urgency = "Low"
        
        return {
            "defect_type": defect_type,
            "recommended_materials": materials,
            "recommended_equipment": equipment,
            "equipment_details": equipment_details,
            "repair_method": repair_method,
            "urgency": urgency
        }

@recommendation_bp.route('/defect', methods=['POST'])
def get_repair_recommendation():
    """
    API endpoint to get repair recommendations for detected defects
    """
    data = request.json
    
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided"
        }), 400
    
    defect_type = data.get('defect_type')
    area_cm2 = data.get('area_cm2')
    depth_cm = data.get('depth_cm')
    volume = data.get('volume')
    
    if not defect_type:
        return jsonify({
            "success": False,
            "message": "Defect type is required"
        }), 400
    
    # Convert to float if string values are provided
    if area_cm2 and isinstance(area_cm2, str):
        try:
            area_cm2 = float(area_cm2)
        except ValueError:
            area_cm2 = None
    
    if depth_cm and isinstance(depth_cm, str):
        try:
            depth_cm = float(depth_cm)
        except ValueError:
            depth_cm = None
    
    if volume and isinstance(volume, str):
        try:
            volume = float(volume)
        except ValueError:
            volume = None
    
    try:
        recommendations = analyze_repair_defect(defect_type, area_cm2, depth_cm, volume)
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating recommendations: {str(e)}"
        }), 500

@recommendation_bp.route('/auto', methods=['POST'])
def auto_pothole_recommendations():
    """
    API endpoint to get automatic recommendations for potholes
    """
    data = request.json
    
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided"
        }), 400
    
    db = connect_to_db()
    if db is None:
        return jsonify({
            "success": False,
            "message": "Failed to connect to database"
        }), 500
    
    try:
        pothole_data = []
        
        # Check if specific pothole data was provided
        if 'potholeData' in data and data['potholeData']:
            pothole_data = data.get('potholeData', [])
        
        # Check if an image ID was provided to get potholes from a specific image
        elif 'imageId' in data and data['imageId']:
            image_id = data['imageId']
            
            # Try to find the image in the pothole_images collection
            image = db.pothole_images.find_one({"image_id": image_id})
            if image and "potholes" in image:
                pothole_data = image["potholes"]
        
        # If neither specific pothole data nor image ID was provided, get the most recent potholes
        else:
            # First try to get from the new data model
            latest_image = db.pothole_images.find_one({}, sort=[("timestamp", -1)])
            
            if latest_image and "potholes" in latest_image:
                pothole_data = latest_image["potholes"]
            else:
                # Fall back to old data model
                # Get the most recent timestamp
                latest_pothole = db.potholes.find_one({}, {"timestamp": 1}, sort=[("timestamp", -1)])
                
                if latest_pothole:
                    latest_timestamp = latest_pothole["timestamp"]
                    
                    # Get all potholes with that timestamp (assumed to be from the same image)
                    pothole_data = list(db.potholes.find({"timestamp": latest_timestamp}))
        
        if not pothole_data:
            return jsonify({
                "success": False,
                "message": "No pothole data found"
            }), 404
        
        recommendations = get_pothole_recommendations(pothole_data)
        
        if not recommendations:
            return jsonify({
                "success": False,
                "message": "Could not generate recommendations from the provided data"
            }), 500
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating recommendations: {str(e)}"
        }), 500

@recommendation_bp.route('/analyze', methods=['POST'])
def analyze_pothole_repair():
    """
    API endpoint to analyze manual repair parameters for potholes
    """
    data = request.json
    
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided"
        }), 400
    
    try:
        selected_materials = data.get('selectedMaterials', [])
        selected_equipment = data.get('selectedEquipment', [])
        labor_counts = data.get('laborCounts', {})
        pothole_count = data.get('potholeCount', 0)
        avg_pothole_volume = data.get('avgPotholeVolume', 0)
        road_length = data.get('roadLength', 0)
        
        if not selected_materials or pothole_count == 0:
            return jsonify({
                "success": False,
                "message": "Missing required parameters"
            }), 400
        
        results = analyze_repair(
            selected_materials, 
            selected_equipment, 
            labor_counts, 
            pothole_count, 
            avg_pothole_volume, 
            road_length
        )
        
        return jsonify({
            "success": True,
            "results": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error analyzing repair: {str(e)}"
        }), 500

@recommendation_bp.route('/list', methods=['GET'])
def list_recommendations():
    """
    API endpoint to list all repair recommendations
    """
    try:
        db = connect_to_db()
        if db is None:
            return jsonify({
                "success": False,
                "message": "Failed to connect to database"
            }), 500
        
        # Get recommendations from database
        recommendations = list(db.recommendations.find())
        
        # Convert ObjectId to string for JSON serialization
        for rec in recommendations:
            if '_id' in rec:
                rec['_id'] = str(rec['_id'])
                rec['id'] = str(rec['_id'])  # Add id field for frontend compatibility
        
        return jsonify(recommendations)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error retrieving recommendations: {str(e)}"
        }), 500 