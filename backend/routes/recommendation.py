from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Optional
import json
import pandas as pd
import os
import math
from config.db import connect_to_db
from pydantic import BaseModel

router = APIRouter()

class DefectPayload(BaseModel):
    defect_type: str
    area_cm2: Optional[float] = None
    depth_cm: Optional[float] = None
    volume: Optional[float] = None

class AutoPotholePayload(BaseModel):
    potholeData: Optional[List[Dict]] = None
    imageId: Optional[str] = None

class AnalyzePayload(BaseModel):
    selectedMaterials: List[str]
    selectedEquipment: List[str]
    laborCounts: Dict[str, int]
    potholeCount: int
    avgPotholeVolume: float
    roadLength: float

# --- Pydantic Models ---
# Pydantic models are defined above

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
    if not pothole_data:
        return None

    total_volume = 0
    pothole_count = len(pothole_data)
    
    for pothole in pothole_data:
        volume = 0
        if isinstance(pothole, dict):
            volume = pothole.get("volume", pothole.get("Volume", 0))
            if isinstance(volume, str):
                try:
                    volume = float(volume)
                except ValueError:
                    volume = 0
        total_volume += volume
    
    avg_volume = total_volume / pothole_count if pothole_count > 0 else 0
    
    if avg_volume < 1000:
        pothole_type = "Small Pothole"
        cost_range = (1400, 1600)
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
        cost_range = (2700, 3000)
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
        cost_range = (3750, 4150)
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
    if avg_pothole_volume < 1000:
        category = "Small"
        base_material_cost = 700
        base_equipment_cost = 350
        base_labor_cost = 450
        base_time_per_pothole = 20
        daily_equip_cost = 50000
        recommended_materials = 2
        recommended_labor_total = 3.5
        recommended_equipment = 1
    elif avg_pothole_volume < 10000:
        category = "Medium"
        base_material_cost = 1350
        base_equipment_cost = 700
        base_labor_cost = 800
        base_time_per_pothole = 35
        daily_equip_cost = 65000
        recommended_materials = 2
        recommended_labor_total = 7
        recommended_equipment = 1
    else:
        category = "Large"
        base_material_cost = 2000
        base_equipment_cost = 1050
        base_labor_cost = 900
        base_time_per_pothole = 50
        daily_equip_cost = 80000
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
    base_repair_hours = 0.5
    base_cost_per_m2 = 50
    area_m2 = area_cm2 / 10000
    
    if area_cm2 < 1000:
        repair_time = base_repair_hours
        repair_method = "Patch with cold mix asphalt"
        urgency = "Low"
    elif area_cm2 < 5000:
        repair_time = base_repair_hours * 2
        repair_method = "Remove damaged material and fill with hot mix asphalt"
        urgency = "Medium"
    else:
        repair_time = base_repair_hours * 4
        repair_method = "Full-depth patching with hot mix asphalt"
        urgency = "High"
    
    if depth_cm < 3:
        material_amount = "Minimal"
        equipment = ["Hand Tools", "Compactor"]
    elif depth_cm < 7:
        material_amount = "Moderate"
        equipment = ["Hand Tools", "Compactor", "Asphalt Roller"]
    else:
        material_amount = "Substantial"
        equipment = ["Jackhammer", "Hand Tools", "Compactor", "Asphalt Roller"]
    
    estimated_cost = area_m2 * base_cost_per_m2 * (1 + (depth_cm / 10))
    
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
    area_m2 = area_cm2 / 10000
    base_cost_per_m2 = 50
    labor_rate_per_hour = 35
    material_cost = area_m2 * base_cost_per_m2 * (1 + (depth_cm / 10))
    labor_hours = 0.5
    if area_cm2 > 1000:
        labor_hours = 1.0
    if area_cm2 > 5000:
        labor_hours = 2.0
    if depth_cm > 5:
        labor_hours *= 1.5
    labor_cost = labor_hours * labor_rate_per_hour
    equipment_cost = labor_hours * 20
    total_cost = material_cost + labor_cost + equipment_cost
    
    return {
        "material_cost": round(material_cost, 2),
        "labor_cost": round(labor_cost, 2),
        "equipment_cost": round(equipment_cost, 2),
        "total_cost": round(total_cost, 2),
        "labor_hours": labor_hours
    }

def analyze_repair_defect(defect_type, area_cm2=None, depth_cm=None, volume=None):
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

@router.post('/defect')
def get_repair_recommendation(payload: DefectPayload):
    try:
        recommendations = analyze_repair_defect(payload.defect_type, payload.area_cm2, payload.depth_cm, payload.volume)
        return {"success": True, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post('/auto')
def auto_pothole_recommendations(payload: AutoPotholePayload):
    db = connect_to_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Failed to connect to database")
    
    try:
        pothole_data = []
        if payload.potholeData:
            pothole_data = payload.potholeData
        elif payload.imageId:
            image = db.pothole_images.find_one({"image_id": payload.imageId})
            if image and "potholes" in image:
                pothole_data = image["potholes"]
        else:
            latest_image = db.pothole_images.find_one({}, sort=[("timestamp", -1)])
            if latest_image and "potholes" in latest_image:
                pothole_data = latest_image["potholes"]
            else:
                latest_pothole = db.potholes.find_one({}, {"timestamp": 1}, sort=[("timestamp", -1)])
                if latest_pothole:
                    latest_timestamp = latest_pothole["timestamp"]
                    pothole_data = list(db.potholes.find({"timestamp": latest_timestamp}))
        
        if not pothole_data:
            raise HTTPException(status_code=404, detail="No pothole data found")
        
        recommendations = get_pothole_recommendations(pothole_data)
        
        if not recommendations:
            raise HTTPException(status_code=500, detail="Could not generate recommendations from the provided data")
        
        return {"success": True, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post('/analyze')
def analyze_pothole_repair(payload: AnalyzePayload):
    try:
        results = analyze_repair(
            payload.selectedMaterials, 
            payload.selectedEquipment, 
            payload.laborCounts, 
            payload.potholeCount, 
            payload.avgPotholeVolume, 
            payload.roadLength
        )
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing repair: {str(e)}")

@router.get('/list')
def list_recommendations():
    try:
        db = connect_to_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Failed to connect to database")
        
        recommendations = list(db.recommendations.find())
        
        for rec in recommendations:
            if '_id' in rec:
                rec['_id'] = str(rec['_id'])
                rec['id'] = str(rec['_id'])
        
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recommendations: {str(e)}")