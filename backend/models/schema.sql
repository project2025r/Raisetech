-- MongoDB Schema for video_processing collection
/*
{
  "video_id": ObjectId,  // Primary key
  "original_video_url": String,  // S3 URL of original video
  "processed_video_url": String,  // S3 URL of processed video
  "s3_path": String,  // Full S3 path
  "role": String,  // Admin / Supervisor / Field Officer
  "username": String,  // Uploader's name/id
  "timestamp": DateTime,  // Upload time
  "models_run": [String],  // List of models used ["kerbs", "cracks", etc]
  "status": String,  // processing / completed / failed
  "model_outputs": {
    "kerbs": [
      {
        "kerb_id": String,
        "kerb_type": String,
        "coordinates": Array,
        "confidence": Float,
        "timestamp": DateTime,
        "bbox": Array
      }
    ],
    "potholes": [
      {
        "pothole_id": String,
        "pothole_type": String,
        "area_cm2": Float,
        "area_range": String,
        "coordinates": Array,
        "confidence": Float,
        "timestamp": DateTime,
        "bbox": Array
      }
    ],
    "cracks": [
      {
        "crack_id": String,
        "crack_type": String,
        "area_cm2": Float,
        "area_range": String,
        "coordinates": Array,
        "confidence": Float,
        "timestamp": DateTime,
        "bbox": Array
      }
    ]
  },
  "created_at": DateTime,
  "updated_at": DateTime
}
*/

-- Create potholes table
CREATE TABLE IF NOT EXISTS potholes (
    id SERIAL PRIMARY KEY,
    pothole_id INTEGER NOT NULL,
    area_cm2 FLOAT NOT NULL,
    depth_cm FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    volume_range VARCHAR(255) NOT NULL,
    coordinates VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create cracks table
CREATE TABLE IF NOT EXISTS cracks (
    id SERIAL PRIMARY KEY,
    crack_id INTEGER NOT NULL,
    crack_type VARCHAR(255) NOT NULL,
    area_cm2 FLOAT NOT NULL,
    area_range VARCHAR(255) NOT NULL,
    coordinates VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create kerbs table
CREATE TABLE IF NOT EXISTS kerbs (
    id SERIAL PRIMARY KEY,
    kerb_id INTEGER NOT NULL,
    kerb_type VARCHAR(255) NOT NULL,
    length_m FLOAT NOT NULL,
    condition VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL,
    coordinates VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    location VARCHAR(255) NOT NULL,
    issueType VARCHAR(255) NOT NULL,
    priority VARCHAR(50) NOT NULL, -- High, Medium, Low
    estimatedCost FLOAT NOT NULL,
    status VARCHAR(50) DEFAULT 'Pending', -- Pending, Approved, Rejected
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
); 