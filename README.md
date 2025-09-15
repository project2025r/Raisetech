# Road AI Safety Enhancement Application

# Testing Git -Vish

This application is designed to detect and analyze road conditions using computer vision and AI technology. It helps identify potholes, cracks, and various road infrastructure elements to assist in maintenance planning and road safety enhancement.

## Project Structure

```
ReactApp/
  ├── backend/          # Flask backend API
  ├── frontend/         # React.js frontend
  ├── assets/           # Images and model files
  ├── dashboard/        # Dashboard data files
  └── map_frames/       # Map visualization frames
```

## Technology Stack

- **Backend:** Flask (Python)
- **Frontend:** React.js
- **Database:** MongoDB
- **Computer Vision:** OpenCV, PyTorch, YOLO
- **Data Visualization:** Plotly

## Setup Instructions

### Prerequisites

- Python 3.9+
- Node.js 16+
- MongoDB database

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd ReactApp/backend
   ```

2. Create a Python virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the backend directory with the following variables:
   ```
   MONGO_URI=mongodb://localhost:27017
   DB_NAME=LTA

   ```

6. Start the Flask server:
   ```
   python app.py
   ```
   The backend will run on http://localhost:5000.

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd ReactApp/frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the React development server:
   ```
   npm start
   ```
   The frontend will run on http://localhost:3000.

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Log in with the following credentials:
   - Username: `supervisor1`, Password: `123456`, Role: `Supervisor`
   - Username: `fieldofficer1`, Password: `123456`, Role: `Field Officer`
   - Username: `admin1`, Password: `123456`, Role: `Admin`

3. Use the sidebar to navigate between different modules:
   - Home: Overview of all modules
   - Pavement: Detect potholes and cracks in road pavements
   - Road Infrastructure: Identify road infrastructure elements like kerbs
   - Recommendation: Get repair recommendations for detected issues
   - Dashboard: View statistics and analytics from collected data

## Features

- **Authentication:** Role-based access control
- **Pavement Analysis:** Detection of potholes and various types of cracks
- **Road Infrastructure:** Detection of road infrastructure elements
- **Recommendation:** AI-driven repair recommendations with cost estimates
- **Dashboard:** Visualization of collected data and statistics

## Credits

Based on the StreamlitApp version of the Road AI Safety Enhancement system by AiSPRY.

## License

Copyright © 2025 Road AI Safety Enhancement. All rights reserved. 