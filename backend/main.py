from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your API routers
from routes import auth, pavement, road_infrastructure, recommendation, dashboard, users
from routes.pavement import preload_models_on_startup

app = FastAPI(
    title="Road AI Safety Enhancement API",
    description="A FastAPI-powered API for detecting road defects and infrastructure analysis.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend origin e.g., "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Actions to run on application startup."""
    logger.info("🚀 Starting Road AI Safety Enhancement API server...")
    logger.info("🧠 Preloading AI models...")
    preload_models_on_startup()
    logger.info("✅ Models preloaded successfully.")

@app.get("/")
async def root():
    return {
        "message": "Road AI Safety Enhancement API",
        "status": "online",
        "version": "1.0.0"
    }

# Include all the routers from your application
app.include_router(auth.router, prefix='/api/auth', tags=['Authentication'])
app.include_router(pavement.router, prefix='/api/pavement', tags=['Pavement'])
app.include_router(road_infrastructure.router, prefix='/api/road-infrastructure', tags=['Road Infrastructure'])
app.include_router(recommendation.router, prefix='/api/recommendation', tags=['Recommendation'])
app.include_router(dashboard.router, prefix='/api/dashboard', tags=['Dashboard'])
app.include_router(users.router, prefix='/api/users', tags=['Users'])

if __name__ == "__main__":
    logger.info("Starting server with Uvicorn...")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
