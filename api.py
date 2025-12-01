"""
KisaanLink FastAPI Backend
RESTful API for the AI Agronomist application with multi-language support
"""

import os
import uuid
import tempfile
import base64
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after loading env
from graph import (
    app as agent_app, 
    run_agent, 
    DEFAULT_LOCATION,
    get_supported_languages,
    get_default_hyperparams,
    SUPPORTED_LANGUAGES
)
from tools import get_satellite_image, get_crop_health_ndvi, get_agri_weather, check_mandi_prices


# --- Pydantic Models ---
class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class Hyperparameters(BaseModel):
    """User-configurable parameters for the AI agent"""
    language: Optional[str] = Field(default="english", description="Response language (english, urdu, punjabi, sindhi, hindi)")
    history_days: Optional[int] = Field(default=30, ge=7, le=90, description="Days of satellite/weather history to analyze")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="LLM creativity (0=focused, 1=creative)")
    include_satellite: Optional[bool] = Field(default=True, description="Include satellite imagery in analysis")
    include_weather: Optional[bool] = Field(default=True, description="Include weather data in analysis")
    include_ndvi: Optional[bool] = Field(default=True, description="Include NDVI crop health in analysis")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID for memory")
    location: Optional[Location] = Field(default=None, description="User location")
    crop_name: Optional[str] = Field(default=None, description="Crop name for context")
    city: Optional[str] = Field(default=None, description="City for market queries")
    # Hyperparameters
    hyperparams: Optional[Hyperparameters] = Field(default=None, description="AI configuration parameters")


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    timestamp: str
    language: str = "english"
    agent_steps: Optional[List[str]] = None


class ImageAnalysisRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    message: Optional[str] = Field(default="Please analyze this crop image for diseases.", description="Additional context")
    thread_id: Optional[str] = Field(default=None)
    location: Optional[Location] = Field(default=None)
    hyperparams: Optional[Hyperparameters] = Field(default=None, description="AI configuration parameters")


class MarketPriceRequest(BaseModel):
    crop_name: str = Field(..., min_length=1, description="Crop name")
    city: str = Field(..., min_length=1, description="City/Mandi name")
    language: Optional[str] = Field(default="english", description="Response language")


class MarketPriceResponse(BaseModel):
    crop_name: str
    city: str
    price_info: str
    language: str = "english"
    timestamp: str


class SatelliteDataRequest(BaseModel):
    location: Location
    days: Optional[int] = Field(default=30, ge=1, le=60, description="Days to analyze")


class SatelliteDataResponse(BaseModel):
    satellite_image: Optional[dict] = None
    ndvi_data: Optional[dict] = None
    weather_data: Optional[dict] = None
    timestamp: str


class LanguageInfo(BaseModel):
    code: str
    name: str
    native: str
    greeting: str


class SettingsResponse(BaseModel):
    supported_languages: Dict[str, LanguageInfo]
    default_hyperparams: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    earth_engine_status: Optional[str] = None
    google_ai_status: Optional[str] = None
    supported_languages: List[str] = []


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("=" * 60)
    print("🌾 KISAANLINK API STARTING")
    print("=" * 60)
    print(f"   Google AI Key: {'✅ Configured' if os.getenv('GOOGLE_API_KEY') else '❌ Missing'}")
    print(f"   EE Project: {os.getenv('EE_PROJECT', '❌ Not Set')}")
    print("=" * 60)
    yield
    print("🌾 KisaanLink API Shutting down...")


# --- FastAPI App ---
app = FastAPI(
    title="KisaanLink API",
    description="""
    AI-powered Agricultural Assistant API for Pakistani farmers.
    
    Features:
    - 🌱 Crop disease diagnosis from images
    - 📊 Market/Mandi price queries
    - 🛰️ Satellite-based crop health monitoring (Google Earth Engine)
    - 🌤️ Weather and soil analysis
    - 💬 General farming advice chatbot
    """,
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def cleanup_temp_file(file_path: str):
    """Background task to cleanup temporary files"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass


def get_timestamp() -> str:
    return datetime.now().isoformat()


# --- API Endpoints ---

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Health check endpoint"""
    from tools import EE_INITIALIZED, EE_PROJECT
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=get_timestamp(),
        earth_engine_status=f"✅ Connected (project: {EE_PROJECT})" if EE_INITIALIZED else "❌ Not initialized",
        google_ai_status="✅ Configured" if os.getenv("GOOGLE_API_KEY") else "❌ Missing API key",
        supported_languages=list(SUPPORTED_LANGUAGES.keys())
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check with service status"""
    from tools import EE_INITIALIZED, EE_PROJECT
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=get_timestamp(),
        earth_engine_status=f"✅ Connected (project: {EE_PROJECT})" if EE_INITIALIZED else "❌ Not initialized",
        google_ai_status="✅ Configured" if os.getenv("GOOGLE_API_KEY") else "❌ Missing API key",
        supported_languages=list(SUPPORTED_LANGUAGES.keys())
    )


@app.get("/settings", response_model=SettingsResponse, tags=["Settings"])
async def get_settings():
    """
    Get available settings and defaults.
    
    Returns supported languages with their native names and default hyperparameters.
    """
    languages_info = {}
    for code, info in SUPPORTED_LANGUAGES.items():
        languages_info[code] = LanguageInfo(
            code=code,
            name=info["name"],
            native=info["native"],
            greeting=info["greeting"]
        )
    
    return SettingsResponse(
        supported_languages=languages_info,
        default_hyperparams=get_default_hyperparams()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the AI agronomist.
    
    The agent will automatically route your query to the appropriate expert:
    - Disease diagnosis (if image context available)
    - Market prices (if asking about prices/rates)
    - Satellite/weather analysis (if asking about crop health)
    - General farming advice (everything else)
    
    You can tune the response via hyperparams:
    - language: Response language (english, urdu, punjabi, sindhi, hindi)
    - history_days: Days of satellite/weather data to analyze
    - temperature: AI creativity (0=focused, 1=creative)
    - include_satellite/weather/ndvi: Toggle data sources
    
    Natural language settings also work:
    - "Switch to Urdu" or "Show me 60 days of weather"
    """
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        location = {"lat": request.location.lat, "lon": request.location.lon} if request.location else None
        
        # Extract hyperparameters
        hp = request.hyperparams or Hyperparameters()
        
        response = run_agent(
            user_message=request.message,
            location=location,
            crop_name=request.crop_name,
            city=request.city,
            thread_id=thread_id,
            language=hp.language,
            history_days=hp.history_days,
            temperature=hp.temperature,
            include_satellite=hp.include_satellite,
            include_weather=hp.include_weather,
            include_ndvi=hp.include_ndvi
        )
        
        return ChatResponse(
            response=response,
            thread_id=thread_id,
            timestamp=get_timestamp(),
            language=hp.language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.post("/analyze-image", response_model=ChatResponse, tags=["Disease Diagnosis"])
async def analyze_image(request: ImageAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze a crop image for disease diagnosis.
    
    Send a base64-encoded image and get:
    - Disease identification
    - Treatment recommendations
    - Environmental context from satellite data
    
    Supports multiple languages via hyperparams.language
    """
    temp_file_path = None
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        location = {"lat": request.location.lat, "lon": request.location.lon} if request.location else DEFAULT_LOCATION
        hp = request.hyperparams or Hyperparameters()
        
        # Decode base64 image and save to temp file
        image_data = base64.b64decode(request.image_base64)
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(image_data)
        temp_file.close()
        temp_file_path = temp_file.name
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        response = run_agent(
            user_message=request.message,
            image_path=temp_file_path,
            location=location,
            thread_id=thread_id,
            language=hp.language,
            temperature=hp.temperature
        )
        
        return ChatResponse(
            response=response,
            thread_id=thread_id,
            timestamp=get_timestamp(),
            language=hp.language,
            agent_steps=["Pathologist", "Treatment Advisor"]
        )
    except Exception as e:
        # Cleanup on error
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")


@app.post("/analyze-image-upload", response_model=ChatResponse, tags=["Disease Diagnosis"])
async def analyze_image_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Crop image file"),
    message: str = Form(default="Please analyze this crop image for diseases."),
    thread_id: Optional[str] = Form(default=None),
    lat: Optional[float] = Form(default=None),
    lon: Optional[float] = Form(default=None),
    language: str = Form(default="english")
):
    """
    Analyze a crop image uploaded as a file.
    
    Alternative to base64 endpoint for direct file uploads.
    Supports multiple response languages.
    """
    temp_file_path = None
    try:
        thread_id = thread_id or str(uuid.uuid4())
        location = {"lat": lat, "lon": lon} if lat and lon else DEFAULT_LOCATION
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1] or ".jpg"
        
        # Save uploaded file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        temp_file_path = temp_file.name
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        response = run_agent(
            user_message=message,
            image_path=temp_file_path,
            location=location,
            thread_id=thread_id,
            language=language
        )
        
        return ChatResponse(
            response=response,
            thread_id=thread_id,
            timestamp=get_timestamp(),
            language=language,
            agent_steps=["Pathologist", "Treatment Advisor"]
        )
    except Exception as e:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")


@app.post("/analyze-image-upload", response_model=ChatResponse, tags=["Disease Diagnosis"])
async def analyze_image_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Crop image file"),
    message: str = Form(default="Please analyze this crop image for diseases."),
    thread_id: Optional[str] = Form(default=None),
    lat: Optional[float] = Form(default=None),
    lon: Optional[float] = Form(default=None)
):
    """
    Analyze a crop image uploaded as a file.
    
    Alternative to base64 endpoint for direct file uploads.
    """
    temp_file_path = None
    try:
        thread_id = thread_id or str(uuid.uuid4())
        location = {"lat": lat, "lon": lon} if lat and lon else DEFAULT_LOCATION
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1] or ".jpg"
        
        # Save uploaded file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        temp_file_path = temp_file.name
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        response = run_agent(
            user_message=message,
            image_path=temp_file_path,
            location=location,
            thread_id=thread_id
        )
        
        return ChatResponse(
            response=response,
            thread_id=thread_id,
            timestamp=get_timestamp(),
            agent_steps=["Pathologist", "Treatment Advisor"]
        )
    except Exception as e:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")


@app.post("/market-prices", response_model=MarketPriceResponse, tags=["Market Prices"])
async def get_market_prices(request: MarketPriceRequest):
    """
    Get current mandi/market prices for a crop in a specific city.
    """
    try:
        result = check_mandi_prices.invoke({
            "crop_name": request.crop_name,
            "city": request.city
        })
        
        return MarketPriceResponse(
            crop_name=request.crop_name,
            city=request.city,
            price_info=str(result),
            timestamp=get_timestamp()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market prices: {str(e)}")


@app.post("/satellite-data", response_model=SatelliteDataResponse, tags=["Satellite Data"])
async def get_satellite_data(request: SatelliteDataRequest):
    """
    Get satellite-derived agricultural data for a location.
    
    Returns:
    - Satellite imagery URL (Sentinel-2 false color)
    - NDVI crop health index
    - Weather and soil conditions (ERA5)
    """
    try:
        lat = request.location.lat
        lon = request.location.lon
        days = request.days
        
        # Fetch all satellite data
        satellite_image = get_satellite_image.invoke({"lat": lat, "lon": lon})
        ndvi_data = get_crop_health_ndvi.invoke({"lat": lat, "lon": lon, "days": days})
        weather_data = get_agri_weather.invoke({"lat": lat, "lon": lon, "days": days})
        
        return SatelliteDataResponse(
            satellite_image=satellite_image if isinstance(satellite_image, dict) else {"raw": str(satellite_image)},
            ndvi_data=ndvi_data if isinstance(ndvi_data, dict) else {"raw": str(ndvi_data)},
            weather_data=weather_data if isinstance(weather_data, dict) else {"raw": str(weather_data)},
            timestamp=get_timestamp()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching satellite data: {str(e)}")


@app.get("/satellite-image", tags=["Satellite Data"])
async def get_satellite_image_url(
    lat: float,
    lon: float
):
    """
    Get just the satellite image URL for a location.
    Quick endpoint for displaying satellite imagery.
    
    - lat: Latitude (-90 to 90)
    - lon: Longitude (-180 to 180)
    """
    if not -90 <= lat <= 90:
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")
    if not -180 <= lon <= 180:
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180")
    
    try:
        result = get_satellite_image.invoke({"lat": lat, "lon": lon})
        
        if isinstance(result, dict):
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"result": str(result)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching satellite image: {str(e)}")


@app.get("/crop-health", tags=["Satellite Data"])
async def get_crop_health(
    lat: float,
    lon: float,
    days: int = 30
):
    """
    Get NDVI-based crop health assessment for a location.
    """
    try:
        result = get_crop_health_ndvi.invoke({"lat": lat, "lon": lon, "days": days})
        
        if isinstance(result, dict):
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"result": str(result)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching crop health: {str(e)}")


@app.get("/weather", tags=["Satellite Data"])
async def get_weather(
    lat: float,
    lon: float,
    days: int = 30
):
    """
    Get agricultural weather data (temperature, precipitation, soil moisture) for a location.
    """
    try:
        result = get_agri_weather.invoke({"lat": lat, "lon": lon, "days": days})
        
        if isinstance(result, dict):
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"result": str(result)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")


# --- Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() == "true"
    )
