"""
KisaanLink - External Data Tools
================================

This module provides LangChain tools for fetching external agricultural data.

Data Sources:
1. Google Earth Engine (Sentinel-2): High-resolution satellite imagery
2. Google Earth Engine (ERA5-Land): Climate/weather data
3. AMIS Pakistan (amis.pk): Official Mandi prices
4. DuckDuckGo Search: Fallback for market prices

Design Decisions:
- Tools are decorated with @tool for LangChain compatibility
- Earth Engine uses high-volume endpoint for better performance
- buffer_meters parameter allows customizable area selection
- Graceful fallbacks when primary sources fail

Note: Earth Engine requires authentication via `earthengine authenticate`

Author: KisaanLink Team
"""

import os
import logging
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import ee

load_dotenv()

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# EARTH ENGINE INITIALIZATION
# =============================================================================
# Earth Engine requires a Google Cloud project with the API enabled
EE_PROJECT = os.getenv("EE_PROJECT", None)
EE_INITIALIZED = False


def _init_earth_engine():
    """
    Initialize Google Earth Engine with proper authentication.
    
    Uses high-volume endpoint for better performance in production.
    Falls back gracefully if authentication fails.
    """
    global EE_INITIALIZED
    
    logger.info("=" * 50)
    logger.info("🛰️  EARTH ENGINE CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"   EE_PROJECT env var: {EE_PROJECT or 'NOT SET'}")
    
    try:
        # Use high-volume endpoint for production workloads
        if EE_PROJECT:
            ee.Initialize(project=EE_PROJECT, opt_url='https://earthengine-highvolume.googleapis.com')
            logger.info(f"   ✅ Earth Engine initialized with project: {EE_PROJECT}")
        else:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            logger.info("   ✅ Earth Engine initialized (default project)")
        
        # Verify connection with a simple test
        test = ee.Number(1).getInfo()
        logger.info(f"   ✅ Earth Engine connection verified (test={test})")
        EE_INITIALIZED = True
        
    except Exception as e:
        logger.error(f"   ❌ Earth Engine initialization failed: {e}")
        logger.error("   ")
        logger.error("   To fix this:")
        logger.error("   1. Run: earthengine authenticate")
        logger.error("   2. Set EE_PROJECT in .env file with your Google Cloud project ID")
        logger.error("   3. Make sure Earth Engine API is enabled in Google Cloud Console")
        EE_INITIALIZED = False
    
    logger.info("=" * 50)


# Initialize Earth Engine on module load
_init_earth_engine()


def _check_ee_initialized():
    """Guard function to check Earth Engine status before API calls."""
    if not EE_INITIALIZED:
        return False, "Earth Engine not initialized. Please check your configuration."
    return True, None


# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_LOOKBACK_DAYS = 30  # Default days for weather/NDVI analysis


# =============================================================================
# TOOL 1: SATELLITE IMAGERY (Sentinel-2)
# =============================================================================
@tool
def get_satellite_image(lat: float, lon: float, start_date: str = None, end_date: str = None, buffer_meters: int = 500):
    """
    Fetches a Sentinel-2 satellite image for a specific location showing crop health.
    Returns a URL to the image tile showing vegetation in false color (NIR/Red/Green).
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location  
        start_date: Start date (YYYY-MM-DD), defaults to 60 days ago
        end_date: End date (YYYY-MM-DD), defaults to today
        buffer_meters: Radius in meters around the point (default 500m = 1km diameter)
    """
    # Check if EE is initialized
    is_ready, error = _check_ee_initialized()
    if not is_ready:
        return {"error": error, "source": "Google Earth Engine"}
    
    try:
        # Dynamic date range
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        
        logger.info(f"🛰️ Fetching satellite image for ({lat}, {lon}) buffer={buffer_meters}m from {start_date} to {end_date}")
        
        # Define the Region of Interest (ROI) with configurable buffer
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(buffer_meters).bounds()

        # Get Sentinel-2 Image Collection (Surface Reflectance)
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Less than 20% clouds
            .sort('CLOUDY_PIXEL_PERCENTAGE')
            .first()
        )

        # Check if image exists
        if collection is None:
            return "Error: No cloud-free satellite images available for this location and date range."

        # False Color visualization for Agriculture (NIR, Red, Green)
        # Bright red = healthy vegetation, Gray/Brown = stressed or bare soil
        vis_params = {
            'min': 0,
            'max': 3000,
            'bands': ['B8', 'B4', 'B3']  # NIR, Red, Green
        }

        # Generate a Thumbnail URL (1024px for higher resolution)
        image_url = collection.getThumbURL({
            'region': roi,
            'dimensions': 1024,
            'format': 'png',
            **vis_params
        })
        
        # Get image date
        image_date = ee.Date(collection.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        return {
            "source": "Google Earth Engine - Sentinel-2",
            "image_url": image_url,
            "image_date": image_date,
            "location": {"lat": lat, "lon": lon},
            "visualization": "False Color (NIR/Red/Green) - Bright red indicates healthy vegetation"
        }

    except Exception as e:
        return f"Error fetching satellite image: {str(e)}"


# --- 2. Earth Engine NDVI (Crop Health Index) Tool ---
@tool
def get_crop_health_ndvi(lat: float, lon: float, days: int = DEFAULT_LOOKBACK_DAYS, buffer_meters: int = 500):
    """
    Calculates NDVI (Normalized Difference Vegetation Index) for crop health assessment.
    NDVI ranges from -1 to 1: Higher values (0.6-0.9) = healthy crops, Low values (<0.2) = stressed/bare.
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        days: Number of days to analyze (default: 30)
        buffer_meters: Radius in meters around the point (default 500m)
    """
    # Check if EE is initialized
    is_ready, error = _check_ee_initialized()
    if not is_ready:
        return {"error": error, "source": "Google Earth Engine"}
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"🌱 Calculating NDVI for ({lat}, {lon}) buffer={buffer_meters}m over {days} days")
        
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(buffer_meters).bounds()

        # Get Sentinel-2 imagery
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi)
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        )
        
        # Calculate NDVI for each image
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        with_ndvi = collection.map(add_ndvi)
        
        # Get mean NDVI over the period
        mean_ndvi = with_ndvi.select('NDVI').mean()
        
        # Sample the NDVI value at the point
        ndvi_value = mean_ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(100),
            scale=10
        ).get('NDVI').getInfo()
        
        # Interpret NDVI value
        if ndvi_value is None:
            interpretation = "No data available"
            health_status = "Unknown"
        elif ndvi_value > 0.6:
            interpretation = "Excellent crop health - Dense, healthy vegetation"
            health_status = "Excellent"
        elif ndvi_value > 0.4:
            interpretation = "Good crop health - Moderate vegetation density"
            health_status = "Good"
        elif ndvi_value > 0.2:
            interpretation = "Fair crop health - Sparse vegetation or early growth stage"
            health_status = "Fair"
        else:
            interpretation = "Poor crop health or bare soil - May indicate stress, disease, or harvested field"
            health_status = "Poor/Bare"
        
        return {
            "source": "Google Earth Engine - Sentinel-2 NDVI",
            "ndvi_value": round(ndvi_value, 3) if ndvi_value else None,
            "health_status": health_status,
            "interpretation": interpretation,
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "location": {"lat": lat, "lon": lon}
        }

    except Exception as e:
        return f"Error calculating NDVI: {str(e)}"


# --- 3. Earth Engine Weather Data Tool ---
@tool
def get_agri_weather(lat: float, lon: float, days: int = DEFAULT_LOOKBACK_DAYS):
    """
    Fetches agricultural weather data (Temperature, Precipitation, Soil Moisture) 
    from Google Earth Engine ERA5 climate data.
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        days: Number of days to fetch (default: 30, max: 60)
    """
    # Check if EE is initialized
    is_ready, error = _check_ee_initialized()
    if not is_ready:
        return {"error": error, "source": "Google Earth Engine"}
    
    try:
        end_date = datetime.now() - timedelta(days=5)  # ERA5 has ~5 day lag
        start_date = end_date - timedelta(days=min(days, 60))
        
        logger.info(f"🌤️ Fetching weather data for ({lat}, {lon}) over {days} days")
        
        point = ee.Geometry.Point([lon, lat])
        
        # ERA5-Land Daily data (better resolution for agriculture)
        era5 = (
            ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filterBounds(point)
        )
        
        # Get mean values over the period
        mean_data = era5.mean()
        
        # Extract values at the point
        values = mean_data.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=11132  # ERA5-Land resolution
        ).getInfo()
        
        # Convert units
        temp_k = values.get('temperature_2m')
        temp_c = round(temp_k - 273.15, 1) if temp_k else None
        
        precip = values.get('total_precipitation_sum')
        precip_mm = round(precip * 1000, 2) if precip else None  # Convert m to mm
        
        soil_moisture = values.get('volumetric_soil_water_layer_1')
        soil_moisture_pct = round(soil_moisture * 100, 1) if soil_moisture else None
        
        # Get latest day data for current conditions
        latest = era5.sort('system:time_start', False).first()
        latest_values = latest.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=11132
        ).getInfo()
        
        latest_temp_k = latest_values.get('temperature_2m')
        latest_temp_c = round(latest_temp_k - 273.15, 1) if latest_temp_k else None
        
        return {
            "source": "Google Earth Engine - ERA5-Land",
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "location": {"lat": lat, "lon": lon},
            "current_conditions": {
                "temperature_celsius": latest_temp_c
            },
            "period_averages": {
                "avg_temperature_celsius": temp_c,
                "total_precipitation_mm": precip_mm,
                "soil_moisture_percent": soil_moisture_pct
            },
            "agricultural_advisory": _get_weather_advisory(temp_c, precip_mm, soil_moisture_pct)
        }

    except Exception as e:
        return f"Error fetching weather data: {str(e)}"


def _get_weather_advisory(temp: float, precip: float, soil_moisture: float) -> str:
    """Generate agricultural advisory based on weather data"""
    advisories = []
    
    if temp is not None:
        if temp > 35:
            advisories.append("⚠️ High temperature - ensure adequate irrigation")
        elif temp < 5:
            advisories.append("⚠️ Low temperature - protect crops from frost")
    
    if precip is not None:
        if precip > 50:
            advisories.append("🌧️ High rainfall - avoid pesticide spraying, check for waterlogging")
        elif precip < 10:
            advisories.append("☀️ Low rainfall - consider irrigation")
    
    if soil_moisture is not None:
        if soil_moisture < 20:
            advisories.append("🏜️ Low soil moisture - irrigation recommended")
        elif soil_moisture > 40:
            advisories.append("💧 High soil moisture - reduce watering, monitor for fungal diseases")
    
    return " | ".join(advisories) if advisories else "✅ Conditions are favorable for most crops"


# --- 4. Market Price Tool (AMIS Pakistan + Search Fallback) ---
@tool
def check_mandi_prices(crop_name: str, city: str):
    """
    Fetches the latest market (Mandi) prices for a specific crop in Pakistan.
    Primary source: AMIS Pakistan (amis.pk)
    Fallback: DuckDuckGo search
    
    Args:
        crop_name: Name of the crop (e.g., wheat, rice, cotton)
        city: City name for the mandi market
    """
    import requests
    from bs4 import BeautifulSoup
    
    logger.info(f"💰 Fetching prices for {crop_name} in {city}")
    
    # Crop name mapping for AMIS Pakistan
    crop_mapping = {
        "wheat": "گندم",
        "rice": "چاول",
        "cotton": "کپاس", 
        "maize": "مکئی",
        "sugarcane": "گنا",
        "potato": "آلو",
        "onion": "پیاز",
        "tomato": "ٹماٹر",
        "mango": "آم",
        "apple": "سیب",
        "banana": "کیلا",
        "chickpea": "چنا",
        "lentil": "مسور",
        "mustard": "سرسوں",
    }
    
    # City mapping for AMIS
    city_mapping = {
        "lahore": "لاہور",
        "karachi": "کراچی",
        "faisalabad": "فیصل آباد",
        "multan": "ملتان",
        "rawalpindi": "راولپنڈی",
        "peshawar": "پشاور",
        "quetta": "کوئٹہ",
        "hyderabad": "حیدرآباد",
        "gujranwala": "گوجرانوالہ",
        "sialkot": "سیالکوٹ",
    }
    
    results = []
    
    # Try AMIS Pakistan website scraping
    try:
        # AMIS Pakistan daily prices page
        amis_url = "http://amis.pk/Agristatistics/DailyPrices.aspx"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(amis_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for price tables
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_text = ' '.join([cell.get_text(strip=True) for cell in cells]).lower()
                    
                    # Check if this row contains our crop
                    crop_lower = crop_name.lower()
                    crop_urdu = crop_mapping.get(crop_lower, "")
                    city_lower = city.lower()
                    city_urdu = city_mapping.get(city_lower, "")
                    
                    if (crop_lower in row_text or crop_urdu in row_text):
                        results.append(f"AMIS Data: {' | '.join([cell.get_text(strip=True) for cell in cells])}")
            
            if results:
                logger.info(f"✅ Found {len(results)} price records from AMIS")
    
    except Exception as e:
        logger.warning(f"AMIS scraping failed: {e}")
    
    # Also try the agriculture department website
    try:
        agri_url = f"http://www.amis.pk/PriceSearch.aspx"
        # This would need form submission for specific searches
    except Exception:
        pass
    
    # Fallback to DuckDuckGo search with Pakistan-specific query
    try:
        search = DuckDuckGoSearchRun()
        current_date = datetime.now().strftime("%B %Y")
        
        # More specific Pakistan-focused queries
        queries = [
            f"{crop_name} mandi price {city} pakistan {current_date} PKR per maund",
            f"{crop_name} rate {city} pakistan market today",
        ]
        
        for query in queries:
            try:
                search_result = search.invoke(query)
                if search_result and "price" in search_result.lower():
                    results.append(f"Search Result: {search_result}")
                    break
            except Exception:
                continue
    
    except Exception as e:
        logger.warning(f"Search fallback failed: {e}")
    
    if results:
        # Format the results nicely
        formatted = f"""
                📊 **Market Prices for {crop_name.title()} in {city.title()}**
                Source: AMIS Pakistan (amis.pk) & Market Search

                {chr(10).join(results[:3])}

                ⚠️ Note: Prices may vary. Visit your local mandi for exact rates.
                Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """
        return formatted
    
    return f"""
            ❌ Could not find exact prices for {crop_name} in {city}.

            💡 Suggestions:
            1. Visit http://amis.pk for official Pakistan agriculture prices
            2. Check your local mandi directly
            3. Try different crop/city combinations

            Common mandis in Pakistan:
            - Lahore, Multan, Faisalabad (Punjab)
            - Karachi, Hyderabad (Sindh)
            - Peshawar (KPK)
            - Quetta (Balochistan)
            """