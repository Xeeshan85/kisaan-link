import operator
import os
import mimetypes
import logging
from typing import Annotated, List, TypedDict, Union, Optional, Literal
from dotenv import load_dotenv

# Load environment variables BEFORE importing langchain modules
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from tools import get_agri_weather, check_mandi_prices, get_satellite_image, get_crop_health_ndvi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
DEFAULT_LOCATION = {
    "lat": float(os.getenv("DEFAULT_LAT", "31.5204")),  # Default: Lahore
    "lon": float(os.getenv("DEFAULT_LON", "74.3587"))
}

# Supported languages with their prompts
SUPPORTED_LANGUAGES = {
    "english": {
        "name": "English",
        "instruction": "Respond in English. Use simple, farmer-friendly language.",
        "greeting": "Hello! I'm KisaanLink, your AI farming assistant."
    },
    "urdu": {
        "name": "اردو (Urdu)", 
        "instruction": "اردو میں جواب دیں۔ سادہ زبان استعمال کریں جو کسانوں کے لیے آسان ہو۔",
        "greeting": "السلام علیکم! میں کسان لنک ہوں، آپ کا AI زرعی معاون۔"
    },
    "punjabi": {
        "name": "پنجابی (Punjabi)",
        "instruction": "پنجابی وچ جواب دیو۔ سادی بولی ورتو جیہڑی کساناں لئی سمجھنی سوکھی ہووے۔",
        "greeting": "سلام! میں کسان لنک آں، تہاڈا AI زرعی مددگار۔"
    },
    "sindhi": {
        "name": "سنڌي (Sindhi)",
        "instruction": "سنڌي ۾ جواب ڏيو۔ سادي ٻولي استعمال ڪريو جيڪا هارين لاءِ سمجھڻ ۾ آسان هجي۔",
        "greeting": "سلام! مان ڪسان لنڪ آهيان، توهان جو AI زرعي مددگار۔"
    },
    "hindi": {
        "name": "हिंदी (Hindi)",
        "instruction": "हिंदी में जवाब दें। सरल भाषा का उपयोग करें जो किसानों के लिए समझने में आसान हो।",
        "greeting": "नमस्ते! मैं किसान लिंक हूं, आपका AI कृषि सहायक।"
    }
}

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "history_days": 30,          # Days of satellite/weather history
    "max_context_messages": 10,   # Max messages to keep in context
    "temperature": 0.0,           # LLM temperature (0-1)
    "language": "english",        # Response language
    "include_satellite": True,    # Include satellite imagery in analysis
    "include_weather": True,      # Include weather data
    "include_ndvi": True          # Include NDVI crop health
}

# --- 1. State Definition ---
class AgentState(TypedDict):
    # Core state
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
    image_path: Optional[str]
    diagnosis: Optional[str]
    
    # Location & context
    location: Optional[dict]      # {"lat": float, "lon": float}
    crop_name: Optional[str]
    city: Optional[str]
    
    # Satellite & environmental data
    satellite_data: Optional[dict]
    
    # User preferences / Hyperparameters
    language: Optional[str]       # Response language
    history_days: Optional[int]   # Days of data to fetch
    temperature: Optional[float]  # LLM temperature
    include_satellite: Optional[bool]
    include_weather: Optional[bool]
    include_ndvi: Optional[bool]
    
    # Context management
    context_summary: Optional[str]  # Summary of older conversation


def _get_llm(state: AgentState) -> ChatGoogleGenerativeAI:
    """Get LLM instance with state-based temperature"""
    temp = state.get("temperature", DEFAULT_TEMPERATURE)
    return ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        temperature=temp,
        google_api_key=GOOGLE_API_KEY
    )


def _get_language_instruction(state: AgentState) -> str:
    """Get language instruction for the current state"""
    lang = state.get("language", "english").lower()
    if lang in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[lang]["instruction"]
    return SUPPORTED_LANGUAGES["english"]["instruction"]


def _manage_context(state: AgentState) -> List:
    """Manage conversation context to prevent token overflow"""
    messages = state.get("messages", [])
    max_messages = state.get("max_context_messages", DEFAULT_HYPERPARAMS["max_context_messages"])
    
    if len(messages) <= max_messages:
        return messages
    
    # Keep the most recent messages, summarize older ones
    recent_messages = messages[-max_messages:]
    
    # Add context summary if available
    if state.get("context_summary"):
        summary_msg = SystemMessage(content=f"Previous conversation summary: {state['context_summary']}")
        return [summary_msg] + recent_messages
    
    return recent_messages


# --- 2. Initialize Default Model ---
llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL, 
    temperature=DEFAULT_TEMPERATURE,
    google_api_key=GOOGLE_API_KEY
)

# --- 3. Node Definitions ---
def router_node(state: AgentState):
    """Uses LLM to intelligently route the user request"""
    messages = state['messages']
    last_message = messages[-1].content
    
    # Check for image first (highest priority)
    if state.get('image_path'):
        return "pathologist"
    
    # Check for settings/configuration requests
    settings_keywords = ["settings", "language", "days", "configure", "change", "set"]
    if any(kw in last_message.lower() for kw in settings_keywords):
        return "settings_handler"
    
    # Use LLM for intelligent classification
    classification_prompt = f"""
    Classify the following farmer's query into exactly ONE category:
    - "price_check": If asking about market prices, rates, mandi prices, selling crops
    - "satellite": If asking about satellite imagery, crop health from space, NDVI, field analysis
    - "weather": If asking about weather, rain, temperature, soil moisture
    - "general": For general questions, greetings, or anything else
    
    Query: "{last_message}"
    
    Respond with only the category name, nothing else.
    """
    
    try:
        current_llm = _get_llm(state)
        response = current_llm.invoke([HumanMessage(content=classification_prompt)])
        category = response.content.strip().lower()
        
        if "price" in category:
            return "market_analyst"
        elif "satellite" in category or "ndvi" in category:
            return "satellite_analyst"
        elif "weather" in category:
            return "weather_analyst"
        else:
            return "general_chat"
    except Exception:
        # Fallback to keyword matching if LLM fails
        last_lower = last_message.lower()
        if any(keyword in last_lower for keyword in ["price", "rate", "mandi", "sell", "market"]):
            return "market_analyst"
        elif any(keyword in last_lower for keyword in ["satellite", "ndvi", "field", "imagery"]):
            return "satellite_analyst"
        elif any(keyword in last_lower for keyword in ["weather", "rain", "temperature", "moisture"]):
            return "weather_analyst"
        return "general_chat"


def settings_handler(state: AgentState):
    """Handle user requests to change settings via natural language"""
    last_message = state['messages'][-1].content.lower()
    lang_instruction = _get_language_instruction(state)
    current_llm = _get_llm(state)
    
    # Parse settings from message
    updates = {}
    response_parts = []
    
    # Language detection
    for lang_key, lang_info in SUPPORTED_LANGUAGES.items():
        if lang_key in last_message or lang_info["name"].lower() in last_message:
            updates["language"] = lang_key
            response_parts.append(f"Language set to {lang_info['name']}")
    
    # Days detection
    import re
    days_match = re.search(r'(\d+)\s*days?', last_message)
    if days_match:
        days = int(days_match.group(1))
        if 1 <= days <= 90:
            updates["history_days"] = days
            response_parts.append(f"History period set to {days} days")
    
    # Temperature detection
    temp_match = re.search(r'temperature[:\s]+(\d*\.?\d+)', last_message)
    if temp_match:
        temp = float(temp_match.group(1))
        if 0 <= temp <= 1:
            updates["temperature"] = temp
            response_parts.append(f"LLM temperature set to {temp}")
    
    if response_parts:
        response = "✅ Settings updated:\n- " + "\n- ".join(response_parts)
    else:
        # Show current settings
        lang = state.get("language", "english")
        days = state.get("history_days", DEFAULT_HYPERPARAMS["history_days"])
        temp = state.get("temperature", DEFAULT_HYPERPARAMS["temperature"])
        
        response = f"""📋 **Current Settings:**
- Language: {SUPPORTED_LANGUAGES.get(lang, {}).get('name', lang)}
- History Days: {days}
- LLM Temperature: {temp}

To change settings, say things like:
- "Change language to Urdu"
- "Set history to 60 days"
- "Set temperature to 0.5"
"""
    
    return {"messages": [AIMessage(content=response)], **updates}


def satellite_analyst(state: AgentState):
    """Dedicated satellite/NDVI analysis agent - can analyze any location mentioned"""
    last_message = state['messages'][-1].content
    lang_instruction = _get_language_instruction(state)
    current_llm = _get_llm(state)
    lang = state.get("language", "english")
    lang_name = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["english"])["name"]
    days = state.get("history_days", DEFAULT_HYPERPARAMS["history_days"])
    
    # Try to extract location from user query
    location_prompt = f"""Extract the location (city/place name) from this query. 
If a specific place is mentioned, provide its approximate latitude and longitude.
If no specific location is mentioned, respond with "DEFAULT".

Query: "{last_message}"

Respond in this exact format:
LOCATION: <place name or DEFAULT>
LAT: <latitude or DEFAULT>
LON: <longitude or DEFAULT>
"""
    
    try:
        loc_response = current_llm.invoke([HumanMessage(content=location_prompt)])
        loc_text = loc_response.content.strip()
        
        location_name = "your field"
        lat = DEFAULT_LOCATION["lat"]
        lon = DEFAULT_LOCATION["lon"]
        
        for line in loc_text.split('\n'):
            if line.startswith('LOCATION:'):
                loc_val = line.replace('LOCATION:', '').strip()
                if loc_val and loc_val.upper() != "DEFAULT":
                    location_name = loc_val
            elif line.startswith('LAT:'):
                lat_val = line.replace('LAT:', '').strip()
                if lat_val and lat_val.upper() != "DEFAULT":
                    try:
                        lat = float(lat_val)
                    except:
                        pass
            elif line.startswith('LON:'):
                lon_val = line.replace('LON:', '').strip()
                if lon_val and lon_val.upper() != "DEFAULT":
                    try:
                        lon = float(lon_val)
                    except:
                        pass
        
        if location_name == "your field":
            location = state.get("location") or DEFAULT_LOCATION
            lat = location.get("lat", DEFAULT_LOCATION["lat"])
            lon = location.get("lon", DEFAULT_LOCATION["lon"])
        
        logger.info(f"Satellite analysis for: {location_name} ({lat}, {lon})")
        
        results = {}
        
        if state.get("include_satellite", True):
            results["satellite"] = get_satellite_image.invoke({"lat": lat, "lon": lon})
        
        if state.get("include_ndvi", True):
            results["ndvi"] = get_crop_health_ndvi.invoke({"lat": lat, "lon": lon, "days": days})
        
        prompt = f"""**CRITICAL**: You MUST respond ONLY in {lang_name}.
{lang_instruction}

You are a Remote Sensing Expert analyzing satellite data.

User Query: {last_message}
Location: {location_name} (Lat: {lat}, Lon: {lon})

SATELLITE DATA:
{results}

Provide a clear analysis including:
1. Overall field/area health assessment for {location_name}
2. Any areas of concern (stress, disease indicators)
3. Recommendations based on the satellite analysis

Remember: Respond ONLY in {lang_name}."""
        response = current_llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response], "satellite_data": results}
    
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error in satellite analysis: {str(e)}")]}


def weather_analyst(state: AgentState):
    """Dedicated weather analysis agent - can fetch weather for any location mentioned"""
    last_message = state['messages'][-1].content
    lang_instruction = _get_language_instruction(state)
    current_llm = _get_llm(state)
    lang = state.get("language", "english")
    lang_name = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["english"])["name"]
    days = state.get("history_days", DEFAULT_HYPERPARAMS["history_days"])
    
    # Try to extract location from user query using LLM
    location_prompt = f"""Extract the location (city/place name) from this query. 
If a specific place is mentioned, provide its approximate latitude and longitude.
If no specific location is mentioned, respond with "DEFAULT".

Query: "{last_message}"

Respond in this exact format (nothing else):
LOCATION: <place name or DEFAULT>
LAT: <latitude as decimal number or DEFAULT>
LON: <longitude as decimal number or DEFAULT>
"""
    
    try:
        loc_response = current_llm.invoke([HumanMessage(content=location_prompt)])
        loc_text = loc_response.content.strip()
        
        # Parse location response
        location_name = "your location"
        lat = DEFAULT_LOCATION["lat"]
        lon = DEFAULT_LOCATION["lon"]
        
        for line in loc_text.split('\n'):
            if line.startswith('LOCATION:'):
                loc_val = line.replace('LOCATION:', '').strip()
                if loc_val and loc_val.upper() != "DEFAULT":
                    location_name = loc_val
            elif line.startswith('LAT:'):
                lat_val = line.replace('LAT:', '').strip()
                if lat_val and lat_val.upper() != "DEFAULT":
                    try:
                        lat = float(lat_val)
                    except:
                        pass
            elif line.startswith('LON:'):
                lon_val = line.replace('LON:', '').strip()
                if lon_val and lon_val.upper() != "DEFAULT":
                    try:
                        lon = float(lon_val)
                    except:
                        pass
        
        # If no location extracted from query, use state location
        if location_name == "your location":
            location = state.get("location") or DEFAULT_LOCATION
            lat = location.get("lat", DEFAULT_LOCATION["lat"])
            lon = location.get("lon", DEFAULT_LOCATION["lon"])
        
        logger.info(f"Weather query for: {location_name} ({lat}, {lon})")
        
        # Fetch weather data
        weather_data = get_agri_weather.invoke({"lat": lat, "lon": lon, "days": days})
        
        prompt = f"""**CRITICAL**: You MUST respond ONLY in {lang_name}. 
{lang_instruction}

You are an Agricultural Meteorologist providing weather advice.

User Query: {last_message}
Location: {location_name} (Lat: {lat}, Lon: {lon})

WEATHER DATA (Last {days} days from ERA5 Climate Reanalysis):
{weather_data}

Provide practical advice including:
1. Current weather conditions summary for {location_name}
2. Impact on farming activities
3. Recommendations (irrigation, spraying timing, etc.)

Remember: Respond ONLY in {lang_name}."""
        
        response = current_llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}
    
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error fetching weather data: {str(e)}")]}


def _get_mime_type(file_path: str) -> str:
    """Detect MIME type from file extension"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "image/jpeg"


def pathologist_agent(state: AgentState):
    """Vision Agent: Analyzes the crop image"""
    import base64
    
    lang_instruction = _get_language_instruction(state)
    current_llm = _get_llm(state)
    
    image_path = state.get('image_path')
    if not image_path:
        return {"messages": [AIMessage(content="No image provided for analysis.")]}
    
    if not os.path.exists(image_path):
        return {"messages": [AIMessage(content=f"Image file not found: {image_path}")]}
    
    try:
        # Load image with dynamic MIME type detection
        mime_type = _get_mime_type(image_path)
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        analysis_prompt = f"""{lang_instruction}

Analyze this crop image carefully. Provide:
1. Crop Name: Identify the crop species
2. Health Status: Healthy or Diseased
3. Disease Identification: If diseased, name the specific disease
4. Confidence Level: Your confidence in this diagnosis (High/Medium/Low)
5. Visual Symptoms: List the visible symptoms that led to your diagnosis

Be specific and accurate as this will guide treatment recommendations."""
            
        msg = HumanMessage(
            content=[
                {"type": "text", "text": analysis_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
            ]
        )
        response = current_llm.invoke([msg])
        return {"diagnosis": response.content, "messages": [response]}
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        return {"diagnosis": error_msg, "messages": [AIMessage(content=error_msg)]}


def _extract_crop_and_city(message: str, state: AgentState) -> tuple:
    """Use LLM to extract crop name and city from user message"""
    # Check if already provided in state
    crop = state.get('crop_name')
    city = state.get('city')
    
    if crop and city:
        return crop, city
    
    extraction_prompt = f"""
    Extract the crop name and city from this farmer's query.
    If not mentioned, use reasonable defaults for Pakistan agriculture.
    
    Query: "{message}"
    
    Respond in this exact format (no other text):
    CROP: <crop_name>
    CITY: <city_name>
    """
    
    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        lines = response.content.strip().split('\n')
        
        extracted_crop = None
        extracted_city = None
        
        for line in lines:
            if line.startswith('CROP:'):
                extracted_crop = line.replace('CROP:', '').strip()
            elif line.startswith('CITY:'):
                extracted_city = line.replace('CITY:', '').strip()
        
        # Use extracted values or fall back to state/defaults
        final_crop = crop or extracted_crop or "Wheat"
        final_city = city or extracted_city or "Lahore"
        
        return final_crop, final_city
    except Exception:
        return crop or "Wheat", city or "Lahore"


def market_analyst_agent(state: AgentState):
    """Tool Agent: Checks prices with dynamic crop/city extraction"""
    last_message = state['messages'][-1].content
    lang_instruction = _get_language_instruction(state)
    current_llm = _get_llm(state)
    lang = state.get("language", "english")
    lang_name = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["english"])["name"]
    
    try:
        # Extract crop and city from message or state
        crop_name, city = _extract_crop_and_city(last_message, state)
        
        # Use the tool with extracted values
        price_info = check_mandi_prices.invoke({"crop_name": crop_name, "city": city})
        
        response = current_llm.invoke([
            SystemMessage(content=f"""**CRITICAL LANGUAGE INSTRUCTION**: 
{lang_instruction}
You MUST respond ONLY in {lang_name}. Do not use any other language.

You are a Market Analyst helping Pakistani farmers.
Summarize the price information clearly and provide:
1. Current market price range
2. Price trend (if available)
3. Best time/place to sell (if inferable)

Keep the language simple and farmer-friendly.
Remember: ALWAYS respond in {lang_name}."""),
            HumanMessage(content=f"User asked: {last_message}\nCrop: {crop_name}, City: {city}\nMarket data: {price_info}")
        ])
        return {"messages": [response], "crop_name": crop_name, "city": city}
    except Exception as e:
        error_msg = f"Error fetching market prices: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}


def treatment_advisor(state: AgentState):
    """Integration Agent: Combines Diagnosis + Earth Engine Satellite Data"""
    diagnosis = state.get("diagnosis")
    lang_instruction = _get_language_instruction(state)
    current_llm = _get_llm(state)
    days = state.get("history_days", DEFAULT_HYPERPARAMS["history_days"])
    
    if not diagnosis:
        return {"messages": [AIMessage(content="No diagnosis available to provide treatment advice.")]}
    
    try:
        # Use location from state or fall back to default
        location = state.get("location") or DEFAULT_LOCATION
        lat = location.get("lat", DEFAULT_LOCATION["lat"])
        lon = location.get("lon", DEFAULT_LOCATION["lon"])
        
        # Fetch Environmental Context from Earth Engine (based on user preferences)
        weather = None
        ndvi_data = None
        satellite = None
        satellite_image_url = None
        
        if state.get("include_weather", True):
            weather = get_agri_weather.invoke({"lat": lat, "lon": lon, "days": days})
        
        if state.get("include_ndvi", True):
            ndvi_data = get_crop_health_ndvi.invoke({"lat": lat, "lon": lon, "days": days})
        
        if state.get("include_satellite", True):
            satellite = get_satellite_image.invoke({"lat": lat, "lon": lon})
            if isinstance(satellite, dict) and satellite.get("image_url"):
                satellite_image_url = satellite.get("image_url")
        
        prompt = f"""
{lang_instruction}

You are an Agricultural Treatment Advisor for Pakistani farmers.

DIAGNOSIS REPORT (from crop image):
{diagnosis}

SATELLITE CROP HEALTH (NDVI from Google Earth Engine - Last {days} days):
{ndvi_data if ndvi_data else "Not requested"}

WEATHER CONDITIONS (ERA5 Climate Data - Last {days} days):
{weather if weather else "Not requested"}

Based on the above data, provide a comprehensive treatment plan:

1. **Immediate Actions**: What should the farmer do right now?
2. **Treatment Recommendations**: Specific pesticides/fungicides/fertilizers with local brand names if possible
3. **Weather Considerations**: 
   - If recent rainfall is high: WARN the farmer NOT to spray pesticides (they will wash away)
   - If soil moisture is low: Recommend irrigation schedule
   - If humidity is high: Warn about fungal spread risk
4. **Satellite Analysis**: Based on the NDVI health status, comment on field-level crop health
5. **Preventive Measures**: How to prevent recurrence
6. **When to Seek Expert Help**: Signs that indicate need for agricultural officer visit
"""
        # If we have satellite image, include it for visual analysis
        if satellite_image_url:
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": satellite_image_url}}
                ]
            )
        else:
            msg = HumanMessage(content=prompt)
            
        response = current_llm.invoke([msg])
        return {"messages": [response], "satellite_data": {"weather": weather, "ndvi": ndvi_data, "satellite": satellite}}
    except Exception as e:
        error_msg = f"Error generating treatment advice: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}


def general_chat_agent(state: AgentState):
    """Handles general queries and greetings"""
    last_message = state['messages'][-1].content
    lang_instruction = _get_language_instruction(state)
    current_llm = _get_llm(state)
    lang = state.get("language", "english")
    lang_info = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["english"])
    greeting = lang_info["greeting"]
    lang_name = lang_info["name"]
    
    # Get chat history from LangGraph memory for context
    config = {"configurable": {"thread_id": state.get("thread_id", "default")}}
    
    # Build context from previous messages in state
    previous_messages = state.get("messages", [])[:-1]  # All except current
    context_str = ""
    if previous_messages:
        context_str = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}..."
            if len(m.content) > 200 else f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in previous_messages[-5:]  # Last 5 messages for context
        ])
    
    try:
        # Only include greeting on first message
        is_first_message = not context_str
        greeting_line = f"Start with a brief greeting like: {greeting}" if is_first_message else "Do NOT include any greeting, jump straight to answering the question."
        
        system_prompt = f"""You are KisaanLink, an AI agricultural assistant for Pakistani farmers.

**CRITICAL LANGUAGE INSTRUCTION**: 
{lang_instruction}
You MUST respond ONLY in {lang_name}. Do not use any other language.

{greeting_line}

You can help with:
- 🌱 Crop disease diagnosis (if they upload an image)
- 💰 Market/Mandi prices (ask for crop name and city)
- 🛰️ Satellite-based crop health monitoring
- 🌤️ Weather and soil conditions
- 🌾 General farming advice

{"CONVERSATION HISTORY:" + chr(10) + context_str if context_str else ""}

Keep your response helpful and use simple language. ALWAYS respond in {lang_name}."""

        response = current_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message)
        ])
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}


# --- 4. Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("pathologist", pathologist_agent)
workflow.add_node("market_analyst", market_analyst_agent)
workflow.add_node("treatment_advisor", treatment_advisor)
workflow.add_node("general_chat", general_chat_agent)
workflow.add_node("settings_handler", settings_handler)
workflow.add_node("satellite_analyst", satellite_analyst)
workflow.add_node("weather_analyst", weather_analyst)

# Conditional Entry based on router logic
workflow.set_conditional_entry_point(
    router_node,
    {
        "pathologist": "pathologist",
        "market_analyst": "market_analyst",
        "general_chat": "general_chat",
        "settings_handler": "settings_handler",
        "satellite_analyst": "satellite_analyst",
        "weather_analyst": "weather_analyst"
    }
)

# Logic Flow
workflow.add_edge("pathologist", "treatment_advisor")  # Diagnosis -> Treatment
workflow.add_edge("treatment_advisor", END)
workflow.add_edge("market_analyst", END)
workflow.add_edge("general_chat", END)
workflow.add_edge("settings_handler", END)
workflow.add_edge("satellite_analyst", END)
workflow.add_edge("weather_analyst", END)

# Persistence (Memory)
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# --- 5. Helper function for running the app ---
def run_agent(
    user_message: str,
    image_path: Optional[str] = None,
    location: Optional[dict] = None,
    crop_name: Optional[str] = None,
    city: Optional[str] = None,
    thread_id: str = "default",
    # New hyperparameters
    language: str = "english",
    history_days: int = 30,
    temperature: float = 0.0,
    include_satellite: bool = True,
    include_weather: bool = True,
    include_ndvi: bool = True
) -> str:
    """
    Convenience function to run the agent with proper state initialization.
    
    Args:
        user_message: The user's query
        image_path: Optional path to crop image for diagnosis
        location: Optional dict with 'lat' and 'lon' keys
        crop_name: Optional crop name for price queries
        city: Optional city name for price queries
        thread_id: Session ID for conversation memory
        language: Response language (english, urdu, punjabi, sindhi, hindi)
        history_days: Days of satellite/weather history to fetch
        temperature: LLM temperature (0.0 - 1.0)
        include_satellite: Include satellite imagery analysis
        include_weather: Include weather data
        include_ndvi: Include NDVI crop health
    
    Returns:
        The agent's response as a string
    """
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "image_path": image_path,
        "diagnosis": None,
        "location": location,
        "crop_name": crop_name,
        "city": city,
        # Hyperparameters
        "language": language,
        "history_days": history_days,
        "temperature": temperature,
        "include_satellite": include_satellite,
        "include_weather": include_weather,
        "include_ndvi": include_ndvi,
        "satellite_data": None,
        "context_summary": None
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        result = app.invoke(initial_state, config)
        if result.get("messages"):
            return result["messages"][-1].content
        return "No response generated."
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Error running agent: {str(e)}"


# Export supported languages for frontend
def get_supported_languages():
    """Return list of supported languages for UI"""
    return SUPPORTED_LANGUAGES


def get_default_hyperparams():
    """Return default hyperparameters for UI"""
    return DEFAULT_HYPERPARAMS