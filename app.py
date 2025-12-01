"""
KisaanLink - AI-Powered Agricultural Assistant
==============================================

This is the main Streamlit frontend application for KisaanLink.

Architecture Overview:
- Uses a two-step flow for image processing:
  1. Image → analyze_image_standalone() → Diagnosis
  2. Diagnosis + User Query → LangGraph Agent → Response
- Session state manages conversation memory and pending uploads
- Four tabs: Chat, Satellite, Weather, Prices

Design Decisions:
- pending_image stored in session_state to survive Streamlit reruns
- uploader_key counter resets file uploader after each submission (one-time use)
- last_diagnosis enables follow-up questions without re-uploading

Author: KisaanLink Team
"""

import streamlit as st
import uuid
import os
import tempfile
import base64
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage, AIMessage
from graph import app, run_agent, DEFAULT_LOCATION, get_supported_languages, get_default_hyperparams, analyze_image_standalone
from tools import get_satellite_image, get_crop_health_ndvi, get_agri_weather
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def extract_text_content(message):
    """
    Extract text content from a LangChain message.
    Handles various content formats:
    - String content
    - List of content blocks (multimodal)
    - AIMessage with tool_calls
    """
    if message is None:
        return ""
    
    # Get the content attribute
    content = getattr(message, 'content', message)
    
    # If it's already a string, return it
    if isinstance(content, str):
        return content
    
    # If it's a list (multimodal content), extract text parts
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        return '\n'.join(text_parts) if text_parts else str(content)
    
    # If it's a dict, try to get text or convert to string
    if isinstance(content, dict):
        if 'text' in content:
            return content['text']
        return str(content)
    
    # Fallback: convert to string
    return str(content) if content else ""


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="KisaanLink - AI Agronomist", 
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load supported languages and default hyperparameters from graph.py
LANGUAGES = get_supported_languages()
DEFAULTS = get_default_hyperparams()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Each session state variable serves a specific purpose:

# thread_id: Unique identifier for LangGraph memory/checkpointing
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# chat_history: Displayed messages in the chat UI (includes images)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# conversation_messages: Text-only messages for context building
if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = []

# language: User's selected language for responses
if "language" not in st.session_state:
    st.session_state.language = "english"

# history_days: Number of days for weather/satellite data analysis
if "history_days" not in st.session_state:
    st.session_state.history_days = DEFAULTS["history_days"]

# temperature: LLM creativity parameter (0 = deterministic, 1 = creative)
if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULTS["temperature"]

# Cached data from external APIs (avoid re-fetching)
if "satellite_data" not in st.session_state:
    st.session_state.satellite_data = None
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "ndvi_data" not in st.session_state:
    st.session_state.ndvi_data = None

# Diagnosis memory: Allows follow-up questions without re-uploading image
if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None
if "last_image_description" not in st.session_state:
    st.session_state.last_image_description = None

# Image upload handling:
# - uploader_key: Incremented after each submission to reset the file uploader
# - pending_image: Stores image bytes to survive Streamlit reruns
# - pending_image_name: Original filename for extension detection
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "pending_image_name" not in st.session_state:
    st.session_state.pending_image_name = None

# LangGraph configuration for memory/checkpointing
thread_id = st.session_state.thread_id
config = {"configurable": {"thread_id": thread_id}}

# =============================================================================
# SIDEBAR - User Settings & Configuration
# =============================================================================
st.sidebar.title("⚙️ Settings")

# Language Selection - Supports 5 regional languages
st.sidebar.header("🌐 Language / زبان")
language_options = {key: info["name"] for key, info in LANGUAGES.items()}
selected_language = st.sidebar.selectbox(
    "Select Response Language",
    options=list(language_options.keys()),
    format_func=lambda x: language_options[x],
    index=list(language_options.keys()).index(st.session_state.language),
    key="lang_select"
)
st.session_state.language = selected_language
greeting = LANGUAGES[selected_language]["greeting"]
st.sidebar.info(f"💬 {greeting}")

# Analysis Settings - Control data fetching behavior
st.sidebar.header("🎛️ Analysis Settings")
history_days = st.sidebar.slider("📅 Days of History", 7, 90, st.session_state.history_days, 7)
st.session_state.history_days = history_days

temperature = st.sidebar.slider("🌡️ AI Creativity", 0.0, 1.0, st.session_state.temperature, 0.1)
st.session_state.temperature = temperature

# Inform user about intelligent data fetching
st.sidebar.info("💡 The AI agent automatically decides when to fetch weather, satellite, or market data based on your question.")

# Location Settings - For satellite imagery and weather data
st.sidebar.header("📍 Location")
use_custom_location = st.sidebar.checkbox("Use Custom Location", value=False)
if use_custom_location:
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, DEFAULT_LOCATION["lat"], format="%.4f")
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, DEFAULT_LOCATION["lon"], format="%.4f")
    location = {"lat": lat, "lon": lon}
else:
    location = DEFAULT_LOCATION
    st.sidebar.caption(f"📍 Default: Lahore ({DEFAULT_LOCATION['lat']:.2f}, {DEFAULT_LOCATION['lon']:.2f})")

# Crop & Market Settings - Used for price lookups
st.sidebar.header("🌿 Crop & Market")
crop_name = st.sidebar.text_input("Crop Name", placeholder="e.g., Wheat, Rice")
city = st.sidebar.text_input("City/Mandi", placeholder="e.g., Lahore, Multan")

# Show stored diagnosis info if available (enables follow-up questions)
if st.session_state.last_diagnosis:
    st.sidebar.header("🔬 Last Diagnosis")
    st.sidebar.success("✅ Diagnosis stored in memory")
    if st.sidebar.button("Clear Diagnosis Memory"):
        st.session_state.last_diagnosis = None
        st.session_state.last_image_description = None
        st.rerun()

# Reset button - Clears all session state and starts fresh
if st.sidebar.button("🗑️ Clear All & Reset"):
    st.session_state.chat_history = []
    st.session_state.conversation_messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.satellite_data = None
    st.session_state.weather_data = None
    st.session_state.ndvi_data = None
    st.session_state.last_diagnosis = None
    st.session_state.last_image_description = None
    st.rerun()

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================
st.title("🌾 KisaanLink: AI Agronomist")
st.markdown("Powered by **Gemini** & **Google Earth Engine**")

# Four main tabs for different functionalities
tab_chat, tab_satellite, tab_weather, tab_prices = st.tabs([
    "💬 Chat", "🛰️ Satellite View", "🌤️ Weather", "💰 Prices"
])

# =============================================================================
# TAB 1: CHAT - Main conversational interface
# =============================================================================
with tab_chat:
    # Display existing chat history with images
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image"):
                st.image(message["image"], caption="Uploaded Crop Image", width=300)

    # Image upload widget
    # Design: Uses dynamic key to reset after each submission (one-time image use)
    uploaded_file = st.file_uploader(
        "📷 Upload crop image for disease diagnosis", 
        type=['jpg', 'png', 'jpeg', 'webp'],
        key=f"chat_image_upload_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )
    
    # CRITICAL: Store uploaded file bytes immediately in session state
    # This ensures the image survives Streamlit's rerun cycle when user presses Enter
    if uploaded_file is not None:
        st.session_state.pending_image = uploaded_file.getvalue()
        st.session_state.pending_image_name = uploaded_file.name
    
    # Show preview of pending image
    if st.session_state.pending_image:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(st.session_state.pending_image, caption="Ready to analyze", width=150)
        with col2:
            st.info(f"📷 **{st.session_state.pending_image_name}** ready. Type your question below and press Enter.")

    # Chat input - Primary interaction method
    user_query = st.chat_input("Ask about farming, weather, prices, or upload an image above...")

    # Placeholder for future auto-analyze feature
    if st.session_state.pending_image and not user_query:
        pass  # Will be handled when user types something
    
    
    # ==========================================================================
    # MAIN PROCESSING LOGIC - Two-Step Image Flow
    # ==========================================================================
    if user_query or (st.session_state.pending_image and st.session_state.get("auto_analyze", False)):
        image_path = None
        image_to_show = None
        image_analysis = None
        
        # Handle uploaded image FROM SESSION STATE (not from uploader directly)
        if st.session_state.pending_image:
            file_ext = os.path.splitext(st.session_state.pending_image_name)[1] or ".jpg"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_file.write(st.session_state.pending_image)
            temp_file.close()
            image_path = temp_file.name
            image_to_show = st.session_state.pending_image
            
            # If no query provided, use default
            if not user_query:
                user_query = "Please analyze this crop image for diseases and provide treatment advice."

        # Add user message to chat history
        user_chat_entry = {"role": "user", "content": user_query}
        if image_to_show:
            user_chat_entry["image"] = image_to_show
        st.session_state.chat_history.append(user_chat_entry)

        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(user_query)
            if image_to_show:
                st.image(image_to_show, caption="Uploaded Crop Image", width=300)

        # Run Agent with two-step flow for images
        with st.chat_message("assistant"):
            final_response = ""
            try:
                # STEP 1: If image exists, analyze it first (outside the graph)
                if image_path and os.path.exists(image_path):
                    with st.spinner("🔬 Analyzing image..."):
                        image_analysis = analyze_image_standalone(
                            image_path=image_path,
                            language=st.session_state.language,
                            temperature=st.session_state.temperature
                        )
                        if image_analysis:
                            st.session_state.last_diagnosis = image_analysis
                            st.success("✅ Image analyzed!")
                            with st.expander("📋 Image Analysis Result", expanded=False):
                                st.markdown(image_analysis)
                
                # STEP 2: Send image analysis + user prompt to main agent
                with st.spinner("Processing your request..."):
                    # Build conversation context
                    conversation_context = ""
                    if st.session_state.conversation_messages:
                        context_items = []
                        for msg in st.session_state.conversation_messages[-6:]:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
                            context_items.append(f"{role}: {content}")
                        conversation_context = "\n".join(context_items)
                    
                    # Build the message with image context if available
                    if image_analysis:
                        # Include the fresh image analysis in the prompt
                        message_content = f"""IMAGE ANALYSIS (just performed):
{image_analysis}

USER'S QUESTION: {user_query}

Based on the image analysis above, please answer the user's question and provide relevant treatment/advice if applicable."""
                    elif st.session_state.last_diagnosis:
                        # Use stored diagnosis from memory
                        message_content = f"""PREVIOUS IMAGE ANALYSIS (from memory):
{st.session_state.last_diagnosis}

USER'S QUESTION: {user_query}

Based on the previous analysis, please answer the user's question."""
                    else:
                        # No image context
                        message_content = user_query
                        if conversation_context:
                            message_content = f"CONVERSATION HISTORY:\n{conversation_context}\n\nCURRENT QUERY: {user_query}"
                    
                    st.session_state.conversation_messages.append({"role": "user", "content": user_query})
                    
                    # Prepare inputs for main agent (NO image_path - already analyzed)
                    inputs = {
                        "messages": [HumanMessage(content=message_content)],
                        "image_path": None,  # Image already analyzed in step 1
                        "diagnosis": image_analysis or st.session_state.last_diagnosis,
                        "location": location,
                        "crop_name": crop_name if crop_name else None,
                        "city": city if city else None,
                        "language": st.session_state.language,
                        "history_days": st.session_state.history_days,
                        "temperature": st.session_state.temperature,
                    }
                    
                    agent_steps = []
                    for event in app.stream(inputs, config=config):
                        for key, value in event.items():
                            agent_steps.append(f"✅ {key.replace('_', ' ').title()}")
                            if "messages" in value and value["messages"]:
                                msg = value["messages"][-1]
                                # Skip tool-call-only messages (no actual text content)
                                if hasattr(msg, 'tool_calls') and msg.tool_calls and not getattr(msg, 'content', ''):
                                    continue
                                # Extract text content properly (handles multimodal/tool responses)
                                extracted = extract_text_content(msg)
                                if extracted and extracted.strip():
                                    final_response = extracted
                            if "satellite_data" in value and value["satellite_data"]:
                                st.session_state.satellite_data = value["satellite_data"]
                    
                    if agent_steps:
                        st.caption(f"🔄 {' → '.join(agent_steps)}")
                
                if final_response:
                    st.markdown(final_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                    st.session_state.conversation_messages.append({"role": "assistant", "content": final_response})
                else:
                    st.warning("No response generated.")
                    
            except Exception as e:
                import traceback
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

        # Cleanup temp file
        if image_path and os.path.exists(image_path):
            try:
                os.unlink(image_path)
            except:
                pass
        
        # Clear pending image from session state (one-time use)
        st.session_state.pending_image = None
        st.session_state.pending_image_name = None
        
        # Increment uploader key to clear the file uploader for next prompt
        st.session_state.uploader_key += 1
        
        st.rerun()

# TAB 2: SATELLITE
with tab_satellite:
    st.header("🛰️ Satellite Imagery & Crop Health")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔄 Fetch High-Res Satellite Data"):
            with st.spinner("Fetching from Earth Engine..."):
                try:
                    st.session_state.satellite_data = get_satellite_image.invoke({"lat": location["lat"], "lon": location["lon"]})
                    st.session_state.ndvi_data = get_crop_health_ndvi.invoke({"lat": location["lat"], "lon": location["lon"], "days": st.session_state.history_days})
                    st.success("✅ Data fetched!")
                except Exception as e:
                    st.error(f"Error: {e}")
    with col2:
        st.metric("📍 Location", f"{location['lat']:.4f}, {location['lon']:.4f}")

    if st.session_state.satellite_data:
        sat_data = st.session_state.satellite_data
        if isinstance(sat_data, dict) and sat_data.get("image_url"):
            st.subheader("Sentinel-2 False Color (10m Resolution)")
            st.image(sat_data["image_url"], caption=f"Date: {sat_data.get('image_date', 'N/A')}", use_container_width=True)
            st.caption("🔴 Bright Red = Healthy Vegetation | 🟤 Brown/Gray = Bare Soil or Stressed Crops")

    if st.session_state.ndvi_data:
        ndvi = st.session_state.ndvi_data
        st.subheader("🌱 NDVI Crop Health Index")
        if isinstance(ndvi, dict) and ndvi.get("ndvi_value"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NDVI Value", f"{ndvi['ndvi_value']:.3f}")
            with col2:
                st.metric("Health Status", ndvi.get("health_status", "Unknown"))
            with col3:
                st.metric("Analysis Period", ndvi.get("period", "N/A"))
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=ndvi["ndvi_value"],
                title={'text': "Vegetation Health Index"},
                gauge={'axis': {'range': [-0.2, 1]}, 'bar': {'color': "darkgreen"},
                       'steps': [{'range': [-0.2, 0.2], 'color': "#8B4513"},
                                 {'range': [0.2, 0.4], 'color': "#FFD700"},
                                 {'range': [0.4, 0.6], 'color': "#90EE90"},
                                 {'range': [0.6, 1], 'color': "#228B22"}]}
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"📋 **Interpretation:** {ndvi.get('interpretation', 'N/A')}")

# TAB 3: WEATHER
with tab_weather:
    st.header("🌤️ Agricultural Weather Dashboard")
    if st.button("🔄 Fetch Weather Data"):
        with st.spinner("Fetching ERA5 climate data..."):
            try:
                st.session_state.weather_data = get_agri_weather.invoke({"lat": location["lat"], "lon": location["lon"], "days": st.session_state.history_days})
                st.success("✅ Weather data fetched!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.weather_data:
        weather = st.session_state.weather_data
        if isinstance(weather, dict) and not weather.get("error"):
            current = weather.get("current_conditions", {})
            averages = weather.get("period_averages", {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                temp = current.get("temperature_celsius") or averages.get("avg_temperature_celsius")
                st.metric("🌡️ Temperature", f"{temp}°C" if temp else "N/A")
            with col2:
                st.metric("🌧️ Precipitation", f"{averages.get('total_precipitation_mm')} mm" if averages.get('total_precipitation_mm') else "N/A")
            with col3:
                st.metric("💧 Soil Moisture", f"{averages.get('soil_moisture_percent')}%" if averages.get('soil_moisture_percent') else "N/A")
            with col4:
                st.metric("📅 Period", weather.get("period", "N/A"))
            if weather.get("agricultural_advisory"):
                st.info(f"📋 **Advisory:** {weather['agricultural_advisory']}")
    else:
        st.info("Click 'Fetch Weather Data' to load agricultural weather for your location.")

# TAB 4: PRICES
with tab_prices:
    st.header("💰 Market Prices (Mandi Rates)")
    st.markdown("**Sources:** [AMIS Pakistan](http://amis.pk) & Web Search")
    col1, col2 = st.columns(2)
    with col1:
        price_crop = st.text_input("Crop", value=crop_name or "Wheat", key="price_crop")
    with col2:
        price_city = st.text_input("City/Mandi", value=city or "Lahore", key="price_city")
    if st.button("🔍 Check Current Prices"):
        with st.spinner(f"Searching prices for {price_crop} in {price_city}..."):
            from tools import check_mandi_prices
            result = check_mandi_prices.invoke({"crop_name": price_crop, "city": price_city})
            st.markdown("### Results")
            st.markdown(result)
    st.markdown("---")
    # st.info("💡 For official rates, visit [AMIS Pakistan](http://amis.pk)")

# DEBUG
with st.sidebar.expander("🔧 Debug Info"):
    st.json({
        "thread_id": thread_id[:8] + "...",
        "language": st.session_state.language,
        "history_days": st.session_state.history_days,
        "chat_count": len(st.session_state.chat_history),
        "has_diagnosis": st.session_state.last_diagnosis is not None,
    })
