import streamlit as st
import uuid
import os
import tempfile
import base64
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage
from graph import app, run_agent, DEFAULT_LOCATION, get_supported_languages, get_default_hyperparams, analyze_image_standalone
from tools import get_satellite_image, get_crop_health_ndvi, get_agri_weather
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="KisaanLink - AI Agronomist", 
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load supported languages and defaults
LANGUAGES = get_supported_languages()
DEFAULTS = get_default_hyperparams()

# Session State initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = []
if "language" not in st.session_state:
    st.session_state.language = "english"
if "history_days" not in st.session_state:
    st.session_state.history_days = DEFAULTS["history_days"]
if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULTS["temperature"]
if "satellite_data" not in st.session_state:
    st.session_state.satellite_data = None
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "ndvi_data" not in st.session_state:
    st.session_state.ndvi_data = None
# Store last diagnosis in memory so user doesn't need to re-upload
if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None
if "last_image_description" not in st.session_state:
    st.session_state.last_image_description = None
# Counter to reset file uploader after each use
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
# Store uploaded image in session state to persist across reruns
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "pending_image_name" not in st.session_state:
    st.session_state.pending_image_name = None

thread_id = st.session_state.thread_id
config = {"configurable": {"thread_id": thread_id}}

# SIDEBAR - Settings only (no image upload)
st.sidebar.title("⚙️ Settings")

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

st.sidebar.header("🎛️ Analysis Settings")
history_days = st.sidebar.slider("📅 Days of History", 7, 90, st.session_state.history_days, 7)
st.session_state.history_days = history_days

temperature = st.sidebar.slider("🌡️ AI Creativity", 0.0, 1.0, st.session_state.temperature, 0.1)
st.session_state.temperature = temperature

# Note: Data sources are now intelligently selected by the AI agent
st.sidebar.info("💡 The AI agent automatically decides when to fetch weather, satellite, or market data based on your question.")

st.sidebar.header("📍 Location")
use_custom_location = st.sidebar.checkbox("Use Custom Location", value=False)
if use_custom_location:
    lat = st.sidebar.number_input("Latitude", -90.0, 90.0, DEFAULT_LOCATION["lat"], format="%.4f")
    lon = st.sidebar.number_input("Longitude", -180.0, 180.0, DEFAULT_LOCATION["lon"], format="%.4f")
    location = {"lat": lat, "lon": lon}
else:
    location = DEFAULT_LOCATION
    st.sidebar.caption(f"📍 Default: Lahore ({DEFAULT_LOCATION['lat']:.2f}, {DEFAULT_LOCATION['lon']:.2f})")

st.sidebar.header("🌿 Crop & Market")
crop_name = st.sidebar.text_input("Crop Name", placeholder="e.g., Wheat, Rice")
city = st.sidebar.text_input("City/Mandi", placeholder="e.g., Lahore, Multan")

# Show stored diagnosis info if available
if st.session_state.last_diagnosis:
    st.sidebar.header("🔬 Last Diagnosis")
    st.sidebar.success("✅ Diagnosis stored in memory")
    if st.sidebar.button("Clear Diagnosis Memory"):
        st.session_state.last_diagnosis = None
        st.session_state.last_image_description = None
        st.rerun()

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

# MAIN AREA
st.title("🌾 KisaanLink: AI Agronomist")
st.markdown("Powered by **Gemini 2.5** & **Google Earth Engine**")

tab_chat, tab_satellite, tab_weather, tab_prices = st.tabs([
    "💬 Chat", "🛰️ Satellite View", "🌤️ Weather", "💰 Prices"
])

# TAB 1: CHAT
with tab_chat:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image"):
                st.image(message["image"], caption="Uploaded Crop Image", width=300)

    # Image upload - store in session state immediately to persist across reruns
    uploaded_file = st.file_uploader(
        "📷 Upload crop image for disease diagnosis", 
        type=['jpg', 'png', 'jpeg', 'webp'],
        key=f"chat_image_upload_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )
    
    # CRITICAL: Store uploaded file in session state immediately
    if uploaded_file is not None:
        # Read file bytes and store in session state
        st.session_state.pending_image = uploaded_file.getvalue()
        st.session_state.pending_image_name = uploaded_file.name
    
    # Show uploaded image preview (from session state)
    if st.session_state.pending_image:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(st.session_state.pending_image, caption="Ready to analyze", width=150)
        with col2:
            st.info(f"� **{st.session_state.pending_image_name}** ready. Type your question below and press Enter.")

    # Chat input at bottom
    user_query = st.chat_input("Ask about farming, weather, prices, or upload an image above...")

    # Auto-generate query if image uploaded but no text
    if st.session_state.pending_image and not user_query:
        pass  # Will be handled when user types something
    
    # Process when user submits a query OR has pending image with auto-analyze
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

        # Display user message
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
                    with st.spinner("🔬 Step 1: Analyzing image..."):
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
                with st.spinner("🤖 Step 2: Processing your request..."):
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
                                final_response = value["messages"][-1].content
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
