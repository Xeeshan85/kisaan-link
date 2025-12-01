"""
KisaanLink - LangGraph Agent Orchestration
==========================================

This module implements the AI agent using LangGraph's StateGraph pattern.

Architecture:
- Uses a ReAct-style agent where the LLM decides when to call tools
- Tool binding allows intelligent data fetching (only when relevant)
- Multi-language support for Pakistani regional languages
- Two-step image processing: standalone analysis + agent synthesis

Graph Structure:
    START → router_node → [pathologist | main_agent | settings_handler]
                              ↓              ↓               ↓
                     treatment_advisor   tools/END         END
                              ↓              ↓
                         tools/END      synthesize
                              ↓              ↓
                          synthesize       END
                              ↓
                             END

Design Decisions:
- router_node determines flow based on image_path presence
- Tools are bound to LLM (not hardcoded rules)
- MemorySaver enables conversation persistence via thread_id
- Language instructions are injected into all prompts

Author: KisaanLink Team
"""

import operator
import os
import mimetypes
import logging
import base64
from typing import Annotated, List, TypedDict, Union, Optional
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from tools import get_agri_weather, check_mandi_prices, get_satellite_image, get_crop_health_ndvi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Environment variables and defaults
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
DEFAULT_LOCATION = {
    "lat": float(os.getenv("DEFAULT_LAT", "31.5204")),  # Lahore, Pakistan
    "lon": float(os.getenv("DEFAULT_LON", "74.3587"))
}

# =============================================================================
# MULTI-LANGUAGE SUPPORT - 5 Pakistani regional languages
# =============================================================================
# Each language has: name (for display), instruction (for LLM), greeting (first message)
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

# Default hyperparameters for the agent
DEFAULT_HYPERPARAMS = {
    "history_days": 30,           # Days of historical data for weather/NDVI
    "max_context_messages": 10,   # Max messages to include in context
    "temperature": 0.0,           # LLM temperature (0 = deterministic)
    "language": "english",        # Default response language
}

# =============================================================================
# TOOL REGISTRY - Tools available to the agent
# =============================================================================
# The LLM decides when to call these tools based on user's query
# This is the ReAct pattern - no hardcoded rules for when to fetch data
AGENT_TOOLS = [get_agri_weather, check_mandi_prices, get_satellite_image, get_crop_health_ndvi]


# =============================================================================
# STATE SCHEMA - TypedDict defining the state structure
# =============================================================================
class AgentState(TypedDict):
    """
    State schema for the LangGraph agent.
    
    Attributes:
        messages: Conversation history (uses operator.add for accumulation)
        image_path: Path to uploaded image (if any)
        diagnosis: Result from image analysis
        location: {lat, lon} for satellite/weather queries
        crop_name: User's specified crop
        city: User's city for market prices
        satellite_data: Cached satellite imagery data
        language: Response language code
        history_days: Days of historical data to fetch
        temperature: LLM creativity parameter
    """
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]], operator.add]
    image_path: Optional[str]
    diagnosis: Optional[str]
    location: Optional[dict]
    crop_name: Optional[str]
    city: Optional[str]
    satellite_data: Optional[dict]
    language: Optional[str]
    history_days: Optional[int]
    temperature: Optional[float]


def _get_llm(state: AgentState, with_tools: bool = False) -> ChatGoogleGenerativeAI:
    temp = state.get("temperature", DEFAULT_TEMPERATURE)
    llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL, temperature=temp, google_api_key=GOOGLE_API_KEY)
    return llm.bind_tools(AGENT_TOOLS) if with_tools else llm


def _get_language_info(state: AgentState) -> tuple:
    lang = state.get("language", "english").lower()
    info = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["english"])
    return info["instruction"], info["name"], info["greeting"]


def _get_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "image/jpeg"


# --- Standalone Image Analysis Function (called from app.py) ---
def analyze_image_standalone(image_path: str, language: str = "english", temperature: float = 0.0) -> str:
    """
    Step 1: Analyze image and return detailed description.
    This is called BEFORE the main agent to get image context.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    
    lang_info = SUPPORTED_LANGUAGES.get(language.lower(), SUPPORTED_LANGUAGES["english"])
    lang_instruction = lang_info["instruction"]
    lang_name = lang_info["name"]
    
    try:
        llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL, temperature=temperature, google_api_key=GOOGLE_API_KEY)
        mime_type = _get_mime_type(image_path)
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = f"""**CRITICAL**: Respond ONLY in {lang_name}. {lang_instruction}

Analyze this image thoroughly and provide:

1. **What is in the image**: Identify the crop/plant type
2. **Health Assessment**: 
   - Overall health status (Healthy / Diseased / Stressed / Damaged)
   - If diseased: Name the specific disease
   - If pest damage: Identify the pest
   - If nutrient deficiency: Identify which nutrient
3. **Visual Symptoms**: Describe exactly what you see (color changes, spots, wilting, etc.)
4. **Severity**: Low / Medium / High / Critical
5. **Confidence**: High / Medium / Low

Be detailed and accurate - this analysis will be used to provide treatment advice."""
        
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
        ])
        
        response = llm.invoke([msg])
        logger.info(f"Image analysis completed: {len(response.content)} chars")
        return response.content
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return f"Error analyzing image: {str(e)}"


# --- Router ---
def router_node(state: AgentState) -> str:
    messages = state.get('messages', [])
    last_message = messages[-1].content if messages else ""
    image_path = state.get('image_path')
    
    if image_path and os.path.exists(image_path):
        logger.info("Router -> pathologist (image detected)")
        return "pathologist"
    
    settings_keywords = ["settings", "change language", "set language", "configure"]
    if any(kw in last_message.lower() for kw in settings_keywords):
        logger.info("Router -> settings_handler")
        return "settings_handler"
    
    logger.info("Router -> main_agent")
    return "main_agent"


# --- Settings Handler ---
def settings_handler(state: AgentState):
    import re
    last_message = state['messages'][-1].content.lower()
    updates, response_parts = {}, []
    
    for lang_key, lang_info in SUPPORTED_LANGUAGES.items():
        if lang_key in last_message:
            updates["language"] = lang_key
            response_parts.append(f"Language set to {lang_info['name']}")
    
    days_match = re.search(r'(\d+)\s*days?', last_message)
    if days_match:
        days = int(days_match.group(1))
        if 1 <= days <= 90:
            updates["history_days"] = days
            response_parts.append(f"History set to {days} days")
    
    if response_parts:
        response = "✅ Settings updated:\n- " + "\n- ".join(response_parts)
    else:
        lang = state.get("language", "english")
        days = state.get("history_days", 30)
        response = f"📋 **Current Settings:**\n- Language: {SUPPORTED_LANGUAGES.get(lang, {}).get('name', lang)}\n- History Days: {days}\n\nSay 'change language to urdu' or 'set 60 days' to change."
    
    return {"messages": [AIMessage(content=response)], **updates}


# --- Pathologist (Image Analysis) ---
def pathologist_agent(state: AgentState):
    lang_instruction, lang_name, _ = _get_language_info(state)
    llm = _get_llm(state)
    image_path = state.get('image_path')
    
    # Get user's actual question from messages
    user_query = ""
    messages = state.get('messages', [])
    if messages and hasattr(messages[-1], 'content'):
        content = messages[-1].content
        # Extract actual query if it contains CURRENT QUERY marker
        if "CURRENT QUERY:" in content:
            user_query = content.split("CURRENT QUERY:")[-1].strip()
        else:
            user_query = content
    
    if not image_path or not os.path.exists(image_path):
        return {"messages": [AIMessage(content="No valid image provided. Please upload an image for analysis.")]}
    
    try:
        mime_type = _get_mime_type(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Include user's question in the analysis prompt
        user_question_part = f"\n\nUSER'S QUESTION: {user_query}" if user_query else ""
        
        prompt = f"""**CRITICAL**: Respond ONLY in {lang_name}. {lang_instruction}

Analyze this crop/plant image and provide:
1. **Crop/Plant**: Identify what plant/crop this is
2. **Health Status**: Healthy / Diseased / Stressed / Damaged
3. **Problem Identified**: If any issue, name the disease/pest/deficiency
4. **Confidence**: High / Medium / Low
5. **Visual Symptoms**: What signs do you observe{user_question_part}

Answer the user's specific question if they asked one. Be accurate and helpful - this guides treatment decisions."""
        
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
        ])
        response = llm.invoke([msg])
        return {"diagnosis": response.content, "messages": [response]}
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return {"diagnosis": str(e), "messages": [AIMessage(content=f"Image analysis error: {e}")]}


# --- Treatment Advisor ---
def treatment_advisor(state: AgentState):
    diagnosis = state.get("diagnosis")
    if not diagnosis:
        return {"messages": [AIMessage(content="No diagnosis available.")]}
    
    lang_instruction, lang_name, _ = _get_language_info(state)
    llm = _get_llm(state, with_tools=True)
    location = state.get("location") or DEFAULT_LOCATION
    days = state.get("history_days", 30)
    
    system_prompt = f"""**CRITICAL**: Respond ONLY in {lang_name}. {lang_instruction}

You are a Treatment Advisor. Based on the diagnosis, provide treatment advice.

TOOLS AVAILABLE (use ONLY if relevant):
- get_agri_weather(lat, lon, days): Weather data. Use if weather affects treatment (e.g., fungal disease - check humidity/rain)
- get_crop_health_ndvi(lat, lon, days): NDVI health index. Use if field-wide assessment helps.
- get_satellite_image(lat, lon): Satellite image. Rarely needed.
- check_mandi_prices(crop_name, city): Market prices. NOT relevant for treatment.

DIAGNOSIS: {diagnosis}
LOCATION: lat={location['lat']}, lon={location['lon']}
DAYS: {days}

DECISION RULES:
- Fungal/bacterial disease? -> GET weather (humidity matters)
- Nutrient deficiency? -> NO tools needed, just give fertilizer advice
- Pest damage? -> NO tools needed, give pesticide recommendations
- Want field overview? -> GET NDVI

Provide practical treatment advice in {lang_name}."""

    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=f"Provide treatment for: {diagnosis}")])
        if response.tool_calls:
            logger.info(f"Treatment calling tools: {[tc['name'] for tc in response.tool_calls]}")
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {e}")]}


# --- Main Agent (General Chat with Tools) ---
def main_agent(state: AgentState):
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    lang_instruction, lang_name, greeting = _get_language_info(state)
    
    # Check if first message
    prev_ai = [m for m in messages[:-1] if isinstance(m, AIMessage)]
    is_first = len(prev_ai) == 0
    
    llm = _get_llm(state, with_tools=True)
    location = state.get("location") or DEFAULT_LOCATION
    days = state.get("history_days", 30)
    crop = state.get("crop_name", "")
    city = state.get("city", "")
    diagnosis = state.get("diagnosis", "")
    
    # Build context
    context = ""
    if len(messages) > 1:
        items = []
        for m in messages[-6:-1]:
            if isinstance(m, HumanMessage):
                items.append(f"User: {m.content[:150]}")
            elif isinstance(m, AIMessage) and m.content:
                items.append(f"Assistant: {m.content[:150]}")
        context = "\n".join(items) if items else ""
    
    greeting_line = f"Start with: {greeting}" if is_first else "NO greeting - answer directly."
    diagnosis_ctx = f"\nPREVIOUS DIAGNOSIS: {diagnosis[:300]}" if diagnosis else ""
    
    system_prompt = f"""**CRITICAL**: Respond ONLY in {lang_name}. {lang_instruction}

You are KisaanLink, an AI farming assistant. {greeting_line}

TOOLS (use ONLY when needed):
- get_agri_weather(lat, lon, days): For weather questions
- check_mandi_prices(crop_name, city): For price/market questions  
- get_satellite_image(lat, lon): ONLY if user asks for satellite imagery
- get_crop_health_ndvi(lat, lon, days): ONLY if user asks for NDVI/crop health from satellite

WHEN TO USE TOOLS:
✅ "What's the weather in Multan?" -> get_agri_weather
✅ "What's wheat price in Lahore?" -> check_mandi_prices
✅ "Show me satellite image" -> get_satellite_image
✅ "Check NDVI for my field" -> get_crop_health_ndvi

WHEN NOT TO USE TOOLS:
❌ "Hello", "Hi" -> Just greet back, NO tools
❌ "How to grow tomatoes?" -> Answer from knowledge, NO tools
❌ "What fertilizer for wheat?" -> Answer from knowledge, NO tools
❌ Follow-up questions -> Usually NO tools unless specifically asking for new data

DEFAULT: lat={location['lat']}, lon={location['lon']}, days={days}
{f"CROP: {crop}" if crop else ""} {f"CITY: {city}" if city else ""}
{f"CONTEXT:{chr(10)}{context}" if context else ""}{diagnosis_ctx}"""

    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
        if response.tool_calls:
            logger.info(f"Main agent calling: {[tc['name'] for tc in response.tool_calls]}")
        else:
            logger.info("Main agent: direct response (no tools)")
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {e}")]}


# --- Tool Result Processor ---
def process_tool_results(state: AgentState):
    messages = state.get("messages", [])
    lang_instruction, lang_name, _ = _get_language_info(state)
    llm = _get_llm(state)
    
    tool_results = [f"[{msg.name}]: {msg.content}" for msg in messages if isinstance(msg, ToolMessage)]
    if not tool_results:
        return {"messages": []}
    
    user_query = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    
    prompt = f"""**CRITICAL**: Respond ONLY in {lang_name}. {lang_instruction}

USER QUESTION: {user_query}

TOOL RESULTS:
{chr(10).join(tool_results)}

Synthesize this into a clear, helpful response for a farmer."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {e}")]}


# --- Flow Control ---
def should_continue(state: AgentState) -> str:
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return "continue"
    return "end"


# --- Build Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("pathologist", pathologist_agent)
workflow.add_node("treatment_advisor", treatment_advisor)
workflow.add_node("settings_handler", settings_handler)
workflow.add_node("main_agent", main_agent)
workflow.add_node("tools", ToolNode(AGENT_TOOLS))
workflow.add_node("synthesize", process_tool_results)

workflow.set_conditional_entry_point(router_node, {
    "pathologist": "pathologist",
    "settings_handler": "settings_handler",
    "main_agent": "main_agent"
})

workflow.add_edge("pathologist", "treatment_advisor")
workflow.add_conditional_edges("treatment_advisor", should_continue, {"continue": "tools", "end": END})
workflow.add_conditional_edges("main_agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "synthesize")
workflow.add_edge("synthesize", END)
workflow.add_edge("settings_handler", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# --- Helper Functions ---
def run_agent(user_message: str, image_path: Optional[str] = None, location: Optional[dict] = None,
              crop_name: Optional[str] = None, city: Optional[str] = None, thread_id: str = "default",
              language: str = "english", history_days: int = 30, temperature: float = 0.0,
              diagnosis: Optional[str] = None) -> str:
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "image_path": image_path, "diagnosis": diagnosis, "location": location,
        "crop_name": crop_name, "city": city, "language": language,
        "history_days": history_days, "temperature": temperature, "satellite_data": None
    }
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = app.invoke(initial_state, config)
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
                    return msg.content
            return result["messages"][-1].content
        return "No response."
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {e}"


def get_supported_languages():
    return SUPPORTED_LANGUAGES

def get_default_hyperparams():
    return DEFAULT_HYPERPARAMS
