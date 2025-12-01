# 🌾 KisaanLink: AI-Powered Agricultural Assistant

<div align="center">

![KisaanLink Banner](https://img.shields.io/badge/KisaanLink-AI%20Agronomist-green?style=for-the-badge&logo=seedling)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Framework-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Satellite%20Data-orange.svg)](https://earthengine.google.com/)
[![Gemini](https://img.shields.io/badge/Gemini%202.5-Flash-blue.svg)](https://ai.google.dev/)

**An intelligent farming assistant for Pakistani farmers, powered by AI and satellite technology.**

[Live Demo](#deployment) • [Features](#-features) • [Architecture](#-architecture) • [Setup](#-setup--installation) • [Usage](#-usage)

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Statement

Pakistani farmers face significant challenges in modern agriculture:

1. **Limited Access to Expert Advice**: Agricultural experts are scarce in rural areas, leaving farmers without timely guidance for crop diseases and pest management.

2. **Language Barriers**: Most agricultural technology solutions are in English, creating accessibility issues for farmers who speak Urdu, Punjabi, Sindhi, or other regional languages.

3. **Lack of Real-Time Data**: Farmers often make decisions without access to current weather patterns, soil conditions, or satellite-based crop health assessments.

4. **Market Information Gap**: Farmers struggle to get accurate, up-to-date market prices (Mandi rates), leading to unfair pricing and reduced profits.

5. **Climate Uncertainty**: With changing weather patterns, farmers need predictive insights to make informed planting and irrigation decisions.

---

## 💡 Solution

**KisaanLink** is an AI-powered agricultural assistant that addresses these challenges by providing:

### 🔬 Intelligent Crop Disease Diagnosis
- Upload a photo of your crop → Get instant disease identification
- AI-powered image analysis using Gemini 2.5 Vision
- Treatment recommendations based on diagnosis

### 🛰️ Satellite-Based Monitoring
- **Sentinel-2 Imagery**: 10m resolution false-color imagery showing crop health
- **NDVI Analysis**: Normalized Difference Vegetation Index for field-wide health assessment
- **ERA5 Weather Data**: Historical and current weather conditions

### 🌐 Multi-Language Support
- **English**, **اردو (Urdu)**, **پنجابی (Punjabi)**, **سنڌي (Sindhi)**, **हिंदी (Hindi)**
- Natural conversation in the farmer's preferred language

### 🤖 Intelligent Agent Architecture
- ReAct-style agent that decides when to fetch external data
- Doesn't call APIs unnecessarily (e.g., won't fetch weather for "Hello")
- Context-aware conversations with memory

### 💰 Market Price Integration
- Real-time Mandi prices from AMIS Pakistan
- Price tracking for major crops across Pakistani cities

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 **Image Analysis** | Upload crop photos for AI-powered disease detection |
| 🛰️ **Satellite View** | Sentinel-2 false-color imagery for vegetation health |
| 🌱 **NDVI Index** | Quantified crop health assessment (0-1 scale) |
| 🌤️ **Weather Dashboard** | Temperature, precipitation, soil moisture from ERA5 |
| 💰 **Price Lookup** | Mandi rates for wheat, rice, cotton, and more |
| 🌐 **5 Languages** | English, Urdu, Punjabi, Sindhi, Hindi |
| 🧠 **Smart Agent** | Only fetches data when relevant to your question |
| 💾 **Memory** | Remembers diagnosis for follow-up questions |

---

## 🏗️ Architecture

### High-Level System Architecture

![System Architecture](sys-architecture.png)

### Agent Flow Diagram

![Agent Flow](agent-flow.png)

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Gemini 2.5 Flash | Conversation & Image Analysis |
| **Agent Framework** | LangGraph | State machine for agent orchestration |
| **Satellite Data** | Google Earth Engine | Sentinel-2, ERA5-Land |
| **Frontend** | Streamlit | Interactive web interface |
| **Visualization** | Plotly | NDVI gauges and charts |
| **Search** | DuckDuckGo | Fallback for market prices |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Google Cloud account with Earth Engine access
- Google AI Studio API key (for Gemini)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Xeeshan85/kisaan-link.git
cd kisaan-link
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```properties
# Required
GOOGLE_API_KEY=your_gemini_api_key_here
EE_PROJECT=your_gcp_project_id_here

# Optional
LLM_MODEL=gemini-2.5-flash
DEFAULT_LAT=31.5204
DEFAULT_LON=74.3587
```

### Step 5: Authenticate Earth Engine

```bash
earthengine authenticate
```

Follow the browser prompts to authenticate with your Google account.

### Step 6: Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📖 Usage

### Chat Interface

1. **General Questions**: Ask farming questions in any supported language
   - "How do I grow tomatoes?"
   - "گندم کی کاشت کیسے کریں؟" (Urdu)

2. **Weather Queries**: Ask about weather for your location
   - "What's the weather like for my crops?"
   - Automatically fetches ERA5 data

3. **Image Analysis**: Upload a crop photo
   - Click the upload button
   - Type your question (e.g., "What disease is this?")
   - Get diagnosis + treatment advice

4. **Market Prices**: Ask about Mandi rates
   - "What's the wheat price in Lahore?"

### Satellite Tab

- View Sentinel-2 false-color imagery
- Check NDVI crop health index
- Adjust location in sidebar settings

### Settings

- Change language (sidebar dropdown)
- Adjust analysis history (7-90 days)
- Set custom location coordinates

---

## 📚 API Reference

### Tools

| Tool | Parameters | Returns |
|------|------------|---------|
| `get_satellite_image` | `lat`, `lon`, `buffer_meters` | Sentinel-2 image URL |
| `get_crop_health_ndvi` | `lat`, `lon`, `days`, `buffer_meters` | NDVI value (0-1) |
| `get_agri_weather` | `lat`, `lon`, `days` | Temperature, precipitation, soil moisture |
| `check_mandi_prices` | `crop_name`, `city` | Market price information |

### State Schema

```python
class AgentState(TypedDict):
    messages: List[Message]      # Conversation history
    image_path: Optional[str]    # Uploaded image path
    diagnosis: Optional[str]     # Image analysis result
    location: Optional[dict]     # {lat, lon}
    crop_name: Optional[str]     # User's crop
    city: Optional[str]          # User's city
    language: Optional[str]      # Response language
    history_days: Optional[int]  # Days for weather/NDVI
    temperature: Optional[float] # LLM creativity
```

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub (ensure `.env` is in `.gitignore`)

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect your GitHub repository

4. Set secrets in Streamlit Cloud dashboard:
   ```toml
   GOOGLE_API_KEY = "your_api_key"
   EE_PROJECT = "your_project_id"
   ```

5. Deploy!

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## 📁 Project Structure

```
kisaan_link/
├── app.py              # Streamlit frontend (414 lines)
├── graph.py            # LangGraph agent orchestration (455 lines)
├── tools.py            # Earth Engine & external tools (478 lines)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Earth Engine** for satellite data access
- **Google AI** for Gemini 2.5 Flash
- **LangChain/LangGraph** for agent framework
- **AMIS Pakistan** for agricultural market data
- **Pakistani Farmers** who inspired this project

---

<div align="center">

**Built with ❤️ for Pakistani Farmers**

🌾 *Empowering Agriculture with AI* 🌾

</div>
