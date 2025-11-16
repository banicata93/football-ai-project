import streamlit as st
from api_client import FootballAPIClient
from components import (
    tab_overview,
    tab_single_match,
    tab_next_round,
    tab_league_explorer,
    tab_scoreline_lab,
    tab_model_system,
    tab_api_explorer,
)


@st.cache_resource
def get_client():
    return FootballAPIClient()


def main():
    st.set_page_config(
        page_title="Football AI Dashboard",
        page_icon="⚽",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e3c72;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚽ Football AI Dashboard</h1>
        <p>Advanced ML-powered football predictions with Hybrid 1X2, Scoreline Analysis & System Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize client
    client = get_client()
    
    # Connection status
    _render_connection_status(client)
    
    # Main tabs
    tabs = st.tabs([
        "🏠 Overview",
        "⚽ Single Match",
        "📅 Next Round",
        "🌍 League Explorer",
        "🎯 Scoreline Lab",
        "📊 Models & System",
        "🧪 API Explorer",
    ])
    
    with tabs[0]:
        tab_overview.render(client)
    
    with tabs[1]:
        tab_single_match.render(client)
    
    with tabs[2]:
        tab_next_round.render(client)
    
    with tabs[3]:
        tab_league_explorer.render(client)
    
    with tabs[4]:
        tab_scoreline_lab.render(client)
    
    with tabs[5]:
        tab_model_system.render(client)
    
    with tabs[6]:
        tab_api_explorer.render(client)


def _render_connection_status(client):
    """Render API connection status"""
    
    health_resp = client.get_health()
    
    if health_resp["ok"]:
        health_data = health_resp["data"]
        status = health_data.get("status", "unknown")
        models_loaded = health_data.get("models_loaded", False)
        version = health_data.get("version", "unknown")
        
        if status == "healthy":
            st.success(f"🟢 Connected to Football AI API - Status: {status} | Models: {'✅' if models_loaded else '❌'} | Version: {version}")
        else:
            st.warning(f"⚠️ API Status: {status} | Models: {'✅' if models_loaded else '❌'} | Version: {version}")
    else:
        st.error(f"🔴 Cannot connect to Football AI API: {health_resp['error']}")


if __name__ == "__main__":
    main()
