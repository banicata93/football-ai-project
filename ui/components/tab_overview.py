import streamlit as st
import pandas as pd
from api_client import FootballAPIClient


def render(client: FootballAPIClient):
    st.header("🏠 Overview")
    
    # Get data
    health_resp = client.get_health()
    stats_resp = client.get_stats()
    models_resp = client.get_models()
    
    # System Status
    st.subheader("🔋 System Status")
    
    if health_resp["ok"]:
        health_data = health_resp["data"]
        status = health_data.get("status", "unknown")
        models_loaded = health_data.get("models_loaded", False)
        uptime = health_data.get("uptime_seconds", 0)
        version = health_data.get("version", "unknown")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if status == "healthy":
                st.success(f"✅ {status.title()}")
            else:
                st.error(f"❌ {status.title()}")
        
        with col2:
            st.metric("Models Loaded", "✅ Yes" if models_loaded else "❌ No")
        
        with col3:
            uptime_hours = uptime / 3600 if uptime > 0 else 0
            st.metric("Uptime", f"{uptime_hours:.1f}h")
        
        with col4:
            st.metric("Version", version)
    else:
        st.error(f"❌ Cannot connect to API: {health_resp['error']}")
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    
    if stats_resp["ok"]:
        stats_data = stats_resp["data"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_teams = stats_data.get("total_teams", 0)
            st.metric("Total Teams", total_teams)
        
        with col2:
            total_matches = stats_data.get("total_matches", 0)
            st.metric("Total Matches", total_matches)
        
        with col3:
            total_features = stats_data.get("total_features", 0)
            st.metric("Total Features", total_features)
    else:
        st.warning(f"⚠️ Stats not available: {stats_resp['error']}")
    
    # Models Overview
    st.subheader("🤖 Models Overview")
    
    if models_resp["ok"]:
        models_data = models_resp["data"]
        models = models_data.get("models", [])
        
        if models:
            # Create table
            table_data = []
            for model in models:
                model_name = model.get("model_name", "Unknown")
                version = model.get("version", "N/A")
                accuracy = model.get("accuracy")
                metrics = model.get("metrics", {})
                
                # Try to get accuracy from metrics if not in main object
                if accuracy is None and metrics:
                    accuracy = (
                        metrics.get("accuracy") or 
                        metrics.get("1x2_accuracy") or 
                        metrics.get("accuracy_1x2")
                    )
                
                table_data.append({
                    "Model": model_name,
                    "Version": version,
                    "Accuracy": f"{accuracy:.3f}" if accuracy else "N/A"
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No models information available")
    else:
        st.warning(f"⚠️ Models info not available: {models_resp['error']}")
    
    # Raw data expander for debugging
    with st.expander("🔍 Raw API Responses"):
        st.json({
            "health": health_resp,
            "stats": stats_resp,
            "models": models_resp
        })
