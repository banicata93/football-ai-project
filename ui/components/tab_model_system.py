import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from api_client import FootballAPIClient


def render(client: FootballAPIClient):
    st.header("📊 Models & System Health")
    
    # Refresh button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh", type="primary", key="model_system_refresh"):
            st.rerun()
    
    # Get all system data
    with st.spinner("Loading system information..."):
        health_resp = client.get_health()
        stats_resp = client.get_stats()
        models_resp = client.get_models()
        teams_resp = client.get_teams(limit=50)
    
    # System Health
    st.subheader("🏥 System Health")
    _render_system_health(health_resp)
    
    # System Statistics
    st.subheader("📈 System Statistics")
    _render_system_stats(stats_resp)
    
    # Models Information
    st.subheader("🤖 Models Information")
    _render_models_info(models_resp)
    
    # Top Teams
    st.subheader("🏆 Top Teams by Elo")
    _render_top_teams(teams_resp)
    
    # Raw data for debugging
    with st.expander("🔍 Raw API Responses"):
        st.json({
            "health": health_resp,
            "stats": stats_resp,
            "models": models_resp,
            "teams": teams_resp
        })


def _render_system_health(health_resp):
    """Render system health status"""
    
    if not health_resp["ok"]:
        st.error(f"❌ Cannot connect to API: {health_resp['error']}")
        return
    
    health_data = health_resp["data"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = health_data.get("status", "unknown")
        if status == "healthy":
            st.success("✅ Healthy")
        elif status == "degraded":
            st.warning("⚠️ Degraded")
        else:
            st.error("❌ Unhealthy")
    
    with col2:
        models_loaded = health_data.get("models_loaded", False)
        if models_loaded:
            st.success("✅ Models Loaded")
        else:
            st.error("❌ Models Not Loaded")
    
    with col3:
        uptime = health_data.get("uptime_seconds", 0)
        uptime_hours = uptime / 3600 if uptime > 0 else 0
        st.metric("Uptime", f"{uptime_hours:.1f}h")
    
    with col4:
        version = health_data.get("version", "unknown")
        st.metric("Version", version)


def _render_system_stats(stats_resp):
    """Render system statistics"""
    
    if not stats_resp["ok"]:
        st.warning(f"⚠️ Stats not available: {stats_resp['error']}")
        return
    
    stats_data = stats_resp["data"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_teams = stats_data.get("total_teams", 0)
        st.metric("Total Teams", total_teams)
    
    with col2:
        total_matches = stats_data.get("total_matches", 0)
        st.metric("Total Matches", total_matches)
    
    with col3:
        total_predictions = stats_data.get("total_predictions", 0)
        st.metric("Predictions Made", total_predictions)
    
    with col4:
        total_features = stats_data.get("total_features", 0)
        st.metric("Features", total_features)
    
    # Performance metrics if available
    performance = stats_data.get("performance", {})
    if performance:
        st.markdown("#### ⚡ Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_response = performance.get("avg_response_time_ms", 0)
            st.metric("Avg Response Time", f"{avg_response:.0f}ms")
        
        with col2:
            requests_per_min = performance.get("requests_per_minute", 0)
            st.metric("Requests/min", f"{requests_per_min:.1f}")
        
        with col3:
            cache_hit_rate = performance.get("cache_hit_rate", 0)
            st.metric("Cache Hit Rate", f"{cache_hit_rate:.1%}")


def _render_models_info(models_resp):
    """Render models information and metrics"""
    
    if not models_resp["ok"]:
        st.warning(f"⚠️ Models info not available: {models_resp['error']}")
        return
    
    models_data = models_resp["data"]
    models = models_data.get("models", [])
    
    if not models:
        st.info("No models information available")
        return
    
    # Models overview
    total_models = len(models)
    loaded_models = sum(1 for model in models if model.get("loaded", True))  # Assume loaded if not specified
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models", total_models)
    
    with col2:
        st.metric("Loaded Models", loaded_models)
    
    with col3:
        load_rate = (loaded_models / total_models * 100) if total_models > 0 else 0
        st.metric("Load Success Rate", f"{load_rate:.1f}%")
    
    # Models table
    st.markdown("#### 📋 Model Details")
    
    table_data = []
    
    for model in models:
        model_name = model.get("model_name", "Unknown")
        version = model.get("version", "N/A")
        trained_date = model.get("trained_date", "N/A")
        accuracy = model.get("accuracy")
        metrics = model.get("metrics", {})
        
        # Try to get accuracy from metrics if not in main object
        if accuracy is None and metrics:
            accuracy = (
                metrics.get("accuracy") or 
                metrics.get("1x2_accuracy") or 
                metrics.get("accuracy_1x2")
            )
        
        # Get log loss
        log_loss = None
        if metrics:
            log_loss = (
                metrics.get("log_loss") or
                metrics.get("1x2_log_loss") or
                metrics.get("log_loss_1x2")
            )
        
        table_data.append({
            "Model": model_name,
            "Version": version,
            "Trained": trained_date,
            "Accuracy": f"{accuracy:.3f}" if accuracy else "N/A",
            "Log Loss": f"{log_loss:.3f}" if log_loss else "N/A",
            "Status": "✅ Loaded" if model.get("loaded", True) else "❌ Not Loaded"
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Model accuracy chart
    _render_model_accuracy_chart(models)


def _render_model_accuracy_chart(models):
    """Render model accuracy comparison chart"""
    
    model_names = []
    accuracies = []
    
    for model in models:
        model_name = model.get("model_name", "Unknown")
        accuracy = model.get("accuracy")
        metrics = model.get("metrics", {})
        
        # Try to get accuracy from metrics
        if accuracy is None and metrics:
            accuracy = (
                metrics.get("accuracy") or 
                metrics.get("1x2_accuracy") or 
                metrics.get("accuracy_1x2")
            )
        
        if accuracy and accuracy > 0:
            model_names.append(model_name)
            accuracies.append(accuracy)
    
    if model_names and accuracies:
        st.markdown("#### 📊 Model Accuracy Comparison")
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=accuracies,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)],
                text=[f"{acc:.3f}" for acc in accuracies],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def _render_top_teams(teams_resp):
    """Render top teams by Elo rating"""
    
    if not teams_resp["ok"]:
        st.warning(f"⚠️ Teams data not available: {teams_resp['error']}")
        return
    
    teams_data = teams_resp["data"]
    teams = teams_data.get("teams", [])
    
    if not teams:
        st.info("No teams data available")
        return
    
    # Process teams data
    table_data = []
    
    for i, team in enumerate(teams[:20]):  # Top 20
        team_name = team.get("name", "Unknown")
        elo_rating = team.get("elo_rating", 1500)
        league = team.get("league", "Unknown")
        matches_played = team.get("matches_played", 0)
        
        table_data.append({
            "Rank": i + 1,
            "Team": team_name,
            "Elo Rating": elo_rating,
            "League": league,
            "Matches": matches_played
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        
        # Sort by Elo rating
        df = df.sort_values("Elo Rating", ascending=False)
        df["Rank"] = range(1, len(df) + 1)
        
        # Color-code by Elo rating
        def highlight_elo(val):
            if isinstance(val, (int, float)):
                if val >= 2000:
                    return 'background-color: #d4edda'  # Light green
                elif val >= 1800:
                    return 'background-color: #fff3cd'  # Light yellow
                elif val >= 1600:
                    return 'background-color: #ffeaa7'  # Light orange
                else:
                    return 'background-color: #f8d7da'  # Light red
            return ''
        
        styled_df = df.style.applymap(highlight_elo, subset=['Elo Rating'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Elo distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, 
                x="Elo Rating", 
                nbins=10,
                title="Elo Rating Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top leagues
            league_counts = df["League"].value_counts().head(5)
            if not league_counts.empty:
                fig = px.pie(
                    values=league_counts.values,
                    names=league_counts.index,
                    title="Top Leagues"
                )
                st.plotly_chart(fig, use_container_width=True)
