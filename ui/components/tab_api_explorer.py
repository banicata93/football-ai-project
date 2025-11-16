import streamlit as st
import json
from api_client import FootballAPIClient


def render(client: FootballAPIClient):
    st.header("🧪 API Explorer")
    
    st.markdown("""
    ### Test API Endpoints
    
    Use this tool to explore and test the Football AI API endpoints directly.
    """)
    
    # Endpoint selection
    endpoint_type = st.selectbox(
        "Select Endpoint Type",
        ["GET Endpoints", "POST Endpoints"]
    )
    
    if endpoint_type == "GET Endpoints":
        _render_get_endpoints(client)
    else:
        _render_post_endpoints(client)


def _render_get_endpoints(client):
    """Render GET endpoints explorer"""
    
    st.subheader("📥 GET Endpoints")
    
    endpoint = st.selectbox(
        "Choose Endpoint",
        ["/health", "/stats", "/models", "/teams", "/predict/leagues"]
    )
    
    # Parameters for some endpoints
    params = {}
    
    if endpoint == "/teams":
        limit = st.number_input("Limit", min_value=1, max_value=1000, value=100)
        params["limit"] = limit
    
    if st.button("🚀 Send Request", type="primary", key="get_request_btn"):
        with st.spinner(f"Calling {endpoint}..."):
            
            if endpoint == "/health":
                result = client.get_health()
            elif endpoint == "/stats":
                result = client.get_stats()
            elif endpoint == "/models":
                result = client.get_models()
            elif endpoint == "/teams":
                result = client.get_teams(limit=params.get("limit", 100))
            elif endpoint == "/predict/leagues":
                result = client.get_leagues()
            else:
                result = {"ok": False, "error": "Unknown endpoint"}
            
            _display_response(result, endpoint)


def _render_post_endpoints(client):
    """Render POST endpoints explorer"""
    
    st.subheader("📤 POST Endpoints")
    
    endpoint = st.selectbox(
        "Choose Endpoint",
        ["/predict/improved", "/predict"]
    )
    
    # JSON body editor
    st.markdown("#### 📝 Request Body (JSON)")
    
    if endpoint == "/predict/improved":
        default_body = {
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "league": "Premier League",
            "date": "2025-11-20"
        }
    else:
        default_body = {
            "home_team": "Manchester City",
            "away_team": "Arsenal",
            "league": "Premier League"
        }
    
    json_body = st.text_area(
        "JSON Body",
        value=json.dumps(default_body, indent=2),
        height=200,
        help="Edit the JSON request body"
    )
    
    # Validate JSON
    try:
        parsed_json = json.loads(json_body)
        st.success("✅ Valid JSON")
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON: {str(e)}")
        parsed_json = None
    
    if st.button("🚀 Send Request", type="primary", key="post_request_btn") and parsed_json:
        with st.spinner(f"Calling {endpoint}..."):
            
            if endpoint == "/predict/improved":
                result = client.predict_improved(
                    home_team=parsed_json.get("home_team", ""),
                    away_team=parsed_json.get("away_team", ""),
                    league=parsed_json.get("league"),
                    date=parsed_json.get("date")
                )
            elif endpoint == "/predict":
                # Use generic POST method
                result = client.call_post(endpoint, parsed_json)
            else:
                result = {"ok": False, "error": "Unknown endpoint"}
            
            _display_response(result, endpoint)


def _display_response(result, endpoint):
    """Display API response"""
    
    st.markdown("#### 📋 Response")
    
    if result["ok"]:
        st.success(f"✅ Request to {endpoint} successful")
        
        # Show formatted response
        data = result["data"]
        
        # Special formatting for certain endpoints
        if endpoint == "/health":
            _format_health_response(data)
        elif endpoint == "/models":
            _format_models_response(data)
        elif endpoint in ["/predict/improved", "/predict"]:
            _format_prediction_response(data)
        else:
            st.json(data)
    else:
        st.error(f"❌ Request to {endpoint} failed")
        st.error(f"Error: {result['error']}")
    
    # Raw JSON response
    with st.expander("🔍 Raw JSON Response"):
        st.json(result)


def _format_health_response(data):
    """Format health response nicely"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = data.get("status", "unknown")
        if status == "healthy":
            st.success(f"Status: {status}")
        else:
            st.error(f"Status: {status}")
    
    with col2:
        models_loaded = data.get("models_loaded", False)
        st.metric("Models Loaded", "✅ Yes" if models_loaded else "❌ No")
    
    with col3:
        version = data.get("version", "unknown")
        st.metric("Version", version)
    
    uptime = data.get("uptime_seconds", 0)
    if uptime > 0:
        st.metric("Uptime", f"{uptime/3600:.1f} hours")


def _format_models_response(data):
    """Format models response nicely"""
    
    models = data.get("models", [])
    total_models = data.get("total_models", len(models))
    
    st.metric("Total Models", total_models)
    
    if models:
        st.markdown("**Models:**")
        for model in models:
            model_name = model.get("model_name", "Unknown")
            version = model.get("version", "N/A")
            accuracy = model.get("accuracy")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text(f"📊 {model_name}")
            with col2:
                st.text(f"Version: {version}")
            with col3:
                if accuracy:
                    st.text(f"Accuracy: {accuracy:.3f}")
                else:
                    st.text("Accuracy: N/A")


def _format_prediction_response(data):
    """Format prediction response nicely"""
    
    # 1X2 Prediction
    pred_1x2 = data.get("prediction_1x2", {})
    if pred_1x2:
        st.markdown("**1X2 Prediction:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prob_home = pred_1x2.get("prob_home_win", 0)
            st.metric("Home Win", f"{prob_home:.1%}")
        
        with col2:
            prob_draw = pred_1x2.get("prob_draw", 0)
            st.metric("Draw", f"{prob_draw:.1%}")
        
        with col3:
            prob_away = pred_1x2.get("prob_away_win", 0)
            st.metric("Away Win", f"{prob_away:.1%}")
        
        predicted_outcome = pred_1x2.get("predicted_outcome", "Unknown")
        confidence = pred_1x2.get("confidence", 0)
        using_hybrid = pred_1x2.get("using_hybrid", False)
        
        st.text(f"Predicted: {predicted_outcome} (Confidence: {confidence:.1%})")
        if using_hybrid:
            st.success("🎯 Using Hybrid Model")
    
    # OU2.5 Prediction
    pred_ou25 = data.get("prediction_ou25", {})
    if pred_ou25:
        st.markdown("**Over/Under 2.5:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prob_over = pred_ou25.get("prob_over", 0)
            st.metric("Over 2.5", f"{prob_over:.1%}")
        
        with col2:
            prob_under = pred_ou25.get("prob_under", 0)
            st.metric("Under 2.5", f"{prob_under:.1%}")
    
    # BTTS Prediction
    pred_btts = data.get("prediction_btts", {})
    if pred_btts:
        st.markdown("**Both Teams to Score:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prob_yes = pred_btts.get("prob_yes", 0)
            st.metric("BTTS Yes", f"{prob_yes:.1%}")
        
        with col2:
            prob_no = pred_btts.get("prob_no", 0)
            st.metric("BTTS No", f"{prob_no:.1%}")
    
    # FII Score
    fii = data.get("fii", {})
    if fii:
        st.markdown("**FII Score:**")
        score = fii.get("score", 0)
        st.metric("Football Intelligence Index", f"{score:.2f}")
    
    # Model versions
    model_versions = data.get("model_versions", {})
    if model_versions:
        st.markdown("**Model Versions:**")
        for model, version in model_versions.items():
            st.text(f"• {model}: {version}")
