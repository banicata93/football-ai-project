import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from api_client import FootballAPIClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.normalize_predictions import normalize_predictions
from utils.league_formatter import format_leagues_for_display


def render(client: FootballAPIClient):
    st.header("⚽ Single Match Prediction")
    
    # League selection first (outside form to enable dynamic team loading)
    st.subheader("1️⃣ Select League")
    
    # Define major leagues we support for team autocomplete
    SUPPORTED_LEAGUES = {
        'premier_league', 'la_liga', 'serie_a', 'bundesliga', 
        'ligue_1', 'eredivisie', 'primeira_liga', 'championship'
    }
    
    leagues_resp = client.get_leagues()
    if leagues_resp["ok"]:
        leagues_data = leagues_resp["data"]
        leagues = leagues_data.get("leagues", [])
        
        if leagues:
            # Format leagues for display
            formatted_leagues = format_leagues_for_display(leagues, group_by_category=False)
            
            # Filter to only show supported leagues (with team data)
            supported_formatted = [(slug, name, cat) for slug, name, cat in formatted_leagues 
                                  if any(supported in slug for supported in SUPPORTED_LEAGUES)]
            
            # If no supported leagues found, show all
            if not supported_formatted:
                supported_formatted = formatted_leagues[:20]  # Show top 20
            
            league_options = [""] + [slug for slug, _, _ in supported_formatted]
            league_display = {slug: display_name for slug, display_name, _ in supported_formatted}
            league_display[""] = "Select a league..."
        else:
            league_options = [""]
            league_display = {"": "No leagues available"}
    else:
        league_options = [""]
        league_display = {"": "Error loading leagues"}
    
    selected_league = st.selectbox(
        "League", 
        league_options,
        format_func=lambda x: league_display.get(x, x),
        key="league_selector"
    )
    
    # Load teams for selected league
    teams_list = []
    if selected_league:
        with st.spinner("Loading teams..."):
            teams_resp = client.get_teams_by_league(selected_league)
            
            if teams_resp.get("ok"):
                teams_data = teams_resp.get("data", {})
                raw_teams = teams_data.get("teams", [])
                
                teams_list = [t.get("name", t) if isinstance(t, dict) else t 
                             for t in raw_teams]
                
                if teams_list:
                    st.success(f"✅ Loaded {len(teams_list)} teams")
                else:
                    st.warning(f"⚠️ No teams found for this league. You can still type team names manually.")
            else:
                st.error(f"❌ Failed to load teams: {teams_resp.get('error')}")
    
    # Input form
    st.markdown("---")
    st.subheader("2️⃣ Select Teams")
    
    with st.form("match_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            if teams_list:
                home_team = st.selectbox(
                    "Home Team",
                    options=[""] + teams_list,
                    format_func=lambda x: x if x else "Select home team..."
                )
            else:
                home_team = st.text_input("Home Team", placeholder="e.g. Manchester United")
        
        with col2:
            if teams_list:
                away_team = st.selectbox(
                    "Away Team", 
                    options=[""] + teams_list,
                    format_func=lambda x: x if x else "Select away team..."
                )
            else:
                away_team = st.text_input("Away Team", placeholder="e.g. Liverpool")
        
        date = st.date_input("Match Date (optional)", value=None)
        
        submitted = st.form_submit_button("🔮 Predict Match", type="primary")
    
    if submitted and home_team and away_team:
        with st.spinner("Getting prediction..."):
            date_str = date.isoformat() if date else None
            
            result = client.predict_improved(
                home_team=home_team,
                away_team=away_team,
                league=selected_league if selected_league else None,
                date=date_str
            )
            
            if not result["ok"]:
                st.error(f"❌ Prediction failed: {result['error']}")
                return
            
            data = result["data"]
            
            # Display results
            st.subheader(f"📊 {home_team} vs {away_team}")
            
            # Normalize predictions
            normalized = normalize_predictions(data)
            pred_1x2 = normalized["pred_1x2"]
            pred_ou25 = normalized["pred_ou25"]
            pred_btts = normalized["pred_btts"]
            
            # Check for hybrid usage
            using_hybrid = pred_1x2.get("using_hybrid", False)
            
            if using_hybrid:
                st.success("🎯 **Hybrid 1X2 Model Active**")
            else:
                st.info("🤖 Using Standard Models")
            
            # Main predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                _render_1x2_prediction(pred_1x2, home_team, away_team)
            
            with col2:
                _render_ou25_prediction(pred_ou25)
            
            with col3:
                _render_btts_prediction(pred_btts)
            
            # FII Score
            fii = data.get("fii", {}) or {}
            if fii:
                _render_fii_score(fii)
            
            # Hybrid breakdown
            if using_hybrid:
                _render_hybrid_breakdown(pred_1x2)
            
            # Raw JSON
            with st.expander("🔍 Raw API Response"):
                st.json(data)


def _render_1x2_prediction(pred_1x2, home_team, away_team):
    st.markdown("### 🎯 Match Result (1X2)")
    
    if not pred_1x2:
        st.warning("No 1X2 prediction available")
        return
    
    prob_home = float(pred_1x2.get("prob_home_win", 0.33) or 0.33)
    prob_draw = float(pred_1x2.get("prob_draw", 0.33) or 0.33)
    prob_away = float(pred_1x2.get("prob_away_win", 0.33) or 0.33)
    predicted_outcome = pred_1x2.get("predicted_outcome", "X")
    confidence = float(pred_1x2.get("confidence", 0.33) or 0.33)
    
    # Bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=[f"{home_team} Win", "Draw", f"{away_team} Win"],
            y=[prob_home, prob_draw, prob_away],
            marker_color=['#28a745', '#ffc107', '#dc3545'],
            text=[f"{prob_home:.1%}", f"{prob_draw:.1%}", f"{prob_away:.1%}"],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Match Result Probabilities",
        yaxis_title="Probability",
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction summary
    outcome_map = {"1": f"{home_team} Win", "X": "Draw", "2": f"{away_team} Win"}
    predicted_text = outcome_map.get(predicted_outcome, "Unknown")
    
    st.metric(
        label="Predicted Result",
        value=predicted_text,
        delta=f"Confidence: {confidence:.1%}"
    )


def _render_ou25_prediction(pred_ou25):
    st.markdown("### 🥅 Over/Under 2.5")
    
    if not pred_ou25:
        st.warning("No OU2.5 prediction available")
        return
    
    prob_over = float(pred_ou25.get("prob_over", 0.5) or 0.5)
    prob_under = float(pred_ou25.get("prob_under", 0.5) or 0.5)
    predicted_outcome = pred_ou25.get("predicted_outcome", "Under")
    
    # Donut chart
    fig = go.Figure(data=[go.Pie(
        labels=['Over 2.5', 'Under 2.5'],
        values=[prob_over, prob_under],
        hole=0.4,
        marker_colors=['#ff6b6b', '#4ecdc4']
    )])
    
    fig.update_layout(
        title="Goals O/U 2.5",
        showlegend=True,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric(
        label="Predicted",
        value=predicted_outcome,
        delta=f"{max(prob_over, prob_under):.1%}"
    )


def _render_btts_prediction(pred_btts):
    st.markdown("### ⚽⚽ Both Teams to Score")
    
    if not pred_btts:
        st.warning("No BTTS prediction available")
        return
    
    prob_yes = float(pred_btts.get("prob_yes", 0.5) or 0.5)
    prob_no = float(pred_btts.get("prob_no", 0.5) or 0.5)
    predicted_outcome = pred_btts.get("predicted_outcome", "No")
    
    # Donut chart
    fig = go.Figure(data=[go.Pie(
        labels=['Yes', 'No'],
        values=[prob_yes, prob_no],
        hole=0.4,
        marker_colors=['#95e1d3', '#f38ba8']
    )])
    
    fig.update_layout(
        title="BTTS Probability",
        showlegend=True,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric(
        label="Predicted",
        value=predicted_outcome,
        delta=f"{max(prob_yes, prob_no):.1%}"
    )


def _render_fii_score(fii):
    st.markdown("### 🧠 Football Intelligence Index")
    
    score = float(fii.get("score", 0) or 0)
    confidence_level = fii.get("confidence_level", "Medium")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "FII Score"},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgray"},
                    {'range': [3, 7], 'color': "gray"},
                    {'range': [7, 10], 'color': "lightgreen"}
                ]
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("FII Score", f"{score:.2f}", confidence_level)


def _render_hybrid_breakdown(pred_1x2):
    st.markdown("### 🎯 Hybrid Model Breakdown")
    
    hybrid_sources = pred_1x2.get("hybrid_sources", {}) or {}
    weights_used = pred_1x2.get("weights_used", {}) or {}
    
    if not any([hybrid_sources, weights_used]):
        st.info("Hybrid breakdown not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if hybrid_sources:
            st.markdown("**Sources Used:**")
            for source, used in hybrid_sources.items():
                status = "✅" if used else "❌"
                st.text(f"{status} {source.replace('_', ' ').title()}")
    
    with col2:
        if weights_used:
            st.markdown("**Model Weights:**")
            
            labels = list(weights_used.keys())
            values = list(weights_used.values())
            
            fig = go.Figure(data=[go.Pie(
                labels=[label.replace('_', ' ').title() for label in labels],
                values=values,
                textinfo='label+percent'
            )])
            
            fig.update_layout(
                title="Model Contribution",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
