import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from api_client import FootballAPIClient


def render(client: FootballAPIClient):
    st.header("🎯 Scoreline Lab")
    
    st.markdown("""
    ### Detailed Scoreline Analysis
    
    Analyze exact scoreline probabilities using our advanced prediction engine.
    """)
    
    # Input form
    with st.form("scoreline_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            home_team = st.text_input("Home Team", placeholder="e.g. Manchester City")
        
        with col2:
            away_team = st.text_input("Away Team", placeholder="e.g. Liverpool")
        
        with col3:
            # League selection
            leagues_resp = client.get_leagues()
            if leagues_resp["ok"]:
                leagues_data = leagues_resp["data"]
                leagues = leagues_data.get("leagues", [])
                
                if leagues and isinstance(leagues[0], dict):
                    league_options = [""] + [f"{lg.get('name', lg.get('slug', str(lg)))}" for lg in leagues]
                elif leagues:
                    league_options = [""] + leagues
                else:
                    league_options = [""]
            else:
                league_options = [""]
            
            league = st.selectbox("League", league_options)
        
        date = st.date_input("Match Date (optional)", value=None)
        
        submitted = st.form_submit_button("🎯 Analyze Scorelines", type="primary")
    
    if submitted and home_team and away_team:
        with st.spinner("Analyzing scoreline probabilities..."):
            date_str = date.isoformat() if date else None
            
            result = client.predict_improved(
                home_team=home_team,
                away_team=away_team,
                league=league if league else None,
                date=date_str
            )
            
            if not result["ok"]:
                st.error(f"❌ Analysis failed: {result['error']}")
                return
            
            data = result["data"]
            
            st.subheader(f"🎯 Scoreline Analysis: {home_team} vs {away_team}")
            
            # Look for scoreline data in various places
            scoreline_data = None
            
            # Check multiple possible locations
            if "scoreline_matrix" in data:
                scoreline_data = data["scoreline_matrix"]
            elif "scoreline" in data:
                scoreline_data = data["scoreline"]
            elif "correct_score" in data:
                scoreline_data = data["correct_score"]
            elif "1x2_scoreline" in data:
                scoreline_data = data["1x2_scoreline"]
            else:
                # Check inside prediction_1x2
                pred_1x2 = data.get("prediction_1x2", {}) or {}
                if "scoreline" in pred_1x2:
                    scoreline_data = pred_1x2["scoreline"]
                elif "scoreline_matrix" in pred_1x2:
                    scoreline_data = pred_1x2["scoreline_matrix"]
            
            if scoreline_data and isinstance(scoreline_data, dict):
                _render_scoreline_analysis(scoreline_data, home_team, away_team)
            else:
                _render_no_scoreline_data(data, home_team, away_team)


def _render_scoreline_analysis(scoreline_data, home_team, away_team):
    """Render full scoreline analysis with heatmap and top scores"""
    
    st.success("✅ Scoreline data available!")
    
    # Extract matrix if nested
    if "matrix" in scoreline_data:
        matrix = scoreline_data["matrix"]
    else:
        matrix = scoreline_data
    
    if not isinstance(matrix, dict):
        st.warning("Scoreline data format not recognized")
        return
    
    # Create heatmap
    st.subheader("🔥 Scoreline Probability Heatmap")
    
    max_goals = 4
    heatmap_data = [[0 for _ in range(max_goals + 1)] for _ in range(max_goals + 1)]
    heatmap_text = [["" for _ in range(max_goals + 1)] for _ in range(max_goals + 1)]
    
    for scoreline, prob in matrix.items():
        try:
            if isinstance(prob, (int, float)) and "-" in str(scoreline):
                home_goals, away_goals = map(int, str(scoreline).split('-'))
                if home_goals <= max_goals and away_goals <= max_goals:
                    heatmap_data[home_goals][away_goals] = float(prob)
                    heatmap_text[home_goals][away_goals] = f"{prob:.3f}"
        except:
            continue
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"Away {i}" for i in range(max_goals + 1)],
        y=[f"Home {i}" for i in range(max_goals + 1)],
        colorscale='Viridis',
        text=heatmap_text,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"Scoreline Probability Matrix: {home_team} vs {away_team}",
        xaxis_title="Away Team Goals",
        yaxis_title="Home Team Goals",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top scorelines table
    st.subheader("📊 Most Probable Scorelines")
    
    # Sort scorelines by probability
    sorted_scores = []
    for scoreline, prob in matrix.items():
        if isinstance(prob, (int, float)):
            sorted_scores.append((str(scoreline), float(prob)))
    
    sorted_scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = sorted_scores[:10]
    
    if top_scores:
        df = pd.DataFrame([
            {"Rank": i+1, "Scoreline": score, "Probability": f"{prob:.1%}"}
            for i, (score, prob) in enumerate(top_scores)
        ])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Derived metrics
    st.subheader("📈 Derived Metrics from Scoreline")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate derived probabilities
    p_home = sum(prob for (score, prob) in sorted_scores if score.split('-')[0] > score.split('-')[1])
    p_draw = sum(prob for (score, prob) in sorted_scores if score.split('-')[0] == score.split('-')[1])
    p_away = sum(prob for (score, prob) in sorted_scores if score.split('-')[0] < score.split('-')[1])
    
    with col1:
        st.metric("Home Win (from scoreline)", f"{p_home:.1%}")
    
    with col2:
        st.metric("Draw (from scoreline)", f"{p_draw:.1%}")
    
    with col3:
        st.metric("Away Win (from scoreline)", f"{p_away:.1%}")


def _render_no_scoreline_data(data, home_team, away_team):
    """Render alternative analysis when scoreline data is not available"""
    
    st.warning("⚠️ Scoreline data not available from backend")
    
    st.markdown("""
    **Scoreline analysis requires:**
    - Scoreline prediction engine to be active
    - Historical match data for both teams
    - Sufficient data quality for matrix generation
    """)
    
    # Show available predictions instead
    st.subheader("📊 Available Predictions")
    
    pred_1x2 = data.get("prediction_1x2", {}) or {}
    pred_ou25 = data.get("prediction_ou25", {}) or {}
    pred_btts = data.get("prediction_btts", {}) or {}
    
    if pred_1x2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prob_home = float(pred_1x2.get("prob_home_win", 0) or 0)
            st.metric("Home Win", f"{prob_home:.1%}")
        
        with col2:
            prob_draw = float(pred_1x2.get("prob_draw", 0) or 0)
            st.metric("Draw", f"{prob_draw:.1%}")
        
        with col3:
            prob_away = float(pred_1x2.get("prob_away_win", 0) or 0)
            st.metric("Away Win", f"{prob_away:.1%}")
    
    if pred_ou25:
        st.subheader("⚽ Goals Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            prob_over = float(pred_ou25.get("prob_over", 0) or 0)
            st.metric("Over 2.5 Goals", f"{prob_over:.1%}")
        
        with col2:
            prob_under = float(pred_ou25.get("prob_under", 0) or 0)
            st.metric("Under 2.5 Goals", f"{prob_under:.1%}")
    
    if pred_btts:
        st.subheader("🥅 Both Teams to Score")
        col1, col2 = st.columns(2)
        
        with col1:
            prob_yes = float(pred_btts.get("prob_yes", 0) or 0)
            st.metric("BTTS Yes", f"{prob_yes:.1%}")
        
        with col2:
            prob_no = float(pred_btts.get("prob_no", 0) or 0)
            st.metric("BTTS No", f"{prob_no:.1%}")
    
    # Raw data for debugging
    with st.expander("🔍 Raw Response (for debugging)"):
        st.json(data)
