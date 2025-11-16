import streamlit as st
import pandas as pd
from api_client import FootballAPIClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.normalize_predictions import normalize_predictions
from utils.league_formatter import format_leagues_for_display


def render(client: FootballAPIClient):
    st.header("📅 Next Round Predictions")
    
    # League selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        leagues_resp = client.get_leagues()
        if leagues_resp["ok"]:
            leagues_data = leagues_resp["data"]
            leagues = leagues_data.get("leagues", [])
            
            if leagues:
                # Format leagues with clean names
                formatted_leagues = format_leagues_for_display(leagues, group_by_category=False)
                league_options = [(slug, display_name) for slug, display_name, _ in formatted_leagues]
            else:
                league_options = [("premier_league", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League")]
        else:
            league_options = [("premier_league", "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League")]
        
        selected_league = st.selectbox(
            "Select League",
            options=[opt[0] for opt in league_options],
            format_func=lambda x: next((opt[1] for opt in league_options if opt[0] == x), x)
        )
    
    with col2:
        if st.button("🔄 Refresh", type="primary", key="next_round_refresh"):
            st.rerun()
    
    if selected_league:
        with st.spinner(f"Loading next round predictions..."):
            result = client.predict_next_round(selected_league)
            
            if not result["ok"]:
                st.error(f"❌ Failed to load predictions: {result['error']}")
                return
            
            data = result["data"]
            
            # Round info
            round_info = data.get("round_info", {}) or {}
            round_name = round_info.get("round", "Next Round")
            date_range = round_info.get("date_range", "")
            
            st.subheader(f"🏆 {round_name}")
            if date_range:
                st.info(f"📅 {date_range}")
            
            # Summary metrics
            matches = data.get("matches", [])
            successful_predictions = data.get("successful_predictions", 0)
            failed_predictions = data.get("failed_predictions", 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Matches", len(matches))
            
            with col2:
                st.metric("Successful Predictions", successful_predictions)
            
            with col3:
                st.metric("Failed Predictions", failed_predictions)
            
            # Matches table
            if matches:
                _render_matches_table(matches)
            else:
                st.warning("No matches found for this round")


def _render_matches_table(matches):
    st.markdown("### 📊 Matches & Predictions")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
    
    with col2:
        result_filter = st.selectbox(
            "Filter by Result",
            ["All", "Home Win (1)", "Draw (X)", "Away Win (2)"]
        )
    
    # Process matches data
    table_data = []
    
    for match in matches:
        date = match.get("date", "TBD")
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        
        predictions = match.get("predictions", {}) or {}
        normalized = normalize_predictions(predictions)
        pred_1x2 = normalized["pred_1x2"]
        pred_ou25 = normalized["pred_ou25"]
        pred_btts = normalized["pred_btts"]
        
        # 1X2 info
        predicted_outcome_1x2 = pred_1x2.get("predicted_outcome", "X")
        confidence_1x2 = float(pred_1x2.get("confidence", 0.33) or 0.33)
        using_hybrid = pred_1x2.get("using_hybrid", False)
        
        # OU2.5 info
        predicted_outcome_ou25 = pred_ou25.get("predicted_outcome", "Under")
        confidence_ou25 = float(pred_ou25.get("confidence", 0.5) or 0.5)
        
        # BTTS info
        predicted_outcome_btts = pred_btts.get("predicted_outcome", "No")
        confidence_btts = float(pred_btts.get("confidence", 0.5) or 0.5)
        
        # Apply filters
        if confidence_1x2 < min_confidence:
            continue
        
        if result_filter != "All":
            if result_filter == "Home Win (1)" and predicted_outcome_1x2 != "1":
                continue
            elif result_filter == "Draw (X)" and predicted_outcome_1x2 != "X":
                continue
            elif result_filter == "Away Win (2)" and predicted_outcome_1x2 != "2":
                continue
        
        table_data.append({
            "Date": date,
            "Home": home_team,
            "Away": away_team,
            "1X2": f"{predicted_outcome_1x2} ({confidence_1x2:.1%})",
            "OU2.5": f"{predicted_outcome_ou25} ({confidence_ou25:.1%})",
            "BTTS": f"{predicted_outcome_btts} ({confidence_btts:.1%})",
            "Hybrid": "🎯" if using_hybrid else "🤖",
            "Confidence": confidence_1x2
        })
    
    if not table_data:
        st.warning("No matches match the current filters")
        return
    
    df = pd.DataFrame(table_data)
    
    # Style the dataframe
    def highlight_confidence(val):
        if isinstance(val, (int, float)):
            if val >= 0.7:
                return 'background-color: #d4edda'  # Light green
            elif val >= 0.5:
                return 'background-color: #fff3cd'  # Light yellow
            else:
                return 'background-color: #f8d7da'  # Light red
        return ''
    
    styled_df = df.style.applymap(highlight_confidence, subset=['Confidence'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Summary stats
    if len(df) > 0:
        st.markdown("### 📈 Round Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_confidence = df['Confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col2:
            high_confidence = (df['Confidence'] >= 0.7).sum()
            st.metric("High Confidence Matches", high_confidence)
        
        with col3:
            hybrid_count = (df['Hybrid'] == '🎯').sum()
            st.metric("Using Hybrid Model", hybrid_count)
