import streamlit as st
import pandas as pd
from api_client import FootballAPIClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.normalize_predictions import normalize_predictions
from utils.league_formatter import format_leagues_for_display, get_league_display_name, group_leagues_by_category


def render(client: FootballAPIClient):
    st.header("🌍 League Explorer")
    
    # Get leagues
    leagues_resp = client.get_leagues()
    if not leagues_resp["ok"]:
        st.error(f"❌ Cannot load leagues: {leagues_resp['error']}")
        return
    
    leagues_data = leagues_resp["data"]
    leagues = leagues_data.get("leagues", [])
    
    if not leagues:
        st.warning("No leagues available")
        return
    
    # Format leagues with clean names
    formatted_leagues = format_leagues_for_display(leagues, group_by_category=False)
    
    # Group by category for display
    grouped_leagues = group_leagues_by_category(leagues)
    
    # League selection
    st.subheader("🏆 Select League")
    
    # Create tabs for categories
    if grouped_leagues:
        category_tabs = st.tabs(list(grouped_leagues.keys()))
        
        selected_leagues = []
        
        for tab, (category, category_leagues) in zip(category_tabs, grouped_leagues.items()):
            with tab:
                # Display leagues in grid (3 columns)
                cols = st.columns(3)
                
                for i, league in enumerate(category_leagues[:15]):  # Max 15 per category
                    slug = league.get("slug", "")
                    display_name = get_league_display_name(league)
                    
                    with cols[i % 3]:
                        if st.button(display_name, key=f"league_btn_{category}_{i}_{slug}", use_container_width=True):
                            selected_leagues.append(slug)
    
    # Or use searchable selectbox for all leagues
    st.markdown("---")
    st.subheader("🔍 Or Search All Leagues")
    
    # Create options for selectbox
    league_options = [(slug, display_name) for slug, display_name, _ in formatted_leagues]
    
    selected_league = st.selectbox(
        "Search and select:",
        options=[""] + [opt[0] for opt in league_options],
        format_func=lambda x: next((opt[1] for opt in league_options if opt[0] == x), x) if x else "Type to search..."
    )
    
    if selected_league:
        selected_leagues = [selected_league]
    
    # Filters
    if selected_leagues:
        st.subheader("🔍 Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
        
        with col2:
            result_filter = st.selectbox(
                "Filter by Predicted Result",
                ["All", "Home Win (1)", "Draw (X)", "Away Win (2)"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Date", "Confidence", "Home Team", "Away Team"]
            )
        
        # Load predictions for selected league
        for league_slug in selected_leagues:
            _load_league_predictions(client, league_slug, min_confidence, result_filter, sort_by)


def _load_league_predictions(client, league_slug, min_confidence, result_filter, sort_by):
    with st.spinner(f"Loading predictions for {league_slug}..."):
        result = client.predict_next_round(league_slug)
        
        if not result["ok"]:
            st.error(f"❌ Failed to load {league_slug}: {result['error']}")
            return
        
        data = result["data"]
        
        # Round info
        round_info = data.get("round_info", {}) or {}
        round_name = round_info.get("round", "Next Round")
        
        st.subheader(f"🏆 {league_slug.replace('_', ' ').title()} - {round_name}")
        
        matches = data.get("matches", [])
        if not matches:
            st.warning("No matches found")
            return
        
        # Process and filter matches
        filtered_matches = []
        
        for match in matches:
            predictions = match.get("predictions", {}) or {}
            normalized = normalize_predictions(predictions)
            pred_1x2 = normalized["pred_1x2"]
            
            predicted_outcome = pred_1x2.get("predicted_outcome", "X")
            confidence = float(pred_1x2.get("confidence", 0.33) or 0.33)
            
            # Apply filters
            if confidence < min_confidence:
                continue
            
            if result_filter != "All":
                if result_filter == "Home Win (1)" and predicted_outcome != "1":
                    continue
                elif result_filter == "Draw (X)" and predicted_outcome != "X":
                    continue
                elif result_filter == "Away Win (2)" and predicted_outcome != "2":
                    continue
            
            filtered_matches.append(match)
        
        if not filtered_matches:
            st.warning("No matches match the current filters")
            return
        
        # Create table data
        table_data = []
        
        for match in filtered_matches:
            date = match.get("date", "TBD")
            home_team = match.get("home_team", "Unknown")
            away_team = match.get("away_team", "Unknown")
            
            predictions = match.get("predictions", {}) or {}
            normalized = normalize_predictions(predictions)
            pred_1x2 = normalized["pred_1x2"]
            pred_ou25 = normalized["pred_ou25"]
            pred_btts = normalized["pred_btts"]
            
            # Extract predictions
            predicted_outcome_1x2 = pred_1x2.get("predicted_outcome", "X")
            confidence_1x2 = float(pred_1x2.get("confidence", 0.33) or 0.33)
            using_hybrid = pred_1x2.get("using_hybrid", False)
            
            predicted_outcome_ou25 = pred_ou25.get("predicted_outcome", "Under")
            predicted_outcome_btts = pred_btts.get("predicted_outcome", "No")
            
            table_data.append({
                "Date": date,
                "Home": home_team,
                "Away": away_team,
                "1X2": predicted_outcome_1x2,
                "1X2 Confidence": confidence_1x2,
                "OU2.5": predicted_outcome_ou25,
                "BTTS": predicted_outcome_btts,
                "Hybrid": "🎯" if using_hybrid else "🤖",
                "_sort_date": date,
                "_sort_home": home_team,
                "_sort_away": away_team
            })
        
        df = pd.DataFrame(table_data)
        
        # Apply sorting
        if sort_by == "Date":
            df = df.sort_values("_sort_date")
        elif sort_by == "Confidence":
            df = df.sort_values("1X2 Confidence", ascending=False)
        elif sort_by == "Home Team":
            df = df.sort_values("_sort_home")
        elif sort_by == "Away Team":
            df = df.sort_values("_sort_away")
        
        # Remove sort columns
        display_df = df.drop(columns=[col for col in df.columns if col.startswith("_sort")])
        
        # Style based on confidence
        def highlight_confidence(val):
            if isinstance(val, (int, float)):
                if val >= 0.7:
                    return 'background-color: #d4edda'
                elif val >= 0.5:
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            return ''
        
        styled_df = display_df.style.applymap(highlight_confidence, subset=['1X2 Confidence'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(df))
        
        with col2:
            avg_confidence = df['1X2 Confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            high_conf = (df['1X2 Confidence'] >= 0.7).sum()
            st.metric("High Confidence", high_conf)
        
        with col4:
            hybrid_count = (df['Hybrid'] == '🎯').sum()
            st.metric("Hybrid Used", hybrid_count)
        
        st.markdown("---")
