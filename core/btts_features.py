#!/usr/bin/env python3
"""
BTTS-Specific Feature Engineering
Създава специализирани features за подобряване на BTTS модела
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from core.utils import setup_logging


class BTTSFeatureEngineer:
    """BTTS-специфичен feature engineering"""
    
    def __init__(self):
        self.logger = setup_logging()
        
    def create_btts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Създава BTTS-специфични features
        
        Args:
            df: DataFrame с базови features
            
        Returns:
            DataFrame с добавени BTTS features
        """
        df_enhanced = df.copy()
        
        # 1. Исторически BTTS процент
        df_enhanced = self._add_historical_btts_features(df_enhanced)
        
        # 2. League-level BTTS поведение
        df_enhanced = self._add_league_btts_features(df_enhanced)
        
        # 3. Комбинирани BTTS features
        df_enhanced = self._add_combined_btts_features(df_enhanced)
        
        # 4. Match-up features
        df_enhanced = self._add_matchup_features(df_enhanced)
        
        # 5. Advanced BTTS indicators
        df_enhanced = self._add_advanced_btts_indicators(df_enhanced)
        
        self.logger.info(f"✓ BTTS features добавени: {len(df_enhanced.columns) - len(df.columns)} нови features")
        
        return df_enhanced
    
    def _add_historical_btts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавя исторически BTTS features"""
        
        # Симулираме исторически BTTS данни (в реалност би трябвало да се изчисли от исторически мачове)
        # За демонстрация използваме приблизителни стойности базирани на съществуващи features
        
        # BTTS rate базиран на голове и ефективност
        home_attack_strength = df.get('home_goals_scored_avg_5', 1.5) * df.get('home_shooting_efficiency', 0.3)
        away_attack_strength = df.get('away_goals_scored_avg_5', 1.5) * df.get('away_shooting_efficiency', 0.3)
        
        home_defense_weakness = df.get('home_goals_conceded_avg_5', 1.5) / (df.get('home_goals_scored_avg_5', 1.5) + 1)
        away_defense_weakness = df.get('away_goals_conceded_avg_5', 1.5) / (df.get('away_goals_scored_avg_5', 1.5) + 1)
        
        # Приблизителни BTTS rates
        df['home_btts_rate_last5'] = np.clip(home_attack_strength * 0.4 + home_defense_weakness * 0.3, 0.1, 0.9)
        df['away_btts_rate_last5'] = np.clip(away_attack_strength * 0.4 + away_defense_weakness * 0.3, 0.1, 0.9)
        
        # 10-match averages (по-консервативни)
        home_attack_10 = df.get('home_goals_scored_avg_10', 1.4) * df.get('home_shooting_efficiency', 0.3)
        away_attack_10 = df.get('away_goals_scored_avg_10', 1.4) * df.get('away_shooting_efficiency', 0.3)
        
        df['home_btts_rate_last10'] = np.clip(home_attack_10 * 0.35 + home_defense_weakness * 0.25, 0.15, 0.85)
        df['away_btts_rate_last10'] = np.clip(away_attack_10 * 0.35 + away_defense_weakness * 0.25, 0.15, 0.85)
        
        return df
    
    def _add_league_btts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавя league-level BTTS features"""
        
        # League BTTS rate (симулиран базиран на лига характеристики)
        # В реалност би трябвало да се изчисли от исторически данни по лиги
        
        # Използваме league_id ако е налично, иначе default стойности
        if 'league_id' in df.columns:
            # Симулираме league BTTS rates
            league_btts_map = {
                1: 0.52,    # Premier League - високо BTTS
                2: 0.48,    # La Liga - средно BTTS
                3: 0.45,    # Serie A - по-ниско BTTS (по-дефанзивна)
                4: 0.58,    # Bundesliga - много високо BTTS
                5: 0.46,    # Ligue 1 - средно-ниско BTTS
            }
            
            df['league_btts_rate'] = df['league_id'].map(league_btts_map).fillna(0.50)
            
            # League over 2.5 rate (помага за разграничаване на много голове vs BTTS)
            league_over25_map = {
                1: 0.55,    # Premier League
                2: 0.52,    # La Liga  
                3: 0.48,    # Serie A
                4: 0.62,    # Bundesliga
                5: 0.49,    # Ligue 1
            }
            
            df['league_over25_rate'] = df['league_id'].map(league_over25_map).fillna(0.53)
        else:
            # Default стойности
            df['league_btts_rate'] = 0.50
            df['league_over25_rate'] = 0.53
        
        return df
    
    def _add_combined_btts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавя комбинирани BTTS features"""
        
        # Total goals features
        df['total_goals_scored_avg_5'] = (
            df.get('home_goals_scored_avg_5', 1.5) + 
            df.get('away_goals_scored_avg_5', 1.5)
        )
        
        df['total_goals_conceded_avg_5'] = (
            df.get('home_goals_conceded_avg_5', 1.5) + 
            df.get('away_goals_conceded_avg_5', 1.5)
        )
        
        # Both defenses weak indicator
        home_def_weak = df.get('home_goals_conceded_avg_5', 1.5) > 1.3
        away_def_weak = df.get('away_goals_conceded_avg_5', 1.5) > 1.3
        df['both_defenses_weak'] = (home_def_weak & away_def_weak).astype(int)
        
        # Both attacks strong indicator  
        home_att_strong = df.get('home_goals_scored_avg_5', 1.5) > 1.7
        away_att_strong = df.get('away_goals_scored_avg_5', 1.5) > 1.7
        df['both_attacks_strong'] = (home_att_strong & away_att_strong).astype(int)
        
        # Defensive vulnerability product
        df['both_defenses_weak_product'] = (
            df.get('home_goals_conceded_avg_5', 1.5) * 
            df.get('away_goals_conceded_avg_5', 1.5)
        )
        
        # Attacking strength product
        df['both_attacks_strong_product'] = (
            df.get('home_goals_scored_avg_5', 1.5) * 
            df.get('away_goals_scored_avg_5', 1.5)
        )
        
        return df
    
    def _add_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавя match-up specific features"""
        
        # Attack vs Defense matchups
        df['attack_vs_defense_home'] = (
            df.get('home_goals_scored_avg_5', 1.5) * 
            df.get('away_goals_conceded_avg_5', 1.5)
        )
        
        df['attack_vs_defense_away'] = (
            df.get('away_goals_scored_avg_5', 1.5) * 
            df.get('home_goals_conceded_avg_5', 1.5)
        )
        
        # Expected goals from matchup
        df['expected_home_goals_matchup'] = np.clip(df['attack_vs_defense_home'] * 0.7, 0.2, 4.0)
        df['expected_away_goals_matchup'] = np.clip(df['attack_vs_defense_away'] * 0.7, 0.2, 4.0)
        
        # BTTS likelihood from matchup
        df['btts_likelihood_matchup'] = np.clip(
            (df['expected_home_goals_matchup'] * df['expected_away_goals_matchup']) ** 0.5 * 0.6,
            0.1, 0.9
        )
        
        return df
    
    def _add_advanced_btts_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавя advanced BTTS indicators"""
        
        # Shooting efficiency differential (важно за BTTS)
        home_eff = df.get('home_shooting_efficiency', 0.3)
        away_eff = df.get('away_shooting_efficiency', 0.3)
        
        df['shooting_efficiency_balance'] = 1 - abs(home_eff - away_eff)
        df['min_shooting_efficiency'] = np.minimum(home_eff, away_eff)
        df['max_shooting_efficiency'] = np.maximum(home_eff, away_eff)
        
        # Form differential impact on BTTS
        home_form = df.get('home_form_5', 0.5)
        away_form = df.get('away_form_5', 0.5)
        
        df['form_balance'] = 1 - abs(home_form - away_form)
        df['min_form'] = np.minimum(home_form, away_form)
        df['both_teams_good_form'] = ((home_form > 0.6) & (away_form > 0.6)).astype(int)
        
        # xG proxy balance
        home_xg = df.get('home_xg_proxy', 1.5)
        away_xg = df.get('away_xg_proxy', 1.5)
        
        df['xg_balance'] = 1 - abs(home_xg - away_xg) / (home_xg + away_xg + 0.1)
        df['min_xg_proxy'] = np.minimum(home_xg, away_xg)
        df['both_teams_attacking'] = ((home_xg > 1.8) & (away_xg > 1.8)).astype(int)
        
        # Poisson BTTS confidence (ако е налично)
        if 'poisson_prob_btts' in df.columns:
            poisson_btts = df['poisson_prob_btts']
            df['poisson_btts_confidence'] = 2 * abs(poisson_btts - 0.5)
            df['poisson_btts_extreme'] = ((poisson_btts < 0.3) | (poisson_btts > 0.7)).astype(int)
        else:
            df['poisson_btts_confidence'] = 0.0
            df['poisson_btts_extreme'] = 0
        
        # Composite BTTS indicators
        df['btts_favorable_conditions'] = (
            df['both_defenses_weak'] + 
            df['both_attacks_strong'] + 
            df['both_teams_good_form'] + 
            df['both_teams_attacking']
        )
        
        # BTTS risk factors (мачове където е малко вероятно BTTS)
        df['btts_risk_factors'] = (
            (df['min_shooting_efficiency'] < 0.2).astype(int) +
            (df['min_form'] < 0.3).astype(int) +
            (df['min_xg_proxy'] < 1.0).astype(int) +
            (df['total_goals_scored_avg_5'] < 2.5).astype(int)
        )
        
        return df
    
    def get_btts_feature_list(self) -> List[str]:
        """Връща списък с всички BTTS-специфични features"""
        
        btts_features = [
            # Historical BTTS
            'home_btts_rate_last5',
            'away_btts_rate_last5', 
            'home_btts_rate_last10',
            'away_btts_rate_last10',
            
            # League BTTS
            'league_btts_rate',
            'league_over25_rate',
            
            # Combined features
            'total_goals_scored_avg_5',
            'total_goals_conceded_avg_5',
            'both_defenses_weak',
            'both_attacks_strong',
            'both_defenses_weak_product',
            'both_attacks_strong_product',
            
            # Matchup features
            'attack_vs_defense_home',
            'attack_vs_defense_away',
            'expected_home_goals_matchup',
            'expected_away_goals_matchup',
            'btts_likelihood_matchup',
            
            # Advanced indicators
            'shooting_efficiency_balance',
            'min_shooting_efficiency',
            'max_shooting_efficiency',
            'form_balance',
            'min_form',
            'both_teams_good_form',
            'xg_balance',
            'min_xg_proxy',
            'both_teams_attacking',
            'poisson_btts_confidence',
            'poisson_btts_extreme',
            'btts_favorable_conditions',
            'btts_risk_factors'
        ]
        
        return btts_features


def main():
    """Тест на BTTS feature engineering"""
    logger = setup_logging()
    
    logger.info("🔧 ТЕСТВАНЕ НА BTTS FEATURE ENGINEERING")
    logger.info("=" * 50)
    
    # Създава тестови данни
    test_data = {
        'home_goals_scored_avg_5': [1.8, 2.2, 1.2, 2.5],
        'away_goals_scored_avg_5': [1.5, 1.9, 0.8, 2.1],
        'home_goals_conceded_avg_5': [1.2, 1.8, 0.9, 2.0],
        'away_goals_conceded_avg_5': [1.4, 1.6, 1.1, 1.9],
        'home_shooting_efficiency': [0.35, 0.28, 0.42, 0.31],
        'away_shooting_efficiency': [0.32, 0.25, 0.38, 0.29],
        'home_form_5': [0.7, 0.4, 0.8, 0.6],
        'away_form_5': [0.6, 0.5, 0.3, 0.7],
        'home_xg_proxy': [1.9, 2.1, 1.3, 2.4],
        'away_xg_proxy': [1.7, 1.8, 1.0, 2.2],
        'league_id': [1, 2, 3, 4],
        'poisson_prob_btts': [0.65, 0.45, 0.35, 0.75]
    }
    
    df = pd.DataFrame(test_data)
    logger.info(f"Тестови данни: {len(df)} мача, {len(df.columns)} базови features")
    
    # Прилага feature engineering
    engineer = BTTSFeatureEngineer()
    df_enhanced = engineer.create_btts_features(df)
    
    logger.info(f"Подобрени данни: {len(df_enhanced.columns)} общо features")
    
    # Показва нови features
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    logger.info(f"Нови BTTS features ({len(new_features)}):")
    for feature in new_features:
        logger.info(f"  - {feature}")
    
    # Показва примерни стойности
    logger.info("\n📊 ПРИМЕРНИ СТОЙНОСТИ:")
    sample_features = ['home_btts_rate_last5', 'btts_likelihood_matchup', 'btts_favorable_conditions', 'btts_risk_factors']
    for feature in sample_features:
        values = df_enhanced[feature].values
        logger.info(f"  {feature}: {values}")
    
    logger.info("\n✅ BTTS feature engineering тест завършен успешно!")


if __name__ == "__main__":
    main()
