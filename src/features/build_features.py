import pandas as pd
import numpy as np
from pathlib import Path

def calculate_elo(df, initial_rating=1500, k_factor=30):
    """
    Calculates Elo ratings for teams based on match results.
    """
    ratings = {}
    
    # Sort by date to ensure chronological processing
    df = df.sort_values('date')
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']
        
        # Initialize ratings if not present
        if home_team not in ratings:
            ratings[home_team] = initial_rating
        if away_team not in ratings:
            ratings[away_team] = initial_rating
            
        r_home = ratings[home_team]
        r_away = ratings[away_team]
        
        # Expected scores
        e_home = 1 / (1 + 10**((r_away - r_home) / 400))
        e_away = 1 / (1 + 10**((r_home - r_away) / 400))
        
        # Actual outcomes
        if home_score > away_score:
            w_home, w_away = 1, 0
        elif home_score < away_score:
            w_home, w_away = 0, 1
        else:
            w_home, w_away = 0.5, 0.5
            
        # Update ratings
        ratings[home_team] = r_home + k_factor * (w_home - e_home)
        ratings[away_team] = r_away + k_factor * (w_away - e_away)
        
    return ratings

def build_features():
    # Paths
    # Current file is in src/features/build_features.py
    # parents[2] gets us to the root: D:\Mrityunjay\Projects\FIfa2026
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / 'data' / 'raw' / 'results.csv'
    processed_data_dir = project_root / 'data' / 'processed'
    output_path = processed_data_dir / 'team_features.csv'
    
    # Load data
    if not raw_data_path.exists():
        print(f"Error: Raw data not found at {raw_data_path}")
        return
        
    df = pd.read_csv(raw_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for matches strictly after 2018-01-01
    df_filtered = df[df['date'] > '2018-01-01'].copy()
    
    if df_filtered.empty:
        print("Warning: No matches found after 2018-01-01")
        return

    # 1. Calculate Elo Ratings
    elo_ratings = calculate_elo(df_filtered)
    
    # 2. Calculate average goals scored and conceded
    # Home stats
    home_stats = df_filtered.groupby('home_team').agg({
        'home_score': ['sum', 'count'],
        'away_score': 'sum'
    })
    home_stats.columns = ['goals_scored_home', 'matches_home', 'goals_conceded_home']
    
    # Away stats
    away_stats = df_filtered.groupby('away_team').agg({
        'away_score': ['sum', 'count'],
        'home_score': 'sum'
    })
    away_stats.columns = ['goals_scored_away', 'matches_away', 'goals_conceded_away']
    
    # Combine stats
    teams = sorted(list(set(df_filtered['home_team'].unique()) | set(df_filtered['away_team'].unique())))
    team_stats = []
    
    for team in teams:
        # Use .get() or check index to avoid KeyError
        h_row = home_stats.loc[team] if team in home_stats.index else pd.Series(0, index=home_stats.columns)
        a_row = away_stats.loc[team] if team in away_stats.index else pd.Series(0, index=away_stats.columns)
        
        total_matches = h_row['matches_home'] + a_row['matches_away']
        if total_matches > 0:
            avg_scored = (h_row['goals_scored_home'] + a_row['goals_scored_away']) / total_matches
            avg_conceded = (h_row['goals_conceded_home'] + a_row['goals_conceded_away']) / total_matches
            
            team_stats.append({
                'team': team,
                'elo_rating': elo_ratings.get(team, 1500),
                'avg_goals_scored': avg_scored,
                'avg_goals_conceded': avg_conceded
            })
            
    features_df = pd.DataFrame(team_stats)
    
    # Ensure directory exists and save
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

if __name__ == '__main__':
    build_features()
