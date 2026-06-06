import pandas as pd
import os

def main():
    # Paths
    fixtures_path = 'data/raw/group_fixtures.csv'
    if not os.path.exists(fixtures_path):
        fixtures_path = 'data/group_fixtures.csv'
    
    predictions_path = 'submissions/group_stage_goal_predictions.csv'
    output_path = 'data/processed/group_standings.csv'
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1 & 2. Load data
    if not os.path.exists(fixtures_path):
        print(f"Error: Fixtures file not found at {fixtures_path}")
        return
    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found at {predictions_path}")
        return

    df_fixtures = pd.read_csv(fixtures_path)
    df_predictions = pd.read_csv(predictions_path)
    
    # 3. Merge on match_id
    df = pd.merge(df_fixtures, df_predictions, on='match_id')
    
    # 4. Calculate metrics for each match
    # Points: Win=3, Draw=1, Loss=0
    df['home_points'] = 0
    df['away_points'] = 0
    
    df.loc[df['predicted_home_goals'] > df['predicted_away_goals'], 'home_points'] = 3
    df.loc[df['predicted_home_goals'] < df['predicted_away_goals'], 'away_points'] = 3
    df.loc[df['predicted_home_goals'] == df['predicted_away_goals'], ['home_points', 'away_points']] = 1
    
    # 5. Aggregate metrics per team per group
    # We need to reshape to have one row per team per match
    home_stats = df[['group', 'home_team', 'predicted_home_goals', 'predicted_away_goals', 'home_points']].rename(
        columns={
            'home_team': 'team',
            'predicted_home_goals': 'goals_scored',
            'predicted_away_goals': 'goals_conceded',
            'home_points': 'points'
        }
    )
    
    away_stats = df[['group', 'away_team', 'predicted_away_goals', 'predicted_home_goals', 'away_points']].rename(
        columns={
            'away_team': 'team',
            'predicted_away_goals': 'goals_scored',
            'predicted_home_goals': 'goals_conceded',
            'away_points': 'points'
        }
    )
    
    all_stats = pd.concat([home_stats, away_stats])
    
    # Calculate goal difference
    all_stats['goal_difference'] = all_stats['goals_scored'] - all_stats['goals_conceded']
    
    # Aggregate
    standings = all_stats.groupby(['group', 'team']).agg({
        'points': 'sum',
        'goal_difference': 'sum',
        'goals_scored': 'sum'
    }).reset_index()
    
    # 6 & 7. Sort final DataFrame
    # Sort by: group (ascending), points (descending), goal_difference (descending), goals_scored (descending)
    standings = standings.sort_values(
        by=['group', 'points', 'goal_difference', 'goals_scored'],
        ascending=[True, False, False, False]
    )
    
    # 8. Save
    standings.to_csv(output_path, index=False)
    print(f"Standings successfully saved to {output_path}")

if __name__ == "__main__":
    main()
