from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

def predict_group_stage():
    # Paths
    features_path = PROCESSED_DIR / "team_features.csv"
    # Try both locations for fixtures as specified
    fixtures_path = DATA_DIR / "raw" / "group_fixtures.csv"
    if not fixtures_path.exists():
        fixtures_path = DATA_DIR / "group_fixtures.csv"

    output_path = SUBMISSIONS_DIR / "group_stage_goal_predictions.csv"

    # Load data
    try:
        df_features = pd.read_csv(features_path)
    except FileNotFoundError:
        print(f"Error: {features_path} not found.")
        return

    try:
        df_fixtures = pd.read_csv(fixtures_path)
    except FileNotFoundError:
        print(f"Error: {fixtures_path} not found in data/raw/ or data/.")
        return

    # Elo lookup map
    # Assumes 'team' and 'elo_rating' columns exist in team_features.csv
    elo_dict = dict(zip(df_features['team'], df_features['elo_rating']))

    predictions = []

    for _, row in df_fixtures.iterrows():
        match_id = row['match_id']
        home_team = row['home_team']
        away_team = row['away_team']

        # Default Elo to 1500 if missing
        home_elo = elo_dict.get(home_team, 1500)
        away_elo = elo_dict.get(away_team, 1500)

        # Predicted home goals: 1.2 + (Home_Elo - Away_Elo) / 400
        home_goals_pred = 1.2 + (home_elo - away_elo) / 400
        # Predicted away goals: 1.0 + (Away_Elo - Home_Elo) / 400
        away_goals_pred = 1.0 + (away_elo - home_elo) / 400

        # Round to nearest integer and ensure not negative
        home_goals = int(round(max(0, home_goals_pred)))
        away_goals = int(round(max(0, away_goals_pred)))

        predictions.append({
            'match_id': match_id,
            'predicted_home_goals': home_goals,
            'predicted_away_goals': away_goals
        })

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(predictions)
    df_out.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    predict_group_stage()
