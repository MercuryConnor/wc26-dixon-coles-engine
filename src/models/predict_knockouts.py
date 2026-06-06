import pandas as pd
import numpy as np
import os
import re

def load_data():
    # Load required dataframes
    standings = pd.read_csv('data/processed/group_standings.csv')
    features = pd.read_csv('data/processed/team_features.csv')
    slots = pd.read_csv('data/knockout_slots.csv')
    return standings, features, slots

def get_advancing_teams(standings):
    """
    Determine teams advancing from the group stage.
    - Top 2 from each group (A-L).
    - Top 8 3rd-place teams overall.
    """
    advancers = {}
    groups = sorted(standings['group'].unique())
    
    for g in groups:
        # Sort by points, goal_difference, goals_scored
        grp_teams = standings[standings['group'] == g].sort_values(
            ['points', 'goal_difference', 'goals_scored'], ascending=False
        )
        advancers[f'Winner Group {g}'] = grp_teams.iloc[0]['team']
        advancers[f'Runner-up Group {g}'] = grp_teams.iloc[1]['team']
    
    # Top 8 3rd-place teams overall
    third_place_candidates = []
    for g in groups:
        grp_teams = standings[standings['group'] == g].sort_values(
            ['points', 'goal_difference', 'goals_scored'], ascending=False
        )
        if len(grp_teams) >= 3:
            third_place_candidates.append(grp_teams.iloc[2])
    
    df_third = pd.DataFrame(third_place_candidates).sort_values(
        ['points', 'goal_difference', 'goals_scored'], ascending=False
    )
    best_third_teams = df_third.head(8)['team'].tolist()
    
    for i, team in enumerate(best_third_teams):
        # Index 1-based as requested
        advancers[f'Best 3rd-place {i+1}'] = team
        
    return advancers

def simulate_matches(slots, advancers, elo_dict):
    """
    Iterate through knockout matches and predict results.
    """
    match_winners = {}
    results = []
    
    # Map Best 3rd slots in knockout_slots.csv to our determined Best 3rd-place teams.
    # Instruction: Format 'Best 3rd-place X' -> pull from the sorted list of 8 best 3rd-place teams.
    # If the slot name is 'Best 3rd (Groups ...)', we map it in order of appearance.
    unique_third_slots = []
    for _, row in slots.iterrows():
        for s in [row['slot_home'], row['slot_away']]:
            if 'Best 3rd' in s and s not in unique_third_slots:
                unique_third_slots.append(s)
    
    for i, slot_name in enumerate(unique_third_slots):
        if i < 8:
            advancers[slot_name] = advancers.get(f'Best 3rd-place {i+1}')

    # Process matches in order of match_id
    for _, row in slots.sort_values('match_id').iterrows():
        match_id = int(row['match_id'])
        
        def resolve_team(slot):
            # 1. Check fixed group slots (Winner/Runner-up/Best 3rd)
            if slot in advancers:
                return advancers[slot]
            # 2. Check winner of previous match
            if 'Winner Match' in slot:
                m_id_match = re.search(r'\d+', slot)
                if m_id_match:
                    prev_match_id = int(m_id_match.group())
                    return match_winners.get(prev_match_id)
            return None

        home_team = resolve_team(row['slot_home'])
        away_team = resolve_team(row['slot_away'])

        # Elo heuristic prediction
        home_elo = elo_dict.get(home_team, 1500)
        away_elo = elo_dict.get(away_team, 1500)
        
        # Predicted home goals: round(max(0, 1.2 + (Home_Elo - Away_Elo) / 400))
        home_goals = int(round(max(0, 1.2 + (home_elo - away_elo) / 400)))
        # Predicted away goals: round(max(0, 1.0 + (Away_Elo - Home_Elo) / 400))
        away_goals = int(round(max(0, 1.0 + (away_elo - home_elo) / 400)))
        
        penalties = False
        if home_goals > away_goals:
            winner = home_team
        elif away_goals > home_goals:
            winner = away_team
        else:
            # Tie breaker: team with higher Elo, penalties = True
            winner = home_team if home_elo >= away_elo else away_team
            penalties = True
            
        match_winners[match_id] = winner
        
        results.append({
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_home_goals': home_goals,
            'predicted_away_goals': away_goals,
            'corners': 9,
            'yellow_cards': 3,
            'red_cards': 0,
            'match_winner': winner,
            'penalties': penalties
        })
        
    return results

def main():
    try:
        standings, features, slots = load_data()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Team Elo lookup
    elo_dict = dict(zip(features['team'], features['elo_rating']))
    
    # Get advancers from groups
    advancers = get_advancing_teams(standings)
    
    # Predict knockouts
    results = simulate_matches(slots, advancers, elo_dict)
    
    # Save output
    df_results = pd.DataFrame(results)
    os.makedirs('submissions', exist_ok=True)
    output_file = 'submissions/knockout_stage_predictions.csv'
    df_results.to_csv(output_file, index=False)
    print(f"Knockout predictions saved to {output_file}")

if __name__ == '__main__':
    main()
