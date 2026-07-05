import pandas as pd
import numpy as np
import joblib
from collections import Counter
import os

# Set seed for reproducibility
np.random.seed(42)

def load_data():
    team_features = pd.read_csv('data/processed/team_features.csv')
    group_fixtures = pd.read_csv('data/group_fixtures.csv')
    knockout_slots = pd.read_csv('data/knockout_slots.csv')
    
    # Mapping for inconsistent names
    name_map = {
        'USA': 'United States',
        'Cabo Verde': 'Cape Verde',
        "Côte d'Ivoire": "Ivory Coast"
    }
    team_features['team'] = team_features['team'].replace(name_map)
    
    # Default stats for playoff teams or missing teams
    default_stats = {
        'elo_rating': 1500,
        'avg_goals_scored': 1.2,
        'avg_goals_conceded': 1.2
    }
    
    elo_dict = team_features.set_index('team')['elo_rating'].to_dict()
    
    # Handle playoff teams specifically if needed
    for team in ['UEFA Playoff A', 'UEFA Playoff B', 'UEFA Playoff C', 'UEFA Playoff D']:
        if team not in elo_dict:
            elo_dict[team] = 1600
    for team in ['FIFA Playoff 1', 'FIFA Playoff 2']:
        if team not in elo_dict:
            elo_dict[team] = 1500
            
    return team_features, group_fixtures, knockout_slots, elo_dict, default_stats

def simulate_match(home_team, away_team, elo_dict, default_stats, is_knockout=False):
    home_elo = elo_dict.get(home_team, default_stats['elo_rating'])
    away_elo = elo_dict.get(away_team, default_stats['elo_rating'])
    
    home_lambda = max(0.1, 1.2 + (home_elo - away_elo) / 400.0)
    away_lambda = max(0.1, 1.0 + (away_elo - home_elo) / 400.0)
    
    home_goals = np.random.poisson(home_lambda)
    away_goals = np.random.poisson(away_lambda)
    
    penalties = False
    winner = None
    
    if home_goals > away_goals:
        winner = 'home'
    elif away_goals > home_goals:
        winner = 'away'
    else:
        if is_knockout:
            # Extra Time
            et_home_goals = np.random.poisson(home_lambda * 0.5)
            et_away_goals = np.random.poisson(away_lambda * 0.5)
            home_goals += et_home_goals
            away_goals += et_away_goals
            
            if home_goals > away_goals:
                winner = 'home'
            elif away_goals > home_goals:
                winner = 'away'
            else:
                # Penalties
                penalties = True
                winner = 'home' if np.random.random() > 0.5 else 'away'
        else:
            winner = 'draw'
            
    return home_goals, away_goals, winner, penalties

def resolve_group_standings(matches_results, group_fixtures):
    standings = {}
    for match_id, result in matches_results.items():
        if match_id > 72: continue
        
        m = group_fixtures[group_fixtures['match_id'] == match_id].iloc[0]
        g = m['group']
        h = m['home_team']
        a = m['away_team']
        hg, ag = result['home_goals'], result['away_goals']
        
        if g not in standings: standings[g] = {}
        for team in [h, a]:
            if team not in standings[g]:
                standings[g][team] = {'pts': 0, 'gd': 0, 'gs': 0}
        
        standings[g][h]['gs'] += hg
        standings[g][h]['gd'] += (hg - ag)
        standings[g][a]['gs'] += ag
        standings[g][a]['gd'] += (ag - hg)
        
        if hg > ag: standings[g][h]['pts'] += 3
        elif ag > hg: standings[g][a]['pts'] += 3
        else:
            standings[g][h]['pts'] += 1
            standings[g][a]['pts'] += 1
            
    group_ranks = {}
    third_places = []
    
    for g, teams in standings.items():
        sorted_teams = sorted(teams.items(), key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gs']), reverse=True)
        group_ranks[g] = [t[0] for t in sorted_teams]
        # Store 3rd place team info for best 3rd comparison
        third_team = sorted_teams[2]
        third_places.append({
            'team': third_team[0],
            'group': g,
            'pts': third_team[1]['pts'],
            'gd': third_team[1]['gd'],
            'gs': third_team[1]['gs']
        })
        
    # Rank 3rd place teams
    best_thirds_sorted = sorted(third_places, key=lambda x: (x['pts'], x['gd'], x['gs']), reverse=True)
    
    return group_ranks, best_thirds_sorted

def get_best_third(best_thirds, available_groups, used_teams):
    for entry in best_thirds:
        if entry['group'] in available_groups and entry['team'] not in used_teams:
            return entry['team']
    # Fallback to any available if none match group criteria (shouldn't happen with correct slots)
    for entry in best_thirds:
        if entry['team'] not in used_teams:
            return entry['team']
    return None

def run_simulation(group_fixtures, knockout_slots, elo_dict, default_stats):
    results = {}
    # Simulate Group Stage
    for _, row in group_fixtures.iterrows():
        hg, ag, winner, pens = simulate_match(row['home_team'], row['away_team'], elo_dict, default_stats, False)
        results[row['match_id']] = {'home_goals': hg, 'away_goals': ag, 'winner': winner, 'penalties': pens}
        
    group_ranks, best_thirds = resolve_group_standings(results, group_fixtures)
    
    # Simulate Knockout Stage
    team_at_slot = {}
    # Fill group winners and runners-up
    for g, ranks in group_ranks.items():
        team_at_slot[f'Winner Group {g}'] = ranks[0]
        team_at_slot[f'Runner-up Group {g}'] = ranks[1]
    
    used_best_thirds = set()
    
    # Process knockout matches in order
    for _, row in knockout_slots.sort_values('match_id').iterrows():
        mid = row['match_id']
        slot_h = row['slot_home']
        slot_a = row['slot_away']
        
        def resolve_slot(slot):
            if slot in team_at_slot:
                return team_at_slot[slot]
            if "Best 3rd" in slot:
                import re
                groups = re.findall(r'Groups ([A-L/]+)', slot)
                if groups:
                    group_list = groups[0].split('/')
                    # Standardize group list (e.g. A/B/C/D/F)
                    t = get_best_third(best_thirds, group_list, used_best_thirds)
                    used_best_thirds.add(t)
                    return t
                return None
            if "Winner Match" in slot:
                match_num = int(slot.split('Match ')[1])
                return results[match_num]['winner']
            if "Loser Match" in slot:
                match_num = int(slot.split('Match ')[1])
                # Find loser
                m_res = results[match_num]
                # We need to know who played in that match to find the loser
                # This is tricky because we only stored winner. 
                # Let's refine how we store knockout match participants.
                return None # Will handle below
            return None

        h_team = resolve_slot(slot_h)
        a_team = resolve_slot(slot_a)
        
        # Handle Loser Match specially
        if "Loser Match" in slot_h:
            mn = int(slot_h.split('Match ')[1])
            # To find loser, we need to know participants of match mn
            # Participants are h_team and a_team of match mn
            # But we didn't store them. Let's fix that.
            pass

        # Wait, let's restructure to keep track of participants
        
    # Redoing run_simulation with better state tracking
    pass

def run_simulation_v2(group_fixtures, knockout_slots, elo_dict, default_stats):
    results = {}
    match_participants = {}
    
    # Group Stage
    for _, row in group_fixtures.iterrows():
        mid = row['match_id']
        h, a = row['home_team'], row['away_team']
        hg, ag, winner, pens = simulate_match(h, a, elo_dict, default_stats, False)
        results[mid] = {'home_goals': hg, 'away_goals': ag, 'winner': winner, 'penalties': pens}
        match_participants[mid] = (h, a)
        
    group_ranks, best_thirds = resolve_group_standings(results, group_fixtures)
    
    team_at_slot = {}
    for g, ranks in group_ranks.items():
        team_at_slot[f'Winner Group {g}'] = ranks[0]
        team_at_slot[f'Runner-up Group {g}'] = ranks[1]
    
    used_best_thirds = set()
    
    for _, row in knockout_slots.sort_values('match_id').iterrows():
        mid = row['match_id']
        
        def resolve_slot(slot):
            if slot in team_at_slot: return team_at_slot[slot]
            if "Best 3rd" in slot:
                import re
                groups_part = slot.split('Groups ')[1].strip('()')
                group_list = groups_part.split('/')
                t = get_best_third(best_thirds, group_list, used_best_thirds)
                used_best_thirds.add(t)
                return t
            if "Winner Match" in slot:
                mn = int(slot.split('Match ')[1])
                # Result winner is 'home' or 'away'
                p_mn = match_participants[mn]
                w_mn = results[mn]['winner']
                return p_mn[0] if w_mn == 'home' else p_mn[1]
            if "Loser Match" in slot:
                mn = int(slot.split('Match ')[1])
                p_mn = match_participants[mn]
                w_mn = results[mn]['winner']
                return p_mn[1] if w_mn == 'home' else p_mn[0]
            return None

        h_team = resolve_slot(row['slot_home'])
        a_team = resolve_slot(row['slot_away'])
        
        hg, ag, winner, pens = simulate_match(h_team, a_team, elo_dict, default_stats, True)
        results[mid] = {'home_goals': hg, 'away_goals': ag, 'winner': winner, 'penalties': pens}
        match_participants[mid] = (h_team, a_team)
        
    return results, match_participants

def main():
    team_features, group_fixtures, knockout_slots, elo_dict, default_stats = load_data()
    
    # Load XGBoost models
    model_corners = joblib.load('models/xgb_corners.joblib')
    model_yellow = joblib.load('models/xgb_yellow_cards.joblib')
    model_red = joblib.load('models/xgb_red_cards.joblib')
    
    iterations = 1000
    all_results = []
    
    for i in range(iterations):
        res, participants = run_simulation_v2(group_fixtures, knockout_slots, elo_dict, default_stats)
        iter_res = []
        for mid in range(1, 105):
            r = res[mid]
            p = participants[mid]
            iter_res.append({
                'match_id': mid,
                'home_goals': r['home_goals'],
                'away_goals': r['away_goals'],
                'winner': r['winner'],
                'penalties': r['penalties'],
                'h_team': p[0],
                'a_team': p[1]
            })
        all_results.append(iter_res)
        if (i+1) % 100 == 0:
            print(f"Simulated {i+1} iterations...")
            
    # Aggregate results
    final_rows = []
    for mid in range(1, 105):
        # Create list of result tuples
        m_results = [(res[mid-1]['home_goals'], res[mid-1]['away_goals'], 
                      res[mid-1]['winner'], res[mid-1]['penalties']) for res in all_results]
        
        # Find most common full result tuple
        mode_result = Counter(m_results).most_common(1)[0][0]
        mode_h_goals, mode_a_goals, mode_winner, mode_penalties = mode_result
        
        # For participants, find most common tuple
        m_participants = [(res[mid-1]['h_team'], res[mid-1]['a_team']) for res in all_results]
        most_common_p = Counter(m_participants).most_common(1)[0][0]
        h_team, a_team = most_common_p
        
        h_elo = elo_dict.get(h_team, default_stats['elo_rating'])
        a_elo = elo_dict.get(a_team, default_stats['elo_rating'])
        elo_diff = h_elo - a_elo
        
        input_df = pd.DataFrame([[elo_diff]], columns=['elo_diff'])
        corners = model_corners.predict(input_df)[0]
        yellow = model_yellow.predict(input_df)[0]
        red = model_red.predict(input_df)[0]
        
        final_rows.append({
            'match_id': mid,
            'predicted_home_goals': int(mode_h_goals),
            'predicted_away_goals': int(mode_a_goals),
            'corners': int(round(corners)),
            'yellow_cards': int(round(yellow)),
            'red_cards': int(round(red)),
            'match_winner': mode_winner,
            'penalties': bool(mode_penalties)
        })
        
    df_out = pd.DataFrame(final_rows)
    os.makedirs('submissions', exist_ok=True)
    df_out.to_csv('submissions/monte_carlo_submission.csv', index=False)
    print("Saved submission to submissions/monte_carlo_submission.csv")

if __name__ == "__main__":
    main()
