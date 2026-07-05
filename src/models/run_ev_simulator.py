import pandas as pd
import numpy as np
from scipy.stats import poisson
from collections import Counter
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

# Load data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

params_df = pd.read_csv(PROCESSED_DIR / "dc_parameters_regularized.csv")
team_features = pd.read_csv(PROCESSED_DIR / "team_features.csv")
group_fixtures = pd.read_csv(DATA_DIR / "group_fixtures.csv")
knockout_slots = pd.read_csv(DATA_DIR / "knockout_slots.csv")

# Mapping for quick lookup
team_params = params_df.set_index('team').to_dict('index')
team_elo = team_features.set_index('team')['elo_rating'].to_dict()

# Dixon-Coles tau adjustment
def tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def get_match_probs(home_team, away_team, max_goals=6):
    hp = team_params.get(home_team)
    ap = team_params.get(away_team)
    
    if not hp or not ap:
        # Default fallback if team not in parameters (shouldn't happen with regularized set)
        lam, mu, rho = 1.3, 1.1, 0.0
    else:
        # attack/defense are log-space, host_advantage is log-space, rho is raw
        lam = np.exp(hp['attack'] + ap['defense'] + hp['host_advantage'])
        mu = np.exp(ap['attack'] + hp['defense'])
        rho = hp['rho'] # Assuming rho is same for all as per CSV check
        
        # Apply 1.04 multiplier
        lam *= 1.04
        mu *= 1.04
        
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p_h = poisson.pmf(x, lam)
            p_a = poisson.pmf(y, mu)
            adj = tau(x, y, lam, mu, rho)
            probs[x, y] = max(0, adj * p_h * p_a)
            
    probs /= probs.sum()
    return probs, lam, mu

def calculate_ev_score(probs):
    # ABANDON EV HACKING. Use pure Maximum Likelihood (Most Probable Score)
    flat_probs = probs.flatten()
    best_idx = np.argmax(flat_probs)
    return np.unravel_index(best_idx, probs.shape)

def simulate_match(home_team, away_team, knockout=False):
    probs, lam, mu = get_match_probs(home_team, away_team)
    
    # Sample score
    flat_probs = probs.flatten()
    idx = np.random.choice(len(flat_probs), p=flat_probs)
    h_g, a_g = np.unravel_index(idx, probs.shape)
    
    winner = None
    loser = None
    if h_g > a_g:
        winner = home_team
        loser = away_team
    elif a_g > h_g:
        winner = away_team
        loser = home_team
    elif knockout:
        # Penalty shootout if draw in knockout
        elo_h = team_elo.get(home_team, 1500)
        elo_a = team_elo.get(away_team, 1500)
        p_h_win = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        if np.random.random() < p_h_win:
            winner = home_team
            loser = away_team
        else:
            winner = away_team
            loser = home_team
            
    return h_g, a_g, winner, loser

def resolve_group_standings(matches):
    # matches: list of (home, away, h_g, a_g)
    teams = set()
    for m in matches:
        teams.add(m[0])
        teams.add(m[1])
        
    standings = {t: {'pts': 0, 'gd': 0, 'gs': 0, 'team': t} for t in teams}
    for h, a, h_g, a_g in matches:
        standings[h]['gd'] += (h_g - a_g)
        standings[a]['gd'] += (a_g - h_g)
        standings[h]['gs'] += h_g
        standings[a]['gs'] += a_g
        if h_g > a_g:
            standings[h]['pts'] += 3
        elif a_g > h_g:
            standings[a]['pts'] += 3
        else:
            standings[h]['pts'] += 1
            standings[a]['pts'] += 1
            
    sorted_teams = sorted(standings.values(), key=lambda x: x['team'])
    sorted_teams = sorted(sorted_teams, key=lambda x: (x['pts'], x['gd'], x['gs']), reverse=True)
    return sorted_teams

def simulate_tournament():
    # Group Stage
    group_results = {}
    all_group_matches = []
    for group_id, group_df in group_fixtures.groupby('group'):
        matches = []
        for _, row in group_df.iterrows():
            h_g, a_g, _, _ = simulate_match(row['home_team'], row['away_team'])
            matches.append((row['home_team'], row['away_team'], h_g, a_g))
            all_group_matches.append((row['match_id'], h_g, a_g))
        group_results[group_id] = resolve_group_standings(matches)
        
    # Advancing teams
    winners = {g: res[0]['team'] for g, res in group_results.items()}
    runners_up = {g: res[1]['team'] for g, res in group_results.items()}
    
    third_places = []
    for g, res in group_results.items():
        t = res[2].copy()
        t['group'] = g
        third_places.append(t)
        
    # Deterministic tiebreaker for best 3rd
    third_places = sorted(third_places, key=lambda x: x['team'])
    best_3rd = sorted(third_places, key=lambda x: (x['pts'], x['gd'], x['gs']), reverse=True)[:8]
    best_3rd_teams = {x['team'] for x in best_3rd}
    best_3rd_by_group = {x['group']: x['team'] for x in best_3rd}
    
    # Knockout Stage
    match_participants = {} # match_id -> (home, away)
    match_winners = {} # match_id -> winner
    match_losers = {} # match_id -> loser
    
    # Helper to resolve slot
    def resolve_slot(slot):
        if slot.startswith('Winner Group '):
            return winners[slot[-1]]
        if slot.startswith('Runner-up Group '):
            return runners_up[slot[-1]]
        if slot.startswith('Winner Match '):
            m_id = int(slot.split(' ')[-1])
            return match_winners[m_id]
        if slot.startswith('Loser Match '):
            m_id = int(slot.split(' ')[-1])
            return match_losers[m_id]
        if slot.startswith('Best 3rd '):
            # Best 3rd (Groups A/B/C/D/F)
            groups = slot.split('(')[1].split(')')[0].split('/')
            for g in groups:
                if g in best_3rd_by_group and best_3rd_by_group[g] not in assigned_3rd:
                    assigned_3rd.add(best_3rd_by_group[g])
                    return best_3rd_by_group[g]
            # Fallback if none of the preferred groups advanced
            sorted_groups = sorted(best_3rd_by_group.keys())
            for g in sorted_groups:
                if best_3rd_by_group[g] not in assigned_3rd:
                    assigned_3rd.add(best_3rd_by_group[g])
                    return best_3rd_by_group[g]
        return "Unknown"

    assigned_3rd = set()
    # R32 (Matches 73-88)
    for _, row in knockout_slots[knockout_slots['round'] == 'Round of 32'].iterrows():
        h = resolve_slot(row['slot_home'])
        a = resolve_slot(row['slot_away'])
        h_g, a_g, w, l = simulate_match(h, a, knockout=True)
        match_participants[row['match_id']] = (h, a)
        match_winners[row['match_id']] = w
        match_losers[row['match_id']] = l
        
    # Subsequent rounds
    for rnd in ['Round of 16', 'Quarter-final', 'Semi-final', 'Third-place playoff', 'Final']:
        for _, row in knockout_slots[knockout_slots['round'] == rnd].iterrows():
            h = resolve_slot(row['slot_home'])
            a = resolve_slot(row['slot_away'])
            h_g, a_g, w, l = simulate_match(h, a, knockout=True)
            match_participants[row['match_id']] = (h, a)
            match_winners[row['match_id']] = w
            match_losers[row['match_id']] = l
            
    return match_participants, match_winners, all_group_matches

# Run simulations
num_sims = 10000
match_participants_history = {m_id: [] for m_id in range(73, 105)}
champions = []
total_group_goals = 0
total_draws = 0

print(f"Running {num_sims} simulations...")
for i in range(num_sims):
    if (i + 1) % 100 == 0:
        print(f"Sim {i+1}/{num_sims}...")
    participants, winners, group_matches = simulate_tournament()
    for m_id, teams in participants.items():
        match_participants_history[m_id].append(teams)
    champions.append(winners[104])
    
    for _, h_g, a_g in group_matches:
        total_group_goals += (h_g + a_g)
        if h_g == a_g:
            total_draws += 1

# Consensus Participants
consensus_bracket = {}
for m_id, history in match_participants_history.items():
    if not history:
        print(f"Missing history for match {m_id}")
        continue
    consensus_bracket[m_id] = Counter(history).most_common(1)[0][0]

# Champion Probabilities
champ_probs = pd.DataFrame(Counter(champions).most_common(), columns=['team', 'probability'])
champ_probs['probability'] /= num_sims
champ_probs.to_csv(SUBMISSION_DIR / "champion_probabilities.csv", index=False)

# Final Submission
rows = []
def get_ev_data(h_team, a_team, match_id, is_knockout=False):
    probs, lam, mu = get_match_probs(h_team, a_team)
    
    if is_knockout:
        # ORDER: Safe, deterministic rounding to protect knockout bracket logic
        h = int(np.round(lam))
        a = int(np.round(mu))
    else:
        # CHAOS: Sample the probability matrix for realistic group stage variance
        flat_probs = probs.flatten()
        idx = np.random.choice(len(flat_probs), p=flat_probs)
        h, a = np.unravel_index(idx, probs.shape)
        
    return h, a
#group stage
for _, row in group_fixtures.iterrows():
    h, a = get_ev_data(row['home_team'], row['away_team'], row['match_id'], is_knockout=False) # <-- CHANGED
    winner = 'home' if h > a else ('away' if a > h else 'draw')
    rows.append({
        'match_id': row['match_id'],
        'home_team': row['home_team'],       
        'away_team': row['away_team'],       
        'predicted_home_goals': h,
        'predicted_away_goals': a,
        'corners': 9,
        'yellow_cards': 3,
        'red_cards': 0,
        'match_winner': winner,
        'penalties': False
    })

# --- Fix Match 103 / 104 Multiverse Paradox ---
sf1_teams = set(consensus_bracket[101])
sf2_teams = set(consensus_bracket[102])
final_teams = set(consensus_bracket[104])

# The losers of the Semifinals MUST go to Match 103
third_place_teams = list((sf1_teams | sf2_teams) - final_teams)
if len(third_place_teams) == 2:
    consensus_bracket[103] = (third_place_teams[0], third_place_teams[1])

# Knockouts
for m_id in range(73, 105):
    h_team, a_team = consensus_bracket[m_id]
    h, a = get_ev_data(h_team, a_team, m_id, is_knockout=True) # <-- CHANGED
    
    if h > a:
        winner = 'home'
    elif a > h:
        winner = 'away'
    else:
        # Tie-break using ELO for knockouts
        elo_h = team_elo.get(h_team, 1500)
        elo_a = team_elo.get(a_team, 1500)
        winner = 'home' if elo_h >= elo_a else 'away'
    
    rows.append({
        'match_id': m_id,
        'home_team': h_team,                
        'away_team': a_team,                
        'predicted_home_goals': h,
        'predicted_away_goals': a,
        'corners': 9,
        'yellow_cards': 3,
        'red_cards': 0,
        'match_winner': winner,
        'penalties': (h == a)
    })


submission_df = pd.DataFrame(rows)
submission_df.to_csv(SUBMISSION_DIR / "final_ev_submission.csv", index=False)

# Bracket Narrative
print("\n--- Consensus Bracket Narrative ---")
sf1 = consensus_bracket[101]
sf2 = consensus_bracket[102]
final = consensus_bracket[104]
print(f"Semifinal 1: {sf1[0]} vs {sf1[1]}")
print(f"Semifinal 2: {sf2[0]} vs {sf2[1]}")
print(f"Final: {final[0]} vs {final[1]}")
print(f"Predicted Champion: {champ_probs.iloc[0]['team']}")

# Diagnostics
print(f"Avg Goals (Group Stage): {total_group_goals / (num_sims * 72):.2f}")
print(f"Draw Rate (Group Stage): {total_draws / (num_sims * 72):.2%}")
print("\nTop 5 Champions:")
print(champ_probs.head(5))
