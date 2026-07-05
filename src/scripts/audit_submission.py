import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
DATA_DIR = PROJECT_ROOT / "data"

def audit():
    print("Starting Audit of 'submissions/final_ev_submission.csv'...")
    
    # 1. Setup
    submission_path = SUBMISSIONS_DIR / "final_ev_submission.csv"
    fixtures_path = DATA_DIR / "group_fixtures.csv"
    slots_path = DATA_DIR / "knockout_slots.csv"
    
    if not submission_path.exists():
        print(f"Error: {submission_path} not found.")
        return
    
    df = pd.read_csv(submission_path)
    fixtures = pd.read_csv(fixtures_path)
    slots = pd.read_csv(slots_path)
    
    # 2. CHECK 1 (Distributions)
    total_matches = 104
    total_goals = df['predicted_home_goals'].sum() + df['predicted_away_goals'].sum()
    avg_goals = total_goals / total_matches
    
    draw_rate = (df['match_winner'] == 'draw').mean()
    
    clean_sheet_rate = ((df['predicted_home_goals'] == 0) | (df['predicted_away_goals'] == 0)).mean()
    
    print("\n--- CHECK 1: DISTRIBUTIONS ---")
    print(f"Avg Goals/Game: {avg_goals:.2f}")
    print(f"Draw Rate: {draw_rate:.2%}")
    print(f"Clean Sheet Rate: {clean_sheet_rate:.2%}")
    
    # 4. CHECK 4 (Schema Compliance) - Do this before Check 3 to ensure we can propagate
    print("\n--- CHECK 4: SCHEMA COMPLIANCE ---")
    
    # Matches 1-72
    gs_df = df[df['match_id'] <= 72]
    if not gs_df['match_winner'].isin(['home', 'away', 'draw']).all():
        invalid = gs_df[~gs_df['match_winner'].isin(['home', 'away', 'draw'])]
        raise ValueError(f"Invalid match_winner in group stage (Matches 1-72): {invalid['match_winner'].unique()}")
    
    # Matches 73-104
    ko_df = df[df['match_id'] > 72]
    if not ko_df['match_winner'].isin(['home', 'away']).all():
        invalid = ko_df[~ko_df['match_winner'].isin(['home', 'away'])]
        print(f"FAILED: Found non-home/away winner in knockouts: {invalid['match_winner'].unique()}")
        # We will raise error at the end of this check
        has_schema_error = True
    else:
        has_schema_error = False
        
    # Consistency
    for idx, row in df.iterrows():
        m_id = row['match_id']
        h_g = row['predicted_home_goals']
        a_g = row['predicted_away_goals']
        winner = row['match_winner']
        
        if h_g > a_g and winner != 'home':
            raise ValueError(f"Consistency Error Match {m_id}: {h_g}-{a_g} but winner is {winner}")
        if a_g > h_g and winner != 'away':
            raise ValueError(f"Consistency Error Match {m_id}: {h_g}-{a_g} but winner is {winner}")
        if h_g == a_g:
            if m_id <= 72 and winner != 'draw':
                raise ValueError(f"Consistency Error Match {m_id}: {h_g}-{a_g} but winner is {winner} (should be draw)")
            if m_id > 72 and winner not in ['home', 'away']:
                raise ValueError(f"Consistency Error Match {m_id}: {h_g}-{a_g} but winner is {winner} (should be home or away in KO)")

    if has_schema_error:
        raise ValueError("Schema Compliance Check Failed: Knockout matches must have 'home' or 'away' winner.")
    
    print("Schema Compliance: PASSED")

    # 3. CHECK 3 (Bracket Uniqueness)
    print("\n--- CHECK 3: BRACKET UNIQUENESS ---")
    
    # Resolve Group Stage
    results = []
    for _, row in fixtures.iterrows():
        sub_row = df[df['match_id'] == row['match_id']].iloc[0]
        results.append({
            'group': row['group'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'h_g': sub_row['predicted_home_goals'],
            'a_g': sub_row['predicted_away_goals']
        })
    
    group_results = {}
    for g, g_matches in pd.DataFrame(results).groupby('group'):
        teams = set(g_matches['home_team']) | set(g_matches['away_team'])
        standings = {t: {'pts': 0, 'gd': 0, 'gs': 0, 'team': t} for t in teams}
        for _, m in g_matches.iterrows():
            standings[m['home_team']]['gd'] += (m['h_g'] - m['a_g'])
            standings[m['away_team']]['gd'] += (m['a_g'] - m['h_g'])
            standings[m['home_team']]['gs'] += m['h_g']
            standings[m['away_team']]['gs'] += m['a_g']
            if m['h_g'] > m['a_g']:
                standings[m['home_team']]['pts'] += 3
            elif m['a_g'] > m['h_g']:
                standings[m['away_team']]['pts'] += 3
            else:
                standings[m['home_team']]['pts'] += 1
                standings[m['away_team']]['pts'] += 1
        
        # Tie-break: pts, gd, gs, then name (for deterministic audit)
        sorted_standings = sorted(standings.values(), key=lambda x: (x['pts'], x['gd'], x['gs'], x['team']), reverse=True)
        # Note: In real FIFA, 4th tiebreak is not alphabetical, but for audit we need consistency.
        group_results[g] = sorted_standings

    winners = {g: res[0]['team'] for g, res in group_results.items()}
    runners_up = {g: res[1]['team'] for g, res in group_results.items()}
    
    third_places = []
    for g, res in group_results.items():
        t = res[2].copy()
        t['group'] = g
        third_places.append(t)
    
    best_3rd = sorted(third_places, key=lambda x: (x['pts'], x['gd'], x['gs'], x['team']), reverse=True)[:8]
    best_3rd_by_group = {x['group']: x['team'] for x in best_3rd}
    
    match_teams = {} # m_id -> (home, away)
    match_winner_team = {} # m_id -> team_name
    assigned_3rd = set()

    def resolve_slot(slot):
        if slot.startswith('Winner Group '):
            return winners[slot[-1]]
        if slot.startswith('Runner-up Group '):
            return runners_up[slot[-1]]
        if slot.startswith('Winner Match '):
            m_id = int(slot.split(' ')[-1])
            return match_winner_team[m_id]
        if slot.startswith('Best 3rd '):
            groups = slot.split('(')[1].split(')')[0].split('/')
            for g in groups:
                if g in best_3rd_by_group and best_3rd_by_group[g] not in assigned_3rd:
                    assigned_3rd.add(best_3rd_by_group[g])
                    return best_3rd_by_group[g]
            # Fallback
            for g in sorted(best_3rd_by_group.keys()):
                if best_3rd_by_group[g] not in assigned_3rd:
                    assigned_3rd.add(best_3rd_by_group[g])
                    return best_3rd_by_group[g]
        return "Unknown"

    # R32 (73-88)
    r32_teams = set()
    for _, row in slots[slots['round'] == 'Round of 32'].iterrows():
        h = resolve_slot(row['slot_home'])
        a = resolve_slot(row['slot_away'])
        match_teams[row['match_id']] = (h, a)
        r32_teams.add(h)
        r32_teams.add(a)
        
        sub_row = df[df['match_id'] == row['match_id']].iloc[0]
        match_winner_team[row['match_id']] = h if sub_row['match_winner'] == 'home' else a

    print(f"R32 Unique Teams: {len(r32_teams)}")
    if len(r32_teams) != 32:
        raise ValueError(f"Expected 32 unique teams in R32, got {len(r32_teams)}")

    # R16 (89-96)
    r16_teams = set()
    for _, row in slots[slots['round'] == 'Round of 16'].iterrows():
        h = resolve_slot(row['slot_home'])
        a = resolve_slot(row['slot_away'])
        match_teams[row['match_id']] = (h, a)
        r16_teams.add(h)
        r16_teams.add(a)
        sub_row = df[df['match_id'] == row['match_id']].iloc[0]
        match_winner_team[row['match_id']] = h if sub_row['match_winner'] == 'home' else a
    
    print(f"R16 Unique Teams: {len(r16_teams)}")
    if len(r16_teams) != 16:
        raise ValueError(f"Expected 16 unique teams in R16, got {len(r16_teams)}")

    # QF (97-100)
    qf_teams = set()
    for _, row in slots[slots['round'] == 'Quarter-final'].iterrows():
        h = resolve_slot(row['slot_home'])
        a = resolve_slot(row['slot_away'])
        match_teams[row['match_id']] = (h, a)
        qf_teams.add(h)
        qf_teams.add(a)
        sub_row = df[df['match_id'] == row['match_id']].iloc[0]
        match_winner_team[row['match_id']] = h if sub_row['match_winner'] == 'home' else a
    
    print(f"QF Unique Teams: {len(qf_teams)}")
    if len(qf_teams) != 8:
        raise ValueError(f"Expected 8 unique teams in QF, got {len(qf_teams)}")

    # SF (101-102)
    sf_teams = set()
    for _, row in slots[slots['round'] == 'Semi-final'].iterrows():
        h = resolve_slot(row['slot_home'])
        a = resolve_slot(row['slot_away'])
        match_teams[row['match_id']] = (h, a)
        sf_teams.add(h)
        sf_teams.add(a)
        sub_row = df[df['match_id'] == row['match_id']].iloc[0]
        match_winner_team[row['match_id']] = h if sub_row['match_winner'] == 'home' else a
    
    print(f"SF Unique Teams: {len(sf_teams)}")
    if len(sf_teams) != 4:
        raise ValueError(f"Expected 4 unique teams in SF, got {len(sf_teams)}")

    # Final (104)
    row_final = slots[slots['match_id'] == 104].iloc[0]
    final_h = resolve_slot(row_final['slot_home'])
    final_a = resolve_slot(row_final['slot_away'])
    match_teams[104] = (final_h, final_a)
    sub_row_final = df[df['match_id'] == 104].iloc[0]
    champion = final_h if sub_row_final['match_winner'] == 'home' else final_a

    print("Bracket Uniqueness: PASSED")

    # 5. HUMAN-READABLE NARRATIVE
    print("\n--- HUMAN-READABLE NARRATIVE ---")
    sf1 = match_teams[101]
    sf2 = match_teams[102]
    final = match_teams[104]
    
    print(f"Semifinal 1: {sf1[0]} vs {sf1[1]}")
    print(f"Semifinal 2: {sf2[0]} vs {sf2[1]}")
    print(f"Final: {final[0]} vs {final[1]}")
    print(f"Predicted Champion: {champion}")

if __name__ == "__main__":
    try:
        audit()
    except Exception as e:
        print(f"\nAUDIT FAILED: {e}")
        sys.exit(1)
