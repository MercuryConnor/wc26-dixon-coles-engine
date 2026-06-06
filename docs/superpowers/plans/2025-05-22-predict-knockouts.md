# Predict Knockouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a script to predict knockout stage results based on group standings and Elo ratings.

**Architecture:** Load data, determine advancing teams, simulate matches sequentially using Elo-based scoring, and save results.

**Tech Stack:** Python, Pandas

---

### Task 1: Initialize Script and Load Data

**Files:**
- Create: `src/models/predict_knockouts.py`

- [ ] **Step 1: Write boiler plate and data loading logic**

```python
import pandas as pd
import numpy as np
import os

def load_data():
    standings = pd.read_csv('data/processed/group_standings.csv')
    features = pd.read_csv('data/processed/team_features.csv')
    slots = pd.read_csv('data/knockout_slots.csv')
    return standings, features, slots
```

### Task 2: Determine Advancing Teams

**Files:**
- Modify: `src/models/predict_knockouts.py`

- [ ] **Step 1: Implement logic to find top 2 and best 8 3rd place teams**

```python
def get_advancing_teams(standings):
    # Top 2 from each group
    advancers = {}
    groups = standings['group'].unique()
    for g in groups:
        grp_teams = standings[standings['group'] == g].sort_values(
            ['points', 'goal_difference', 'goals_scored'], ascending=False
        )
        advancers[f'Winner Group {g}'] = grp_teams.iloc[0]['team']
        advancers[f'Runner-up Group {g}'] = grp_teams.iloc[1]['team']
    
    # Best 3rd place teams
    third_place = []
    for g in groups:
        grp_teams = standings[standings['group'] == g].sort_values(
            ['points', 'goal_difference', 'goals_scored'], ascending=False
        )
        if len(grp_teams) >= 3:
            third_place.append(grp_teams.iloc[2])
    
    df_third = pd.DataFrame(third_place).sort_values(
        ['points', 'goal_difference', 'goals_scored'], ascending=False
    )
    best_third = df_third.head(8)['team'].tolist()
    
    for i, team in enumerate(best_third):
        advancers[f'Best 3rd-place {i+1}'] = team
        
    return advancers
```

### Task 3: Implement Match Simulation

**Files:**
- Modify: `src/models/predict_knockouts.py`

- [ ] **Step 1: Add Elo-based prediction and match loop**

```python
def simulate_matches(slots, advancers, elo_dict):
    match_winners = {}
    results = []
    
    # Best 3rd place mapping for slots (simplified as per instruction)
    # The instruction says "Best 3rd-place X" -> pull from sorted list.
    # Looking at knockout_slots.csv: "Best 3rd (Groups A/B/C/D/F)" etc.
    # Instruction says: "Format 'Best 3rd-place X' -> pull from the sorted list of 8 best 3rd-place teams."
    # I will regex match "Best 3rd" and map to index based on order in file or a simplified 1-8 mapping.
    # Actually, the instructions say "Format 'Best 3rd-place X' -> pull from the sorted list of 8 best 3rd-place teams."
    # This implies the slots might already have "Best 3rd-place 1" etc or I need to map them.
    # Wait, the sample slots I saw had "Best 3rd (Groups A/B/C/D/F)".
    # I will map these "Best 3rd (...)" strings to Best 3rd-place 1, 2, 3... in order of appearance.
    
    third_place_slots = slots[slots['slot_home'].str.contains('Best 3rd') | slots['slot_away'].str.contains('Best 3rd')]
    unique_third_slots = []
    for _, row in slots.iterrows():
        for s in [row['slot_home'], row['slot_away']]:
            if 'Best 3rd' in s and s not in unique_third_slots:
                unique_third_slots.append(s)
    
    for i, slot_name in enumerate(unique_third_slots):
        if i < 8:
            advancers[slot_name] = advancers.get(f'Best 3rd-place {i+1}')

    for _, row in slots.sort_values('match_id').iterrows():
        match_id = row['match_id']
        
        home_slot = row['slot_home']
        away_slot = row['slot_away']
        
        home_team = advancers.get(home_slot) or match_winners.get(home_slot.replace('Winner Match ', ''))
        away_team = advancers.get(away_slot) or match_winners.get(away_slot.replace('Winner Match ', ''))
        
        # If home_team/away_team still None, it might be "Winner Match X"
        if not home_team and 'Winner Match' in home_slot:
            m_id = int(home_slot.split(' ')[-1])
            home_team = match_winners.get(m_id)
        if not away_team and 'Winner Match' in away_slot:
            m_id = int(away_slot.split(' ')[-1])
            away_team = match_winners.get(m_id)

        home_elo = elo_dict.get(home_team, 1500)
        away_elo = elo_dict.get(away_team, 1500)
        
        home_goals = round(max(0, 1.2 + (home_elo - away_elo) / 400))
        away_goals = round(max(0, 1.0 + (away_elo - home_elo) / 400))
        
        penalties = False
        if home_goals > away_goals:
            winner = home_team
        elif away_goals > home_goals:
            winner = away_team
        else:
            # Elo tie breaker
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
```

### Task 4: Finalize and Save

**Files:**
- Modify: `src/models/predict_knockouts.py`

- [ ] **Step 1: Add main execution and saving**

```python
def main():
    standings, features, slots = load_data()
    elo_dict = dict(zip(features['team'], features['elo_rating']))
    advancers = get_advancing_teams(standings)
    results = simulate_matches(slots, advancers, elo_dict)
    
    df_results = pd.DataFrame(results)
    os.makedirs('submissions', exist_ok=True)
    df_results.to_csv('submissions/knockout_stage_predictions.csv', index=False)

if __name__ == '__main__':
    main()
```
