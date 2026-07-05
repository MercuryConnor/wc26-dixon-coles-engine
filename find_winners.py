import pandas as pd
import numpy as np
import sys
import os
from collections import Counter

# Import simulation logic from src
sys.path.append(os.getcwd())
from src.models.run_monte_carlo import load_data, run_simulation_v2

def find_winners():
    try:
        # Reduced iterations for speed in analysis
        iterations = 500 
        print(f"Running {iterations} iterations to find likely winners...")
        
        team_features, group_fixtures, knockout_slots, elo_dict, default_stats = load_data()
        
        winners = []
        for _ in range(iterations):
            res, participants = run_simulation_v2(group_fixtures, knockout_slots, elo_dict, default_stats)
            # Final match is ID 104 (World Cup Final)
            final_res = res[104]
            final_parts = participants[104]
            # Map 'home'/'away' back to team name
            winner_team = final_parts[0] if final_res['winner'] == 'home' else final_parts[1]
            winners.append(winner_team)
            
        top_5 = Counter(winners).most_common(5)
        print("\n--- TOP 5 PREDICTED WINNERS ---")
        for team, count in top_5:
            print(f"{team}: {count/iterations*100:.1f}%")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    find_winners()
