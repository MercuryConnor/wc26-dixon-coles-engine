import pandas as pd
from collections import Counter
import sys

def analyze():
    try:
        df = pd.read_csv('submissions/submission.csv')
        
        # Scorelines
        df['scoreline'] = df['predicted_home_goals'].astype(str) + '-' + df['predicted_away_goals'].astype(str)
        top_scores = df['scoreline'].value_counts().head(5)
        print("--- TOP 5 SCORE-LINES ---")
        print(top_scores.to_string())
        
        # Outcomes
        total = len(df)
        hw = len(df[df['predicted_home_goals'] > df['predicted_away_goals']])
        aw = len(df[df['predicted_away_goals'] > df['predicted_home_goals']])
        dr = len(df[df['predicted_home_goals'] == df['predicted_away_goals']])
        
        print("\n--- OUTCOMES ---")
        print(f"Home Win: {hw/total*100:.1f}%")
        print(f"Away Win: {aw/total*100:.1f}%")
        print(f"Draw: {dr/total*100:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    analyze()
