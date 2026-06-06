import pandas as pd
import numpy as np
import os

def main():
    """
    Post-processes the group stage goal predictions to include auxiliary statistics
    and the final winning team based on predicted scores.
    """
    file_path = 'submissions/group_stage_goal_predictions.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    # 1. Load the goal predictions
    df = pd.read_csv(file_path)
    
    # 2. Add constant columns
    df['corners'] = 9
    df['yellow_cards'] = 3
    df['red_cards'] = 0
    
    # 3. Add 'winning_team' column
    # home if predicted_home_goals > predicted_away_goals
    # away if predicted_home_goals < predicted_away_goals
    # draw if predicted_home_goals == predicted_away_goals
    conditions = [
        (df['predicted_home_goals'] > df['predicted_away_goals']),
        (df['predicted_home_goals'] < df['predicted_away_goals'])
    ]
    choices = ['home', 'away']
    df['winning_team'] = np.select(conditions, choices, default='draw')
    
    # 4. Save the updated DataFrame back to the same file
    df.to_csv(file_path, index=False)
    print(f"Successfully updated {file_path} with auxiliary columns and outcomes.")

if __name__ == "__main__":
    main()
