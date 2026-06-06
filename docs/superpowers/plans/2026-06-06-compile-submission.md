# Compile Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a script to merge group stage and knockout stage predictions into a final standardized submission file.

**Architecture:** A standalone Python script using Pandas to load, transform, and concatenate CSV files. It ensures the final schema matches requirements and applies specific logic for group stage matches.

**Tech Stack:** Python 3.x, Pandas

---

### Task 1: Create Compilation Script

**Files:**
- Create: `src/models/compile_submission.py`

- [ ] **Step 1: Write the script content**

```python
import pandas as pd
import os

def compile_submission():
    """
    Loads group and knockout predictions, merges them, and standardizes for submission.
    """
    group_path = 'submissions/group_stage_goal_predictions.csv'
    knockout_path = 'submissions/knockout_stage_predictions.csv'
    output_path = 'submissions/submission.csv'
    
    print(f"Loading {group_path}...")
    group_df = pd.read_csv(group_path)
    print(f"Loading {knockout_path}...")
    knockout_df = pd.read_csv(knockout_path)
    
    # Standardize Group Stage columns
    if 'winning_team' in group_df.columns:
        group_df = group_df.rename(columns={'winning_team': 'match_winner'})
    
    if 'penalties' not in group_df.columns:
        group_df['penalties'] = False
        
    # Concatenate
    print("Concatenating dataframes...")
    df = pd.concat([group_df, knockout_df], ignore_index=True)
    
    # Final column set
    required_cols = [
        'match_id', 'predicted_home_goals', 'predicted_away_goals', 
        'corners', 'yellow_cards', 'red_cards', 'match_winner', 'penalties'
    ]
    
    # Ensure all required columns exist and reorder
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    
    df = df[required_cols]
    
    # Group stage specific logic (match_id 1 to 72)
    group_mask = df['match_id'] <= 72
    
    # Force penalties to False for group stage
    df.loc[group_mask, 'penalties'] = False
    
    # Fill match_winner if missing for group stage based on goals
    def calculate_winner(row):
        if pd.isna(row['match_winner']):
            h = row['predicted_home_goals']
            a = row['predicted_away_goals']
            if h > a: return 'home'
            elif a > h: return 'away'
            else: return 'draw'
        return row['match_winner']
    
    df.loc[group_mask, 'match_winner'] = df[group_mask].apply(calculate_winner, axis=1)
    
    # Save final submission
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully saved submission to {output_path}")
    print("\nFile Schema:")
    print(df.dtypes)
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nSummary:")
    print(f"Total matches: {len(df)}")
    print(f"Group stage matches: {len(df[df['match_id'] <= 72])}")
    print(f"Knockout matches: {len(df[df['match_id'] > 72])}")

if __name__ == "__main__":
    compile_submission()
```

- [ ] **Step 2: Commit (Note: User asked not to run)**

```bash
git add src/models/compile_submission.py
git commit -m "feat(models): add compile_submission script to merge stage predictions"
```
