# Submission Compilation Design

## Goal
Compile group stage and knockout stage predictions into a single standardized submission file.

## Input Files
- `submissions/group_stage_goal_predictions.csv`
- `submissions/knockout_stage_predictions.csv`

## Output File
- `submissions/submission.csv`

## Requirements
1. Concatenate both input files.
2. Standardized Columns: `['match_id', 'predicted_home_goals', 'predicted_away_goals', 'corners', 'yellow_cards', 'red_cards', 'match_winner', 'penalties']`.
3. Group Stage (ID 1-72) Specifics:
    - `penalties` must be `False`.
    - `match_winner` logic: if missing, use goal difference to determine 'home', 'away', or 'draw'.

## Approach
- Use `pandas` for data manipulation.
- Load dataframes, rename columns if necessary.
- Concatenate.
- Apply group-stage-specific transformations.
- Ensure column order and existence.
- Save to CSV.
