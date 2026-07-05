# Fifa 2026 Prediction Competition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the core infrastructure, data handling, and baseline loading for the FIFA 2026 World Cup prediction model, resolving the `FileNotFoundError` and preparing the source code for feature engineering.

**Architecture:** Use a modular Python structure under `src/`. Centralize data loading to handle missing files gracefully. Separate data acquisition, feature engineering, and model prediction logic.

**Tech Stack:** Python, Pandas, Scikit-learn, XGBoost.

---

### Task 1: Update Directory Structure and Memory

**Files:**
- Modify: `MEMORY.md` (or create if missing)
- Modify: `.gitignore`

- [ ] **Step 1: Document project requirements in MEMORY.md**
Add key competition details (scores, corners, cards, knockout multipliers) to the project memory.

- [ ] **Step 2: Update .gitignore**
Ensure `data/raw/*.csv` (except placeholders) and `submissions/` are ignored if they contain large or sensitive data, but keep structure.

### Task 2: Resolve FileNotFoundError (Data Acquisition)

The notebook expects `data/group_fixtures.csv` and `data/knockout_slots.csv`. We need to create these files with the correct columns as described in the notebook background.

**Files:**
- Create: `data/group_fixtures.csv`
- Create: `data/knockout_slots.csv`

- [ ] **Step 1: Create `data/group_fixtures.csv` with headers**
Columns: `match_id`, `group`, `home_team`, `away_team`, `date`, `venue`.
(Populate with at least match 1: Mexico vs South Africa as per notebook example).

- [ ] **Step 2: Create `data/knockout_slots.csv` with headers**
Columns: `match_id`, `round`, `multiplier`, `slot_home`, `slot_away`.

### Task 3: Implement Core Data Loading Script

Create a robust data loader in `src/` to be used by both the notebook and future training scripts.

**Files:**
- Create: `src/data_loader.py`
- Test: `src/test_data_loader.py`

- [ ] **Step 1: Write `src/data_loader.py`**
Implement a class or function to load the CSVs and provide basic validation.

```python
import pandas as pd
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required data file: {file_path}")
    return pd.read_csv(file_path)

def get_fixtures():
    return load_data('data/group_fixtures.csv')

def get_knockout_slots():
    return load_data('data/knockout_slots.csv')
```

- [ ] **Step 2: Verify with a script**
Run a simple python command to ensure `get_fixtures()` returns a DataFrame.

### Task 4: Fix Notebook Data Loading

**Files:**
- Modify: `notebook.ipynb`

- [ ] **Step 1: Replace hardcoded paths with `src.data_loader`**
Update the first few cells of the notebook to use our new loader, ensuring the `FileNotFoundError` is resolved.

### Task 5: Initial Baseline Feature Engineering (Placeholder)

**Files:**
- Create: `src/features/build_features.py`

- [ ] **Step 1: Create feature engineering skeleton**
Define functions for team encoding and historical average merging.

