from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import poisson
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

def tau(x, y, lambda_x, lambda_y, rho):
    if x == 0 and y == 0: return 1 - lambda_x * lambda_y * rho
    elif x == 0 and y == 1: return 1 + lambda_x * rho
    elif x == 1 and y == 0: return 1 + lambda_y * rho
    elif x == 1 and y == 1: return 1 - rho
    return 1

def evaluate():
    res = pd.read_csv(DATA_DIR / "raw" / "results.csv")
    wc22 = res[(res['tournament'] == 'FIFA World Cup') & (res['date'].str.startswith('2022'))].copy()
    
    # Load parameters
    params_df = pd.read_csv(PROCESSED_DIR / "dc_parameters_regularized.csv")
    
    # Extract rho from the first row (it seems repeated across rows in this CSV format)
    rho = params_df['rho'].iloc[0]
    
    # Map teams to attack/defense
    att_dict = params_df.set_index('team')['attack'].to_dict()
    def_dict = params_df.set_index('team')['defense'].to_dict()
    
    brier, logloss, rps, hit, mass = [], [], [], [], []
    pred_draws, pred_goals, pred_cs = [], [], []
    actual_draws, actual_goals, actual_cs = [], [], []

    valid_matches = 0
    for _, row in wc22.iterrows():
        h, a = row['home_team'], row['away_team']
        hg, ag = int(row['home_score']), int(row['away_score'])
        
        att_h = att_dict.get(h)
        def_a = def_dict.get(a)
        att_a = att_dict.get(a)
        def_h = def_dict.get(h)
        
        if None in [att_h, def_a, att_a, def_h]:
            continue
        
        valid_matches += 1
        lh = np.exp(att_h + def_a)
        la = np.exp(att_a + def_h)
        
        # 7x7 matrix (0-6 goals)
        matrix = np.outer(poisson.pmf(range(7), lh), poisson.pmf(range(7), la))
        for x in range(2):
            for y in range(2):
                matrix[x, y] *= tau(x, y, lh, la, rho)
        
        m_sum = matrix.sum()
        mass.append(m_sum)
        matrix /= m_sum
        
        # Probs
        p_h = np.sum(np.tril(matrix, -1))
        p_d = np.sum(np.diag(matrix))
        p_a = np.sum(np.triu(matrix, 1))
        
        # Outcome
        y = [1,0,0] if hg > ag else ([0,1,0] if hg == ag else [0,0,1])
        p = [p_h, p_d, p_a]
        
        brier.append(np.sum((np.array(p) - y)**2))
        logloss.append(-np.log(max(1e-10, p[y.index(1)])))
        
        # RPS
        cp = np.cumsum(p)
        cy = np.cumsum(y)
        rps.append(np.sum((cp[:2] - cy[:2])**2) / 2)
        
        # Score hit
        idx = np.argmax(matrix)
        pred_h, pred_a = idx // 7, idx % 7
        hit.append(1 if (pred_h == hg and pred_a == ag) else 0)
        
        pred_draws.append(p_d)
        pred_goals.append(lh + la)
        # Clean sheet: either team 0
        p_cs = np.sum(matrix[0, :]) + np.sum(matrix[:, 0]) - matrix[0,0]
        pred_cs.append(p_cs)
        
        actual_draws.append(1 if hg == ag else 0)
        actual_goals.append(hg + ag)
        actual_cs.append(1 if hg == 0 or ag == 0 else 0)

    print(f"\n--- Backtest Evaluation: World Cup 2022 ---")
    print(f"Matches: {valid_matches}")
    print(f"Mass Check: {np.mean(mass):.4f}")
    if np.mean(mass) < 0.99:
        print("WARNING: Probability mass below threshold!")
    print(f"Brier Score: {np.mean(brier):.4f}")
    print(f"Log-loss: {np.mean(logloss):.4f}")
    print(f"RPS: {np.mean(rps):.4f}")
    print(f"Exact Score Hit Rate: {np.mean(hit)*100:.2f}%")
    print(f"Draw Rate: Pred {np.mean(pred_draws):.4f}, Actual {np.mean(actual_draws):.4f}")
    print(f"Avg Goals: Pred {np.mean(pred_goals):.4f}, Actual {np.mean(actual_goals):.4f}")
    print(f"Clean Sheet Rate: Pred {np.mean(pred_cs):.4f}, Actual {np.mean(actual_cs):.4f}")
    print(f"-------------------------------------------\n")

if __name__ == '__main__':
    evaluate()
