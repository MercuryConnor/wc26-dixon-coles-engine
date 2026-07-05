import pathlib
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import time
from datetime import datetime
from pathlib import Path

from src.models.predict_group_stage import PROJECT_ROOT

def fit_model():
    # 1. Load data
    print("Loading data...")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "results.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter post-2018
    df = df[df['date'] >= '2019-01-01'].copy()
    
    # Drop NaNs
    df = df.dropna(subset=['home_score', 'away_score', 'home_team', 'away_team'])
    
    # 2. Preprocessing
    print(f"Preprocessing {len(df)} matches...")
    # Cap goals at 6
    df['home_score'] = df['home_score'].clip(upper=6)
    df['away_score'] = df['away_score'].clip(upper=6)
    
    # Time decay
    t_now = df['date'].max()
    half_life = 730 # days
    df['days_ago'] = (t_now - df['date']).dt.days
    df['weight_time'] = np.exp(-np.log(2) * df['days_ago'] / half_life)
    
    # Tournament weights
    def get_tourn_weight(t):
        t = t.lower()
        if 'fifa world cup' in t and 'qualification' not in t:
            return 1.0
        if any(c in t for c in ['euro', 'copa américa', 'asian cup', 'african cup of nations', 'gold cup', 'nations cup']):
            return 0.8
        if 'qualification' in t or 'qualifying' in t:
            return 0.5
        if 'friendly' in t:
            return 0.1
        return 0.3
        
    df['weight_tourn'] = df['tournament'].apply(get_tourn_weight)
    df['weight'] = df['weight_time'] * df['weight_tourn']
    
    # Host advantage mapping
    df['is_host'] = (df['home_team'] == df['country']).astype(int)
    
    # Teams
    teams = sorted(list(set(df['home_team']) | set(df['away_team'])))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    
    home_idx = df['home_team'].map(team_to_idx).values
    away_idx = df['away_team'].map(team_to_idx).values
    home_goals = df['home_score'].values
    away_goals = df['away_score'].values
    weights = df['weight'].values
    is_host = df['is_host'].values
    
    # 3. Optimization
    print(f"Optimizing for {n_teams} teams...")
    
    def tau(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        if x == 1 and y == 0:
            return 1 + mu * rho
        if x == 0 and y == 1:
            return 1 + lam * rho
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def neg_log_likelihood(params):
        att = params[:n_teams]
        dfn = params[n_teams:2*n_teams]
        h_adv = params[2*n_teams]
        rho = params[2*n_teams+1]
        
        # Expected goals (in log space to ensure positivity)
        log_lam = att[home_idx] + dfn[away_idx] + is_host * h_adv
        log_mu = att[away_idx] + dfn[home_idx]
        
        # Stability: clip log values
        log_lam = np.clip(log_lam, -10, 5)
        log_mu = np.clip(log_mu, -10, 5)
        
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)
        
        # Poisson likelihood
        ll_home = home_goals * log_lam - lam
        ll_away = away_goals * log_mu - mu
        
        # Dixon-Coles adjustment
        adj = np.ones_like(lam)
        mask00 = (home_goals == 0) & (away_goals == 0)
        mask10 = (home_goals == 1) & (away_goals == 0)
        mask01 = (home_goals == 0) & (away_goals == 1)
        mask11 = (home_goals == 1) & (away_goals == 1)
        
        adj[mask00] = 1 - lam[mask00] * mu[mask00] * rho
        adj[mask10] = 1 + mu[mask10] * rho
        adj[mask01] = 1 + lam[mask01] * rho
        adj[mask11] = 1 - rho
        
        # Avoid log of negative or zero
        adj = np.clip(adj, 1e-10, None)
        ll_adj = np.log(adj)
        
        total_ll = np.sum(weights * (ll_home + ll_away + ll_adj))
        
        # Regularization (L2 penalty)
        reg_lambda = 0.01
        penalty = reg_lambda * np.sum(att**2 + dfn**2)
        
        # Constraint penalty: sum(att) = 0
        cons_penalty = 1e4 * (np.sum(att)**2)
        
        return -total_ll + penalty + cons_penalty

    # Initial guess
    init_params = np.zeros(2 * n_teams + 2)
    init_params[2*n_teams + 1] = 0.0 # rho
    
    # Constraint: sum(Attack) = 0
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[:n_teams])}]
    
    # Bounds for rho: [-0.2, 0.2]
    bounds = [(None, None)] * (2 * n_teams + 1) + [(-0.2, 0.2)]
    
    start_time = time.time()
    res = minimize(
        neg_log_likelihood, 
        init_params, 
        method='L-BFGS-B', 
        bounds=bounds,
        # constraints not supported by L-BFGS-B, but we can use penalty in likelihood
        # Or switch to SLSQP if sum constraint is strict. 
        # But Dixon-Coles often uses sum(att) = N or mean(att) = 1.
        # User asked for sum(Attack) = 0 (log space means product(alpha) = 1).
    )
    
    # Re-run with SLSQP for the equality constraint if needed, 
    # but L-BFGS-B is faster for high dimensions.
    # Let's use SLSQP if n_teams isn't too large, or just use the constraint in SLSQP.
    if not res.success:
        print("L-BFGS-B failed, trying SLSQP...")
        res = minimize(
            neg_log_likelihood,
            init_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

    runtime = time.time() - start_time
    
    # 4. Diagnostics
    print("\n--- DIAGNOSTICS ---")
    print(f"Final Loss: {res.fun:.4f}")
    print(f"Iterations: {res.nit}")
    print(f"Convergence Status: {res.message}")
    print(f"Runtime: {runtime:.2f}s")
    
    att_final = res.x[:n_teams]
    dfn_final = res.x[n_teams:2*n_teams]
    h_adv_final = res.x[2*n_teams]
    rho_final = res.x[2*n_teams+1]
    
    print(f"Attack Range: [{att_final.min():.4f}, {att_final.max():.4f}]")
    print(f"Defense Range: [{dfn_final.min():.4f}, {dfn_final.max():.4f}]")
    print(f"Host Advantage (exp): {np.exp(h_adv_final):.4f}")
    print(f"Rho: {rho_final:.4f}")
    
    top_att_idx = np.argsort(att_final)[-5:][::-1]
    print("\nTop 5 Attack teams:")
    for idx in top_att_idx:
        print(f"{teams[idx]}: {att_final[idx]:.4f}")
        
    top_dfn_idx = np.argsort(dfn_final)[:5] # Lower is better defense in some models, 
    # but here lambda = exp(att + dfn), so lower dfn means lower opponent score. Correct.
    print("\nTop 5 Defense teams (lowest params):")
    for idx in top_dfn_idx:
        print(f"{teams[idx]}: {dfn_final[idx]:.4f}")
        
    # 5. Output
    output_df = pd.DataFrame({
        'team': teams,
        'attack': att_final,
        'defense': dfn_final
    })
    # Add scalar params as columns or separate file? 
    # Usually better to have them accessible.
    output_df['host_advantage'] = h_adv_final
    output_df['rho'] = rho_final
    
    output_path = 'data/processed/dc_parameters_regularized.csv'
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved parameters to {output_path}")

if __name__ == "__main__":
    fit_model()
