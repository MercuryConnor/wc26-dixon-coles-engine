import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

class DixonColesMatchPredictor:
    def __init__(self):
        self.teams = None
        self.team_to_idx = None
        self.alpha = None # Attack parameters
        self.beta = None  # Defense parameters
        self.rho = 0.0    # Dependence parameter
        self.gamma = 1.0  # Home advantage
        
    @staticmethod
    def tau(x, y, alpha, beta, gamma, delta, rho):
        """
        Dixon-Coles adjustment function for low scores.
        
        Parameters:
        x, y: Goals scored by home and away teams.
        alpha: Expected goals for home team (lambda).
        beta: Expected goals for away team (mu).
        gamma: Unused placeholder.
        delta: Unused placeholder.
        rho: Dependence parameter.
        """
        if x == 0 and y == 0:
            return 1 - alpha * beta * rho
        elif x == 1 and y == 0:
            return 1 + beta * rho
        elif x == 0 and y == 1:
            return 1 + alpha * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def _neg_log_likelihood(self, params, home_team_indices, away_team_indices, home_goals, away_goals):
        n_teams = len(self.teams)
        
        # alpha_vec: attack parameters for each team
        # beta_vec: defense parameters for each team
        # gamma_val: home advantage
        # rho_val: rho
        
        alpha_vec = np.exp(params[:n_teams])
        beta_vec = np.exp(params[n_teams:2*n_teams])
        gamma_val = np.exp(params[2*n_teams])
        rho_val = params[2*n_teams + 1]
        
        # Identify the model by constraining the mean of attack parameters to 1
        penalty = 100 * (np.mean(alpha_vec) - 1.0)**2
        
        total_ll = 0
        for i in range(len(home_goals)):
            h_idx = home_team_indices[i]
            a_idx = away_team_indices[i]
            h_g = home_goals[i]
            a_g = away_goals[i]
            
            # Expected goals
            # lambda = alpha_i * beta_j * gamma
            # mu = alpha_j * beta_i
            lam = alpha_vec[h_idx] * beta_vec[a_idx] * gamma_val
            mu = alpha_vec[a_idx] * beta_vec[h_idx]
            
            # Poisson PMFs
            p_h = poisson.pmf(h_g, lam)
            p_a = poisson.pmf(a_g, mu)
            
            # Dixon-Coles adjustment
            # Passing lam as alpha, mu as beta in tau as per standard DC logic
            adj = self.tau(h_g, a_g, lam, mu, 0, 0, rho_val)
            
            likelihood = adj * p_h * p_a
            
            if likelihood <= 0:
                total_ll -= 1e6 # Penalty for invalid likelihood (e.g. rho too large)
            else:
                total_ll += np.log(likelihood)
                
        return -total_ll + penalty

    def fit(self, home_teams, away_teams, home_goals, away_goals):
        """
        Fits the Dixon-Coles model to the provided match data.
        """
        self.teams = np.unique(np.concatenate([home_teams, away_teams]))
        self.team_to_idx = {team: i for i, team in enumerate(self.teams)}
        n_teams = len(self.teams)
        
        h_indices = np.array([self.team_to_idx[t] for t in home_teams])
        a_indices = np.array([self.team_to_idx[t] for t in away_teams])
        
        # Initial parameters (log space for alpha, beta, gamma)
        # atts, defs, home_adv, rho
        initial_params = np.zeros(2 * n_teams + 2)
        
        # Bounds for rho: usually small, e.g., [-1, 1]
        # Dixon-Coles often find rho around 0.1
        bounds = [(None, None)] * (2 * n_teams + 1) + [(-1.0, 1.0)]
        
        res = minimize(
            self._neg_log_likelihood,
            initial_params,
            args=(h_indices, a_indices, home_goals, away_goals),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if not res.success:
            print(f"Warning: Optimization failed: {res.message}")
            
        self.alpha = np.exp(res.x[:n_teams])
        self.beta = np.exp(res.x[n_teams:2*n_teams])
        self.gamma = np.exp(res.x[2*n_teams])
        self.rho = res.x[2*n_teams + 1]

    def predict_score_probs(self, home_team, away_team, max_goals=10):
        """
        Predicts the probability of each scoreline (up to max_goals).
        """
        if home_team not in self.team_to_idx or away_team not in self.team_to_idx:
            raise ValueError(f"One or both teams ({home_team}, {away_team}) not in training data.")
            
        h_idx = self.team_to_idx[home_team]
        a_idx = self.team_to_idx[away_team]
        
        lam = self.alpha[h_idx] * self.beta[a_idx] * self.gamma
        mu = self.alpha[a_idx] * self.beta[h_idx]
        
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                p_h = poisson.pmf(x, lam)
                p_a = poisson.pmf(y, mu)
                adj = self.tau(x, y, lam, mu, 0, 0, self.rho)
                probs[x, y] = max(0, adj * p_h * p_a)
                
        # Normalize in case max_goals truncated significant probability
        total = probs.sum()
        if total > 0:
            probs /= total
            
        return probs
