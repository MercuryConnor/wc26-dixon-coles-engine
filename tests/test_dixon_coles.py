import unittest
import numpy as np
from src.models.dixon_coles import DixonColesMatchPredictor

class TestDixonColes(unittest.TestCase):
    def setUp(self):
        self.predictor = DixonColesMatchPredictor()

    def test_initialization(self):
        self.assertIsNone(self.predictor.alpha)
        self.assertIsNone(self.predictor.beta)
        self.assertEqual(self.predictor.rho, 0.0)

    def test_tau(self):
        # rho = 0 means no adjustment
        self.assertEqual(self.predictor.tau(0, 0, 1.2, 0.8, 0, 0, 0.0), 1.0)
        
        # Test specific adjustments (with rho = 0.1)
        # x=0, y=0: 1 - alpha * beta * rho = 1 - 1.2 * 0.8 * 0.1 = 1 - 0.096 = 0.904
        self.assertAlmostEqual(self.predictor.tau(0, 0, 1.2, 0.8, 0, 0, 0.1), 0.904)
        
        # x=1, y=0: 1 + beta * rho = 1 + 0.8 * 0.1 = 1.08
        self.assertAlmostEqual(self.predictor.tau(1, 0, 1.2, 0.8, 0, 0, 0.1), 1.08)
        
        # x=0, y=1: 1 + alpha * rho = 1 + 1.2 * 0.1 = 1.12
        self.assertAlmostEqual(self.predictor.tau(0, 1, 1.2, 0.8, 0, 0, 0.1), 1.12)
        
        # x=1, y=1: 1 - rho = 1 - 0.1 = 0.9
        self.assertAlmostEqual(self.predictor.tau(1, 1, 1.2, 0.8, 0, 0, 0.1), 0.9)
        
        # x=2, y=2: 1.0
        self.assertEqual(self.predictor.tau(2, 2, 1.2, 0.8, 0, 0, 0.1), 1.0)

    def test_fit_and_predict(self):
        # Dummy data for at least 2 matches
        home_teams = ["TeamA", "TeamB", "TeamA", "TeamC"]
        away_teams = ["TeamB", "TeamC", "TeamC", "TeamA"]
        home_goals = [2, 1, 0, 1]
        away_goals = [1, 2, 2, 0]
        
        self.predictor.fit(home_teams, away_teams, home_goals, away_goals)
        
        self.assertIsNotNone(self.predictor.alpha)
        self.assertIsNotNone(self.predictor.beta)
        self.assertEqual(len(self.predictor.alpha), 3)
        
        # Verify predict_score_probs returns valid probability distribution
        probs = self.predictor.predict_score_probs("TeamA", "TeamB")
        self.assertEqual(probs.shape, (11, 11))
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0))

if __name__ == '__main__':
    unittest.main()
