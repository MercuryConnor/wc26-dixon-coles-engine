import unittest
import pandas as pd
from src.data_loader import get_fixtures, get_knockout_slots, load_data

class TestDataLoader(unittest.TestCase):
    def test_get_fixtures_returns_dataframe(self):
        df = get_fixtures()
        self.assertIsInstance(df, pd.DataFrame)
        # Check actual columns from data/group_fixtures.csv
        expected_columns = ['match_id', 'group', 'home_team', 'away_team', 'date', 'venue']
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_get_knockout_slots_returns_dataframe(self):
        df = get_knockout_slots()
        self.assertIsInstance(df, pd.DataFrame)
        # Check actual columns from data/knockout_slots.csv
        expected_columns = ['match_id', 'round', 'multiplier', 'slot_home', 'slot_away']
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_load_data_raises_file_not_found(self):
        """Verifies FileNotFoundError is raised for missing files."""
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')

if __name__ == '__main__':
    unittest.main()
