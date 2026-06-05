import pandas as pd
from pathlib import Path
from typing import Union

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded DataFrame is empty.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
        
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
        
    df = pd.read_csv(path)
    
    if df.empty:
        raise ValueError(f"Loaded data from {path} is empty.")
        
    return df

def get_fixtures() -> pd.DataFrame:
    """Returns the group fixtures data."""
    return load_data('data/group_fixtures.csv')

def get_knockout_slots() -> pd.DataFrame:
    """Returns the knockout slots data."""
    return load_data('data/knockout_slots.csv')
