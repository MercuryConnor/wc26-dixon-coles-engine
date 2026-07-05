from pathlib import Path

def test_raw_data_exists():
    raw_data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    results_file = raw_data_dir / 'results.csv'
    assert results_file.exists(), f"Missing {results_file}"

if __name__ == "__main__":
    try:
        test_raw_data_exists()
        print("Test passed: results.csv exists.")
    except AssertionError as e:
        print(f"Test failed: {e}")
