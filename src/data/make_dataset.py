import logging
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    """
    Downloads and unzips the international football results dataset from Kaggle.
    """
    logger = logging.getLogger(__name__)
    
    # Configuration
    dataset = 'martj42/international-football-results-from-1872-to-2017'
    raw_data_dir = Path(__file__).resolve().parents[2] / 'data' / 'raw'
    target_file = 'results.csv'
    
    # 1. Ensure directory exists
    if not raw_data_dir.exists():
        logger.info(f"Creating directory: {raw_data_dir}")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Initialize and authenticate Kaggle API
    logger.info("Authenticating with Kaggle API...")
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error(f"Failed to authenticate with Kaggle API: {e}")
        return

    # 3. Download and unzip
    logger.info(f"Downloading dataset '{dataset}' to {raw_data_dir}...")
    try:
        # unzip=True handles the extraction
        api.dataset_download_files(dataset, path=raw_data_dir, unzip=True)
        logger.info("Download and extraction complete.")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return

    # 4. Verification
    if (raw_data_dir / target_file).exists():
        logger.info(f"Success: {target_file} is present in {raw_data_dir}")
    else:
        logger.warning(f"Warning: {target_file} not found in {raw_data_dir} after download.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
