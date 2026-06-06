# Kaggle Data Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a script to download and unzip international football results from Kaggle to `data/raw/`.

**Architecture:** A Python script `src/data/make_dataset.py` using the `kaggle` Python API for authenticated download and native extraction.

**Tech Stack:** Python, `kaggle` library, `pathlib`.

---

### Task 1: Initialize Data Package

**Files:**
- Create: `src/data/__init__.py`

- [ ] **Step 1: Create __init__.py**

```python
"""
Data ingestion and processing scripts.
"""
```

---

### Task 2: Implement make_dataset.py

**Files:**
- Create: `src/data/make_dataset.py`

- [ ] **Step 1: Write the implementation**

```python
import os
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
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile src/data/make_dataset.py`
Expected: No output (success).

---

### Task 3: Test Dataset Existence (Post-Execution)

**Files:**
- Create: `tests/test_kaggle_data.py`

- [ ] **Step 1: Write test script**

```python
import os
from pathlib import Path

def test_raw_data_exists():
    raw_data_dir = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    results_file = raw_data_dir / 'results.csv'
    assert results_file.exists(), f"Missing {results_file}"

if __name__ == "__main__":
    try:
        test_raw_data_exists()
        print("Test passed: results.csv exists.")
    except AssertionError as e:
        print(f"Test failed: {e}")
```
