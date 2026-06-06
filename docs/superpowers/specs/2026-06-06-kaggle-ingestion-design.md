# Design Doc: Kaggle Data Ingestion

Date: 2026-06-06
Topic: Download international football results from Kaggle

## Goal
Automate the download and extraction of the 'international-football-results-from-1872-to-2017' dataset to the project's raw data directory.

## Architecture
- Script: `src/data/make_dataset.py`
- Library: `kaggle` (Python API)
- Paths: `data/raw/` for storage.

## Components
1. **Directory Setup**: Ensure `data/raw/` exists.
2. **Kaggle Auth**: Use `KaggleApi().authenticate()`.
3. **Download & Extract**: Use `api.dataset_download_files` with `unzip=True`.
4. **Validation**: Check for `results.csv` existence.

## Error Handling
- Capture Kaggle API errors.
- Handle file system permission/existence issues.
