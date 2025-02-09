#!/usr/bin/env python3
import os
import sys
import yaml
import logging
import requests
import zipfile
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


def load_config(config_file="environment.yml"):
    """Load and parse the YAML configuration file."""
    if not os.path.exists(config_file):
        logging.error("Configuration file '%s' not found.", config_file)
        sys.exit(1)
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            logging.info("Configuration loaded successfully.")
            return config
    except Exception as e:
        logging.error("Failed to load configuration: %s", str(e))
        sys.exit(1)


def create_directories(paths):
    """Create directories if they don't exist."""
    for key, path in paths.items():
        try:
            os.makedirs(path, exist_ok=True)
            logging.info("Directory '%s' is ready.", path)
        except Exception as e:
            logging.error("Could not create directory '%s': %s", path, str(e))
            sys.exit(1)


def download_dataset(url, dest_path, dataset_name):
    """Download the dataset ZIP file if not already present with a progress bar."""

    zip_filename = os.path.join(dest_path, f"{dataset_name}.zip")
    if os.path.exists(zip_filename):
        logging.info("Dataset already downloaded at '%s'.", zip_filename)
        return zip_filename

    logging.info("Downloading dataset from %s", url)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get the total size from the Content-Length header (if available)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # Size of each chunk

        with open(zip_filename, "wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dataset_name}", ncols=80) as progress_bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        logging.info("Download completed: '%s'", zip_filename)
    except Exception as e:
        logging.error("Failed to download dataset: %s", str(e))
        sys.exit(1)

    return zip_filename

    return zip_filename


def extract_dataset(zip_filepath, extract_to):
    """Extract the ZIP file to the specified directory."""
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info("Extraction completed to '%s'.", extract_to)
    except zipfile.BadZipFile:
        logging.error("Bad zip file: '%s'.", zip_filepath)
        sys.exit(1)
    except Exception as e:
        logging.error("Failed to extract dataset: %s", str(e))
        sys.exit(1)


def main():
    # Load configuration
    config = load_config("environment.yml")
    try:
        project_config = config["project"]
        dataset_config = project_config["dataset"]
        dataset_url = dataset_config["url"]
        dataset_name = dataset_config["name"]
        paths = dataset_config["paths"]

        raw_path = paths["raw"]
        external_path = paths["external"]
        processed_path = paths["processed"]
    except KeyError as e:
        logging.error("Missing key in configuration: %s", str(e))
        sys.exit(1)

    # Create required directories
    create_directories({
        "raw": raw_path,
        "external": external_path,
        "processed": processed_path
    })

    # Download dataset into the raw folder
    zip_filepath = download_dataset(dataset_url, raw_path, dataset_name)

    # Extract the dataset into the external folder
    extract_dataset(zip_filepath, external_path)

    logging.info("Setup completed successfully.")


if __name__ == '__main__':
    main()
