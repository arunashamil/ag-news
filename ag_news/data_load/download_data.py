import os
import zipfile

import gdown
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    gdrive_url = config["data_load"]["url"]
    output_dir = config["data_load"]["data_path"]
    zip_filename = "ag_news_data.zip"

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading zip file...")
    gdown.download(gdrive_url, zip_filename, quiet=False, fuzzy=True)

    print("Extracting zip file...")
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("Deleting zip file...")
    os.remove(zip_filename)
    print(f"Extracted contents saved to: {output_dir}")


if __name__ == "__main__":
    main()
