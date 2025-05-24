import os
import zipfile

import gdown


def download_and_unzip_from_gdrive(
    gdrive_url: str, output_dir: str, zip_filename="ag_news_data.zip"
) -> None:
    """
    Downloads a zip file from Google Drive and extracts it to the output directory.

    Args:
        gdrive_url (str): Google Drive URL to the zip file
        output_dir (str): Path to the output directory where contents will be extracted
        zip_filename (str): Local filename for the downloaded zip file
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading zip file...")
    gdown.download(gdrive_url, zip_filename, quiet=False, fuzzy=True)

    print("Extracting zip file...")
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("Deleting zip file...")
    os.remove(zip_filename)
    print(f"Extracted contents saved to: {output_dir}")
