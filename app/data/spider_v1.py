import pathlib
import shutil
from typing import Optional
import kagglehub
from app.core.logger import logger

KAGGLE_DATASET_URL = "jeromeblanchet/yale-universitys-spider-10-nlp-dataset"
DEFAULT_DOWNLOAD_PATH = pathlib.Path("./databases/spider")  # This is correct for outside app/


class DatasetDownloader:
    def __init__(
            self,
            download_url: str = KAGGLE_DATASET_URL,
            download_path: Optional[str] = None
    ):
        # Use provided path or default
        self.download_path = pathlib.Path(download_path) if download_path else DEFAULT_DOWNLOAD_PATH

        # Create directory if it doesn't exist
        self.download_path.mkdir(parents=True, exist_ok=True)

        self._logger = logger
        self.download_url = download_url

    def download_from_kaggle(self, force_download: bool = False) -> pathlib.Path:
        try:
            self._logger.info(f"Starting Spider 1.0 dataset download from Kaggle...")
            self._logger.info(f"Source: {self.download_url}")
            self._logger.info(f"Target directory: {self.download_path.absolute()}")

            # Step 1: Download to kagglehub cache
            cached_path = kagglehub.dataset_download(self.download_url)
            self._logger.info(f"Downloaded to cache: {cached_path}")

            # Step 2: Copy to our desired location
            if self.download_path.exists() and any(self.download_path.iterdir()):
                if force_download:
                    shutil.rmtree(self.download_path)
                    self.download_path.mkdir(parents=True, exist_ok=True)
                    self._logger.info("Cleaned existing directory")
                else:
                    self._logger.warning(
                        f"Directory {self.download_path} already exists. Use force_download=True to overwrite")
                    return self.download_path

            # Copy all files from cache to target directory
            shutil.copytree(cached_path, self.download_path, dirs_exist_ok=True)

            self._logger.info(f"Dataset successfully copied to: {self.download_path}")
            return self.download_path

        except Exception as e:
            error_msg = f"Error downloading dataset from Kaggle: {str(e)}"
            self._logger.error(f"{error_msg}")
            raise Exception(error_msg)


if __name__ == "__main__":
    c = DatasetDownloader()
    c.download_from_kaggle()
