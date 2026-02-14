import asyncio
import aiofiles
import aiofiles.os
import pathlib
import shutil
from typing import Optional
import kagglehub
from app.core.logger import logger


KAGGLE_DATASET_URL = "jeromeblanchet/yale-universitys-spider-10-nlp-dataset"
DEFAULT_DOWNLOAD_PATH = pathlib.Path("./databases/spider")


class DatasetDownloader:
    def __init__(
            self,
            download_url: str = KAGGLE_DATASET_URL,
            download_path: Optional[str] = None
    ):
        self.download_path = pathlib.Path(download_path) if download_path else DEFAULT_DOWNLOAD_PATH
        self._logger = logger
        self.download_url = download_url

    async def download_from_kaggle(self, force_download: bool = False) -> pathlib.Path:
        try:
            self._logger.info(f"Starting Spider 1.0 dataset download from Kaggle...")
            self._logger.info(f"Source: {self.download_url}")
            self._logger.info(f"Target directory: {self.download_path.absolute()}")

            # Run kagglehub download in thread pool
            loop = asyncio.get_event_loop()
            cached_path = await loop.run_in_executor(
                None,
                lambda: kagglehub.dataset_download(self.download_url)
            )

            self._logger.info(f"Downloaded to cache: {cached_path}")

            # Create directory if it doesn't exist
            await aiofiles.os.makedirs(self.download_path, exist_ok=True)

            # Check if directory exists and has contents
            if self.download_path.exists():
                contents = await aiofiles.os.listdir(self.download_path)
                if contents:
                    if force_download:
                        await self._rmtree(self.download_path)
                        await aiofiles.os.makedirs(self.download_path, exist_ok=True)
                        self._logger.info("Cleaned existing directory")
                    else:
                        self._logger.warning(
                            f"Directory {self.download_path} already exists. Use force_download=True to overwrite"
                        )
                        return self.download_path

            # Copy all files from cache to target directory
            cached_path = pathlib.Path(cached_path)

            # Check if data is in a nested 'spider' subdirectory
            nested_path = cached_path / "spider"
            if nested_path.exists() and (nested_path / "dev.json").exists():
                source_path = nested_path
            elif (cached_path / "dev.json").exists():
                source_path = cached_path
            else:
                # Search for dev.json in subdirectories
                source_path = cached_path
                for subdir in cached_path.iterdir():
                    if subdir.is_dir() and (subdir / "dev.json").exists():
                        source_path = subdir
                        break

            # Copy contents asynchronously
            await self._copy_contents(source_path, self.download_path)

            self._logger.info(f"Dataset successfully copied to: {self.download_path}")
            return self.download_path

        except Exception as e:
            error_msg = f"Error downloading dataset from Kaggle: {str(e)}"
            self._logger.error(f"{error_msg}")
            raise Exception(error_msg)

    async def _copy_contents(self, source: pathlib.Path, dest: pathlib.Path) -> None:
        """Asynchronously copy directory contents."""
        loop = asyncio.get_event_loop()

        for item in source.iterdir():
            dest_item = dest / item.name
            if item.is_dir():
                await loop.run_in_executor(
                    None,
                    lambda: shutil.copytree(item, dest_item, dirs_exist_ok=True)
                )
            else:
                await loop.run_in_executor(
                    None,
                    lambda: shutil.copy2(item, dest_item)
                )

    async def _rmtree(self, path: pathlib.Path) -> None:
        """Asynchronously remove directory tree."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: shutil.rmtree(path))


async def main():
    # Download dataset
    downloader = DatasetDownloader()
    await downloader.download_from_kaggle()

if __name__ == "__main__":
    asyncio.run(main())
