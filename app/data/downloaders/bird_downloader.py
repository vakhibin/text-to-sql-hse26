import asyncio
import aiohttp
import aiofiles
import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional
from app.core.logger import logger

# Константы
BIRD_URLS = {
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
    "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
}
DEFAULT_BIRD_PATH = Path("./databases/bird")


class BirdDatasetDownloader:
    def __init__(self, download_path: Optional[str] = None):
        self.download_path = Path(download_path) if download_path else DEFAULT_BIRD_PATH
        self._logger = logger

    async def download_and_extract(self, split: str = "dev", force: bool = False):
        """
        Download and unzip train or dev split
        """
        if split not in BIRD_URLS:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'dev'")

        url = BIRD_URLS[split]
        target_dir = self.download_path
        zip_file_path = target_dir / f"{split}.zip"

        # Создаем папку если нет
        target_dir.mkdir(parents=True, exist_ok=True)

        # Проверка на существование данных
        # В BIRD после распаковки dev.zip появляется файл dev.json и папка dev_databases
        check_file = target_dir / split
        if check_file.exists() and not force:
            self._logger.info(f"BIRD {split} dataset already exists in {target_dir}. Skipping.")
            return

        # 1. Загрузка
        await self._download_file(url, zip_file_path)

        # 2. Распаковка
        await self._extract_zip(zip_file_path, target_dir)

        # 3. Очистка (удаление zip)
        if zip_file_path.exists():
            os.remove(zip_file_path)
            self._logger.info(f"Cleaned up temporary file {zip_file_path}")

        self._logger.info(f"BIRD {split} is ready in {target_dir}")

    async def _download_file(self, url: str, dest: Path):
        self._logger.info(f"Downloading BIRD from {url}...")

        timeout = aiohttp.ClientTimeout(
            total=None, 
            sock_read=120,  
            sock_connect=120  
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download. Status: {response.status}")

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                async with aiofiles.open(dest, mode='wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        downloaded += len(chunk)

                        if downloaded % (1024 * 1024 * 100) < 8192:
                            progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                            self._logger.info(f"Progress: {progress:.1f}% ({downloaded // (1024 * 1024)} MB)")
 

    async def _extract_zip(self, zip_path: Path, extract_to: Path):
        self._logger.info(f"Extracting {zip_path} to {extract_to}...")

        def _sync_extract():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            # Найти папку, которая начинается с "dev_" или "train_"
            for item in extract_to.iterdir():
                if item.is_dir() and item.name.startswith(zip_path.stem + "_"):
                    # Переименовать в просто "dev" или "train"
                    new_name = extract_to / zip_path.stem
                    if new_name.exists():
                        shutil.rmtree(new_name)
                    item.rename(new_name)
                    break
            
            # ДОБАВИТЬ: разархивируем вложенный _databases.zip
            databases_zip = extract_to / zip_path.stem / f"{zip_path.stem}_databases.zip"
            if databases_zip.exists():
                self._logger.info(f"Extracting nested {databases_zip}...")
                with zipfile.ZipFile(databases_zip, 'r') as db_zip:
                    db_zip.extractall(extract_to)
                os.remove(databases_zip)
                self._logger.info(f"Removed {databases_zip}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_extract)
        self._logger.info("Extraction complete.")


async def main():
    downloader = BirdDatasetDownloader()

    await downloader.download_and_extract(split="dev")
    await downloader.download_and_extract(split="train")


if __name__ == "__main__":
    asyncio.run(main())