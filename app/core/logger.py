import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("myapp.log", mode="a"),
        logging.StreamHandler(),  # вывод в stdout
    ],
)

logger = logging.getLogger("app.core.logger")