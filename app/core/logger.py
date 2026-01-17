import logging

logging.basicConfig(
    filename='myapp.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a' # 'a' for append, 'w' for overwrite
)

logger = logging.getLogger(__name__)
