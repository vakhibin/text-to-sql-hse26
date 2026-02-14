import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", None)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen-2.5-coder-32b-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 512))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")