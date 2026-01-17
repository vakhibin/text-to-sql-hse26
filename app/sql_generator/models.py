from typing import Optional, Dict, Any
from pydantic import BaseModel

class SQLGeneratorError(Exception):
    pass

class SQLGenerationResult(BaseModel):
    sql: str
    raw_response: str
    metadata: Optional[Dict[str, Any]] = None

