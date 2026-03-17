from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime

class APIResponse(BaseModel):
    message: Optional[str] = None
    data: Optional[Any] = None
    status: str
    timestamp: datetime