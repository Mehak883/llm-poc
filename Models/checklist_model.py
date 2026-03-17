from pydantic import BaseModel
from typing import List

class ChecklistGenerateResponse(BaseModel):
    checklist: List[str]