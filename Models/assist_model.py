from pydantic import BaseModel
from typing import List, Optional


class TranscriptItem(BaseModel):
    role: str
    message: str

class AssistRequest(BaseModel):
    conversationId: str
    transcript: List[TranscriptItem]
    complianceChecklist: Optional[List[str]] = []

class ChecklistItem(BaseModel):
    id: int
    status: str

class AssistResponse(BaseModel):
    checklist_status: List[ChecklistItem]