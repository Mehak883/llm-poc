from pydantic import BaseModel
from typing import List, Optional

class TranscriptItem(BaseModel):
    role: str
    message: str

class AnalyzeRequest(BaseModel):
    conversation_id: str
    agent_id: Optional[str] = None
    transcript: List[TranscriptItem]

class AnalyzeResponse(BaseModel):
    conversation_id: str
    intent: str
    feedback: dict
    performance_scores: dict
    key_moments: list
    opening_response_sentence: str
    words_spoken: int
    customer_satisfaction_score: float