from pydantic import BaseModel

class SessionEndRequest(BaseModel):
    conversationId: str

class SessionEndResponse(BaseModel):
    status: str