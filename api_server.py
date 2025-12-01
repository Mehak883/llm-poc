from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from intent_analysis import analyze_call_structured
import json

app = FastAPI()

# Allow .NET frontend/backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze(request: Request):

    payload = await request.json()
    print("Incoming payload:", json.dumps(payload, indent=2))
    conversation_id = payload.get("conversation_id")
    agent_id = payload.get("agent_id")
    transcript = payload.get("transcript", [])

    if not conversation_id:
        return {"error": "conversation_id missing"}

    # transcript is already user-only (from .NET)
    analysis = analyze_call_structured(conversation_id, transcript)

    # send analysis back to .NET
    return analysis
