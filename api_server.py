import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from intent_analysis import analyze_call_structured
import json

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Server for feedback analysis initialized")

# Allow .NET backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze(request: Request):

    try:
        payload = await request.json()
        logger.debug(f"Incoming payload: {json.dumps(payload, indent=2)}")
        conversation_id = payload.get("conversation_id")
        agent_id = payload.get("agent_id")
        transcript = payload.get("transcript", [])

        if not conversation_id:
            logger.warning("Missing conversation_id in request payload")
            return {"error": "conversation_id missing"}

        # transcript is already user-only (from .NET)
        analysis = analyze_call_structured(conversation_id, transcript)
        if "error" in analysis:
            logger.error(f"Analysis failed for conversation_id: {conversation_id}")
            return JSONResponse(status_code=500, content={"error": analysis["error"]})
        logger.info(f"Analysis completed successfully for conversation_id: {conversation_id}")
        # send analysis back to .NET
        return analysis
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON payload"}
        )
    except Exception as e:
        logger.exception(f"Error processing analysis request: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
