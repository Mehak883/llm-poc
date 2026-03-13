import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from intent_analysis import analyze_call_structured
from Services.assist_service import AssistService   
from Services.checklist_service import ChecklistService
from fastapi import UploadFile, File, Form

import json

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Server for feedback analysis initialized")
assist_service = AssistService()
checklist_service = ChecklistService()

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
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload for analyze api."})
    except Exception as e:
        logger.exception(f"Error processing analysis request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/assist")
async def assist(request: Request):

    try:

        payload = await request.json()

        conversation_id = payload.get("conversationId")
        user_message = payload.get("userMessage", "")
        checklist = payload.get("complianceChecklist", [])
        transcript = [{"role": "user", "message": user_message}]

        return assist_service.analyze(
            conversation_id,
            transcript,
            checklist
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload for the assist api."})
    except Exception as e:
        logger.exception(f"Error processing assist request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/session/end")
async def session_end(request: Request):

    try:
        payload = await request.json()
        conv_id = payload.get("conversationId")
        if not conv_id:
            return {"error": "conv_id missing"}
        assist_service.end_session(conv_id)
        return {"status": "ok"}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload of the session end api."})
    except Exception as e:
        logger.exception(f"Error in ending the request: {e}")
        return JSONResponse(status_code=500, content={"error":str(e)})
    
@app.post("/api/checklist/generate")
async def generate_checklist(
    file: UploadFile = File(...)
):
    try:

        pdf_bytes = await file.read()
        checklist = checklist_service.generate_checklist(pdf_bytes)
        return {
            "checklist": checklist
        }
    except Exception as e:
        logger.exception("Checklist generation failed")

        return {
            "error": str(e)
        }