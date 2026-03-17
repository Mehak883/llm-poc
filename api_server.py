import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from intent_analysis import IntentAnalysis
from Services.assist_service import AssistService   
from Services.checklist_service import ChecklistService
from fastapi import UploadFile, File
from Models.analyze_models import AnalyzeRequest
from Models.assist_models import AssistRequest
from Models.session_models import SessionEndRequest
from Models.base_response import APIResponse
import json

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Server for feedback analysis initialized")
assist_service = AssistService()
checklist_service = ChecklistService()
intent_analysis_service = IntentAnalysis()

# Allow .NET backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):

    try:
        
        conversation_id = request.conversation_id
        agent_id = request.agent_id
        transcript = request.transcript

        if not conversation_id:
            logger.warning("Missing conversation_id in request payload")
            return JSONResponse(
                status_code=400,
                content=APIResponse(
                    status="error",
                    timestamp=datetime.now(timezone.utc),
                    message="conversation_id missing"
                ).model_dump()
            )

        # transcript is already user-only (from .NET)
        analysis = intent_analysis_service.analyze_call_structured(conversation_id, transcript)
        if "error" in analysis:
            logger.error(f"Analysis failed for conversation_id: {conversation_id}")
            return JSONResponse(
                status_code=500,
                content=APIResponse(
                    status="error",
                    timestamp=datetime.now(timezone.utc),
                    message=analysis["error"]
                ).model_dump()
            )
        logger.info(f"Analysis completed successfully for conversation_id: {conversation_id}")
        # send analysis back to .NET
        return APIResponse(
            status="success",
            timestamp=datetime.now(timezone.utc),
            data=analysis
        ).model_dump(exclude_none=True)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message="Invalid JSON payload for analyze api."
            ).model_dump()
        )
    except Exception as e:
        logger.exception(f"Error processing analysis request: {e}")
        return JSONResponse(
            status_code=500,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            ).model_dump()
        )

@app.post("/api/assist")
async def assist(request: AssistRequest):

    try:

        conversation_id = request.conversationId
        transcript = request.transcript
        checklist = request.complianceChecklist
       

        result = assist_service.analyze(
            conversation_id,
            transcript,
            checklist
        )

        return APIResponse(
            status="success",
            timestamp=datetime.now(timezone.utc),
            data=result
        ).model_dump(exclude_none=True)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message="Invalid JSON payload for analyze api."
            ).model_dump()
        )
    except Exception as e:
        logger.exception(f"Error processing assist request: {e}")
        return JSONResponse(
            status_code=500,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            ).model_dump()
        )


@app.post("/api/session/end")
async def session_end(request: SessionEndRequest):

    try:
        conv_id = request.conversationId
        if not conv_id:
            return {"error": "conv_id missing"}
        assist_service.end_session(conv_id)
        return APIResponse(
            status="success",
            timestamp=datetime.now(timezone.utc),
            data={"conversationId": conv_id}
        ).model_dump(exclude_none=True)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message="Invalid JSON payload for analyze api."
            ).model_dump()
        )
    except Exception as e:
        logger.exception("Error processing session management.")

        return JSONResponse(

            status_code=500,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            ).model_dump()
        )
    
@app.post("/api/checklist/generate")
async def generate_checklist(
    file: UploadFile = File(...)
):
    try:
        pdf_bytes = await file.read()
        checklist = checklist_service.generate_checklist(pdf_bytes)
        checklist_labels = [item["label"] for item in checklist]
        return APIResponse(
            status="success",
            timestamp=datetime.now(timezone.utc),
            data={"checklist": checklist_labels}
        ).model_dump(exclude_none=True)
    except Exception as e:
        logger.exception("Checklist generation failed")

        return JSONResponse(

            status_code=500,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            ).model_dump(mode="json")
        )