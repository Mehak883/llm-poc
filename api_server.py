import logging
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from intent_analysis import IntentAnalysis
from Services.checklist_service import ChecklistService
from Services.assist_service import AssistService
from Models.base_response import APIResponse
from Models.analyze_model import AnalyzeRequest
from Models.assist_model import AssistRequest
from Models.session_model import SessionEndRequest
import json

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Server for feedback analysis initialized")
checklist_service = ChecklistService()
intent_analysis_service = IntentAnalysis()
assist_service = AssistService()

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
                content={"error": "Missing conversation_id in analyze request payload"}
            )

        # transcript is already user-only (from .NET)
        analysis = intent_analysis_service.analyze_call_structured(conversation_id, transcript)
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

    except ValueError as e:
        logger.warning(f"Invalid PDF upload: {e}")
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            ).model_dump(mode="json")
        )
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
        logger.info(f"The Assist Api returning {result}")
        return APIResponse(
            status="success",
            timestamp=datetime.now(timezone.utc),
            data=result
        ).model_dump(exclude_none=True)

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
    
@app.post("/api/checklist/validate")
async def validate_pdf(file: UploadFile = File(...)):
    try:
        if not file:
            logger.warning("No file provided in validate request")
            raise ValueError("PDF file must be uploaded")

        file_bytes = await file.read()
        filename = file.filename or ""
        result = checklist_service.validate_pdf(file_bytes, filename)
        logger.info(f"Validation result for {filename}: {result['is_valid']}")

        return APIResponse(
            status="success",
            timestamp=datetime.now(timezone.utc),
            data=result
        ).model_dump(exclude_none=True)

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            ).model_dump(mode="json")
        )

    except Exception as e:
        logger.exception("Validation API failed")
        return JSONResponse(
            status_code=500,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message=str(e)
            ).model_dump(mode="json")
        )

