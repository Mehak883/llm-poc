import logging
import json
from fastapi import FastAPI, Request, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

from intent_analysis import IntentAnalysis
from Services.assist_service import AssistService
from Services.checklist_service import ChecklistService
from Models.analyze_models import AnalyzeRequest
from Models.assist_models import AssistRequest
from Models.session_models import SessionEndRequest
from Models.base_response import APIResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Server for feedback analysis initialized")

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Exception Handlers (replaces copy-pasted try/except blocks) ────────

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            status="error",
            timestamp=datetime.now(timezone.utc),
            message="An internal error occurred. Please try again later."
        ).model_dump()
    )

# ── Dependency Injection ──────────────────────────────────────────────────────

def get_assist_service() -> AssistService:
    return AssistService()

def get_checklist_service() -> ChecklistService:
    return ChecklistService()

def get_intent_analysis_service() -> IntentAnalysis:
    return IntentAnalysis()

# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc)}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/analyze", response_model=APIResponse)
async def analyze(
    request: AnalyzeRequest,
    intent_analysis_service: IntentAnalysis = Depends(get_intent_analysis_service),
):
    conversation_id = request.conversation_id
    transcript = request.transcript

    # Validation is handled by Pydantic (min_length=1 on the model field).
    # No manual `if not conversation_id` check needed here.

    analysis = intent_analysis_service.analyze_call_structured(conversation_id, transcript)

    if "error" in analysis:
        logger.error(f"Analysis failed for conversation_id: {conversation_id}")
        return JSONResponse(
            status_code=500,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message="An internal error occurred. Please try again later."
            ).model_dump()
        )

    logger.info(f"Analysis completed successfully for conversation_id: {conversation_id}")
    return APIResponse(
        status="success",
        timestamp=datetime.now(timezone.utc),
        data=analysis
    ).model_dump(exclude_none=True)


@app.post("/api/assist", response_model=APIResponse)
async def assist(
    request: AssistRequest,
    service: AssistService = Depends(get_assist_service),
):
    result = service.analyze(
        request.conversationId,
        request.transcript,
        request.complianceChecklist,
    )

    return APIResponse(
        status="success",
        timestamp=datetime.now(timezone.utc),
        data=result
    ).model_dump(exclude_none=True)


@app.post("/api/session/end", response_model=APIResponse)
async def session_end(
    request: SessionEndRequest,
    service: AssistService = Depends(get_assist_service),
):
    conv_id = request.conversationId

    # Consistent APIResponse error (was a raw dict before)
    if not conv_id:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                status="error",
                timestamp=datetime.now(timezone.utc),
                message="conversationId missing"
            ).model_dump()
        )

    service.end_session(conv_id)

    return APIResponse(
        status="success",
        timestamp=datetime.now(timezone.utc),
        data={"conversationId": conv_id}
    ).model_dump(exclude_none=True)


@app.post("/api/checklist/generate", response_model=APIResponse)
async def generate_checklist(
    file: UploadFile = File(...),
    service: ChecklistService = Depends(get_checklist_service),
):
    pdf_bytes = await file.read()
    checklist = service.generate_checklist(pdf_bytes)

    return APIResponse(
        status="success",
        timestamp=datetime.now(timezone.utc),
        data={"checklist": checklist}
    ).model_dump(exclude_none=True)