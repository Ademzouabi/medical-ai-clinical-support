"""HTTP transport layer for the deterministic V1 pipeline."""

import logging
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import (
    ClarificationUnavailableError,
    PredictionNotReadyError,
    can_predict,
    get_current_state,
    predict as run_prediction,
    start_pipeline,
    submit_clarification,
)


logger = logging.getLogger(__name__)

# The API is deliberately a thin transport layer; pipeline.py retains business logic.
app = FastAPI(title="Clinical AI V1 API", version="1.0.0")

# Allow a local browser client to call this development API from another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    # Pydantic validates this JSON body before the route function receives it.
    text: str


class SubmitAnswerRequest(BaseModel):
    answer: str


# V1 keeps sessions only in process memory; restarting the server discards them.
sessions: dict[str, Any] = {}


def _session_or_404(session_id: str) -> Any:
    # Centralize missing-session behavior so every route returns the same 404 contract.
    try:
        return sessions[session_id]
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "session_not_found", "message": "Unknown session ID."},
        ) from error


def _state_response(session_id: str, session: Any) -> dict[str, Any]:
    # Add transport identity without duplicating pipeline state-building logic.
    return {"session_id": session_id, **get_current_state(session)}


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    """Keep unanticipated server errors out of the client response."""

    # Log technical details on the server while returning a safe generic client message.
    logger.exception("Unhandled API error for %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "code": "internal_error",
                "message": "Internal server error.",
            }
        },
    )


@app.post("/sessions", status_code=201)
def create_session(payload: CreateSessionRequest) -> dict[str, Any]:
    """Create a server-side pipeline session from clinical text."""

    # UUIDs avoid exposing predictable sequential identifiers to API callers.
    session_id = str(uuid4())
    session = start_pipeline(payload.text)
    sessions[session_id] = session
    return _state_response(session_id, session)


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Return the current state of an existing pipeline session."""

    return _state_response(session_id, _session_or_404(session_id))


@app.post("/sessions/{session_id}/answers")
def answer_session(session_id: str, payload: SubmitAnswerRequest) -> dict[str, Any]:
    """Submit one clarification answer through the existing pipeline."""

    session = _session_or_404(session_id)
    # Pipeline exceptions become HTTP conflict responses instead of uncaught server errors.
    try:
        submit_clarification(session, payload.answer)
    except (ClarificationUnavailableError, RuntimeError) as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "clarification_unavailable", "message": str(error)},
        ) from error
    return _state_response(session_id, session)


@app.post("/sessions/{session_id}/predict")
def predict_session(session_id: str) -> dict[str, Any]:
    """Run the existing prediction function only after its binary gate opens."""

    session = _session_or_404(session_id)
    # Never let an incomplete or UNKNOWN feature vector reach the trained model.
    if not can_predict(session):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "prediction_not_ready",
                "message": "Clarification is incomplete; prediction is not allowed.",
            },
        )

    try:
        predictions = run_prediction(session)
    except PredictionNotReadyError as error:
        raise HTTPException(
            status_code=409,
            detail={"code": "prediction_not_ready", "message": str(error)},
        ) from error

    return {"session_id": session_id, "predictions": predictions}
