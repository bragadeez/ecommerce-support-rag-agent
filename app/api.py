import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .graph import run_pipeline
except ImportError:
    from graph import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="E-Commerce Support Resolution Agent",
    description="Multi-agent RAG system for automated support ticket resolution",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    ticket_id: Optional[str] = "TICKET-001"
    message: str
    order: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/resolve")
def resolve_ticket(request: TicketRequest):
    """
    Submit a support ticket for resolution.
    Returns structured resolution with decision, rationale, citations.
    """
    logger.info(f"Processing ticket: {request.ticket_id}")
    start = time.time()

    try:
        result = run_pipeline(
            ticket_dict={"message": request.message},
            order_dict=request.order,
            ticket_id=request.ticket_id,
        )
        elapsed = round(time.time() - start, 2)
        result["processing_time_seconds"] = elapsed
        logger.info(f"Ticket {request.ticket_id} resolved in {elapsed}s — decision: {result.get('decision')}")
        return result
    except Exception as e:
        logger.error(f"Pipeline error for {request.ticket_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
