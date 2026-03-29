"""
schemas.py — All Pydantic models for agent I/O.
These are the contracts between agents. Strict types prevent silent failures.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─── Input Schemas ────────────────────────────────────────────────────────────

class OrderContext(BaseModel):
    order_id: str
    customer_name: str
    customer_email: str
    order_date: str
    items: list[dict]                      # [{name, qty, price}]
    total_amount: float
    payment_status: str                    # paid / pending / refunded
    shipping_status: str                   # delivered / in_transit / lost / returned
    delivery_date: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None


class SupportTicket(BaseModel):
    ticket_id: str
    customer_message: str
    order_context: OrderContext


# ─── Agent 1: Triage Output ───────────────────────────────────────────────────

class TriageResult(BaseModel):
    issue_type: Literal[
        "refund_request",
        "shipping_issue",
        "damaged_item",
        "wrong_item",
        "cancellation",
        "payment_issue",
        "general_inquiry",
        "spam_or_irrelevant",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    missing_info: list[str] = Field(
        default_factory=list,
        description="List of info needed before resolution can proceed"
    )
    key_facts: dict = Field(
        default_factory=dict,
        description="Extracted facts: amounts, dates, item names"
    )
    retrieval_queries: list[str] = Field(
        description="2-4 semantic queries to send to the policy vector DB"
    )
    can_resolve: bool = Field(
        description="False if missing_info is non-empty and blocks resolution"
    )


# ─── Agent 2: Policy Retriever Output ────────────────────────────────────────

class PolicyChunk(BaseModel):
    source: str                  # filename + section
    content: str                 # the actual text
    score: float                 # cosine similarity score
    chunk_id: str                # unique identifier


class RetrievalResult(BaseModel):
    chunks: list[PolicyChunk]
    queries_used: list[str]
    total_retrieved: int


# ─── Agent 3: Resolution Writer Output ───────────────────────────────────────

class Citation(BaseModel):
    claim: str          # the specific claim being cited
    source: str         # policy document + section
    chunk_id: str       # links back to the retrieved chunk


class ResolutionDraft(BaseModel):
    decision: Literal["approve", "deny", "partial", "escalate", "need_more_info"]
    rationale: str = Field(description="Internal reasoning, policy-grounded")
    citations: list[Citation]
    customer_response: str = Field(description="Friendly, professional customer-facing reply")
    internal_notes: str = Field(description="Notes for support team")
    clarifying_questions: list[str] = Field(
        default_factory=list,
        description="Questions to ask customer if info is missing"
    )
    confidence_score: float = Field(ge=0.0, le=1.0)


# ─── Agent 4: Compliance Output ───────────────────────────────────────────────

class ComplianceFlag(BaseModel):
    flag_type: Literal["hallucination", "missing_citation", "policy_violation", "tone_issue"]
    description: str
    location: str   # which part of the draft has the issue


class ComplianceResult(BaseModel):
    passed: bool
    flags: list[ComplianceFlag] = Field(default_factory=list)
    revised_customer_response: Optional[str] = None   # set if tone fix was minor
    requires_rewrite: bool = False


# ─── Final Output Schema ──────────────────────────────────────────────────────

class FinalResolution(BaseModel):
    ticket_id: str
    classification: str
    severity: str
    decision: str
    rationale: str
    citations: list[Citation]
    customer_response: str
    internal_notes: str
    clarifying_questions: list[str]
    compliance_passed: bool
    confidence_score: float
    retrieved_policies: list[str]   # just the source names for display


# ─── LangGraph State ──────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """
    The shared state object passed between all nodes in the LangGraph.
    Each agent reads what it needs and writes its own output key.
    """
    ticket: SupportTicket
    triage: Optional[TriageResult] = None
    retrieval: Optional[RetrievalResult] = None
    draft: Optional[ResolutionDraft] = None
    compliance: Optional[ComplianceResult] = None
    final: Optional[FinalResolution] = None
    retry_count: int = 0
    error: Optional[str] = None
