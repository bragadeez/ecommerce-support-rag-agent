"""
agents.py — The four core agents.

Each agent is a pure function: (AgentState) → AgentState.
LangGraph calls them as graph nodes.

LLM: Gemini 1.5 Flash (free tier via Google AI Studio)
"""

import json
import os
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from schemas import (
    AgentState,
    TriageResult,
    RetrievalResult,
    ResolutionDraft,
    ComplianceResult,
    ComplianceFlag,
    Citation,
    FinalResolution,
)
from retriever import retriever

load_dotenv()

# ─── LLM Setup ────────────────────────────────────────────────────────────────
# temperature=0 for deterministic, policy-grounded responses
# No creativity wanted here — we need precise, consistent decisions.

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
        convert_system_message_to_human=True,
    )


def _call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Call Gemini and parse JSON response. Strips markdown fences if present.
    Raises ValueError if response is not valid JSON.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if model wrapped the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw response:\n{raw}")


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1: TRIAGE AGENT
# ══════════════════════════════════════════════════════════════════════════════

TRIAGE_SYSTEM_PROMPT = """You are a triage agent for an e-commerce customer support system.
Your ONLY job is to analyze a support ticket and produce a structured JSON output.

STRICT RULES:
1. Classify the issue into exactly one category.
2. Extract all verifiable facts from the ticket and order context.
3. List EVERY piece of missing information that would be needed to resolve the issue.
4. Generate 2-4 specific retrieval queries to search the policy knowledge base.
   - Queries should be specific, not generic. Include relevant numbers/details.
   - Example GOOD: "refund eligibility window days after delivery"
   - Example BAD: "refund policy"
5. Set can_resolve=false if missing_info is non-empty.
6. Output ONLY valid JSON. No explanation, no markdown, no extra text.

OUTPUT FORMAT (strict JSON):
{
  "issue_type": "<one of: refund_request|shipping_issue|damaged_item|wrong_item|cancellation|payment_issue|general_inquiry|spam_or_irrelevant>",
  "severity": "<low|medium|high|critical>",
  "missing_info": ["<item1>", "<item2>"],
  "key_facts": {"key": "value"},
  "retrieval_queries": ["<query1>", "<query2>"],
  "can_resolve": true
}"""


def triage_agent(state: AgentState) -> AgentState:
    """
    Agent 1: Classify ticket, extract facts, identify gaps, generate retrieval queries.
    """
    ticket = state.ticket
    order = ticket.order_context

    user_prompt = f"""SUPPORT TICKET:
Ticket ID: {ticket.ticket_id}
Customer Message: {ticket.customer_message}

ORDER CONTEXT:
Order ID: {order.order_id}
Customer: {order.customer_name} ({order.customer_email})
Order Date: {order.order_date}
Items: {json.dumps(order.items, indent=2)}
Total Amount: ${order.total_amount}
Payment Status: {order.payment_status}
Shipping Status: {order.shipping_status}
Delivery Date: {order.delivery_date or 'Not yet delivered'}
Carrier: {order.carrier or 'Unknown'}
Tracking: {order.tracking_number or 'Not provided'}

Analyze this ticket and return the JSON triage output."""

    result_dict = _call_llm_json(TRIAGE_SYSTEM_PROMPT, user_prompt)
    state.triage = TriageResult(**result_dict)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2: POLICY RETRIEVER AGENT
# ══════════════════════════════════════════════════════════════════════════════

def policy_retriever_agent(state: AgentState) -> AgentState:
    """
    Agent 2: Use triage queries to retrieve relevant policy chunks from FAISS.
    No LLM call here — pure semantic search. Fast and deterministic.
    """
    queries = state.triage.retrieval_queries
    result = retriever.retrieve(queries)
    state.retrieval = result
    return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3: RESOLUTION WRITER AGENT
# ══════════════════════════════════════════════════════════════════════════════

RESOLUTION_SYSTEM_PROMPT = """You are a resolution writer for an e-commerce customer support system.

Your job is to write a complete resolution based STRICTLY on the provided policy context.

ABSOLUTE RULES — VIOLATION OF THESE WILL CAUSE SYSTEM FAILURE:
1. NEVER make any claim not directly supported by the provided policy context.
2. EVERY factual statement in rationale and customer_response MUST have a citation.
3. If the policy context does not cover the situation, set decision="escalate" and explain why.
4. If information is missing, set decision="need_more_info" and list clarifying_questions.
5. Do NOT invent policy rules, numbers, or timeframes not present in the context.
6. Write INSUFFICIENT_CONTEXT in rationale if you cannot find policy support for a decision.
7. customer_response must be professional, empathetic, and clear. No jargon.
8. Output ONLY valid JSON. No explanation, no markdown, no extra text.

CITATION FORMAT: For each claim, reference the exact "POLICY SOURCE N: filename (section X)" label.

OUTPUT FORMAT (strict JSON):
{
  "decision": "<approve|deny|partial|escalate|need_more_info>",
  "rationale": "<internal reasoning with policy references>",
  "citations": [
    {"claim": "<specific claim>", "source": "<policy source>", "chunk_id": "<id>"}
  ],
  "customer_response": "<friendly customer-facing reply>",
  "internal_notes": "<notes for support team>",
  "clarifying_questions": ["<question1>"],
  "confidence_score": 0.95
}"""


def resolution_writer_agent(state: AgentState) -> AgentState:
    """
    Agent 3: Draft a complete resolution using retrieved policy context.
    Strictly grounded — cites every claim.
    """
    ticket = state.ticket
    order = ticket.order_context
    triage = state.triage
    retrieval = state.retrieval

    # Build context string from retrieved chunks
    policy_context = retriever.format_context(retrieval)

    # If no relevant policy found, escalate immediately
    if retrieval.total_retrieved == 0:
        state.draft = ResolutionDraft(
            decision="escalate",
            rationale="INSUFFICIENT_CONTEXT: No relevant policy was found for this issue type.",
            citations=[],
            customer_response=(
                f"Dear {order.customer_name},\n\n"
                "Thank you for reaching out. Your case requires review by our specialized team. "
                "A senior support agent will contact you within 1 business day.\n\n"
                "Best regards,\nSupport Team"
            ),
            internal_notes="No policy context retrieved. Escalate to senior team.",
            clarifying_questions=[],
            confidence_score=0.1,
        )
        return state

    user_prompt = f"""TICKET INFORMATION:
Ticket ID: {ticket.ticket_id}
Customer: {order.customer_name}
Issue Type: {triage.issue_type}
Severity: {triage.severity}
Customer Message: {ticket.customer_message}

ORDER FACTS:
- Order ID: {order.order_id}
- Order Date: {order.order_date}
- Delivery Date: {order.delivery_date or 'Not delivered'}
- Total Amount: ${order.total_amount}
- Payment Status: {order.payment_status}
- Shipping Status: {order.shipping_status}
- Items: {json.dumps(order.items)}

EXTRACTED KEY FACTS FROM TRIAGE:
{json.dumps(triage.key_facts, indent=2)}

MISSING INFORMATION:
{json.dumps(triage.missing_info)}

RETRIEVED POLICY CONTEXT (use ONLY this to make decisions):
{policy_context}

Write the complete resolution JSON. Remember: cite every factual claim."""

    result_dict = _call_llm_json(RESOLUTION_SYSTEM_PROMPT, user_prompt)

    # Parse citations carefully
    raw_citations = result_dict.pop("citations", [])
    citations = []
    for c in raw_citations:
        try:
            citations.append(Citation(
                claim=c.get("claim", ""),
                source=c.get("source", ""),
                chunk_id=c.get("chunk_id", ""),
            ))
        except Exception:
            pass

    result_dict["citations"] = citations
    state.draft = ResolutionDraft(**result_dict)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4: COMPLIANCE / SAFETY AGENT
# ══════════════════════════════════════════════════════════════════════════════

COMPLIANCE_SYSTEM_PROMPT = """You are a compliance and safety checker for an e-commerce support system.

Your job is to verify that the resolution draft is:
1. Grounded — every factual claim is supported by the provided policy context
2. Cited — every claim has a valid citation
3. Safe — no inappropriate, offensive, or legally risky language
4. Accurate — no wrong numbers, dates, or policy rules

HALLUCINATION CHECK PROCESS:
- Read each claim in the rationale and customer_response
- Find its supporting citation
- Verify the citation actually supports the claim in the provided policy context
- Flag if the cited source doesn't contain the claimed information

OUTPUT ONLY valid JSON. No markdown, no explanation.

OUTPUT FORMAT:
{
  "passed": true,
  "flags": [
    {
      "flag_type": "<hallucination|missing_citation|policy_violation|tone_issue>",
      "description": "<what is wrong>",
      "location": "<which part of the draft>"
    }
  ],
  "revised_customer_response": null,
  "requires_rewrite": false
}"""


def compliance_agent(state: AgentState) -> AgentState:
    """
    Agent 4: Verify the draft against retrieved policy.
    Sets requires_rewrite=True if hallucinations or missing citations found.
    """
    draft = state.draft
    retrieval = state.retrieval

    policy_context = retriever.format_context(retrieval)
    citations_str = json.dumps(
        [{"claim": c.claim, "source": c.source} for c in draft.citations],
        indent=2
    )

    user_prompt = f"""RESOLUTION DRAFT TO CHECK:

Decision: {draft.decision}
Rationale: {draft.rationale}
Customer Response: {draft.customer_response}
Internal Notes: {draft.internal_notes}
Confidence Score: {draft.confidence_score}

CITATIONS PROVIDED:
{citations_str}

POLICY CONTEXT (ground truth):
{policy_context}

Verify every claim. Return the compliance JSON."""

    result_dict = _call_llm_json(COMPLIANCE_SYSTEM_PROMPT, user_prompt)

    raw_flags = result_dict.pop("flags", [])
    flags = []
    for f in raw_flags:
        try:
            flags.append(ComplianceFlag(
                flag_type=f.get("flag_type", "hallucination"),
                description=f.get("description", ""),
                location=f.get("location", ""),
            ))
        except Exception:
            pass

    result_dict["flags"] = flags
    state.compliance = ComplianceResult(**result_dict)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# FINALIZER NODE (not an agent — just assembles the final output)
# ══════════════════════════════════════════════════════════════════════════════

def finalizer_node(state: AgentState) -> AgentState:
    """
    Assemble everything into the FinalResolution object.
    Uses compliance-revised response if available.
    """
    draft = state.draft
    triage = state.triage
    compliance = state.compliance
    retrieval = state.retrieval

    # Use compliance-revised response if the compliance agent made a minor fix
    final_response = (
        compliance.revised_customer_response
        if compliance and compliance.revised_customer_response
        else draft.customer_response
    )

    state.final = FinalResolution(
        ticket_id=state.ticket.ticket_id,
        classification=triage.issue_type,
        severity=triage.severity,
        decision=draft.decision,
        rationale=draft.rationale,
        citations=draft.citations,
        customer_response=final_response,
        internal_notes=draft.internal_notes,
        clarifying_questions=draft.clarifying_questions,
        compliance_passed=compliance.passed if compliance else False,
        confidence_score=draft.confidence_score,
        retrieved_policies=list({c.source for c in retrieval.chunks}),
    )
    return state
