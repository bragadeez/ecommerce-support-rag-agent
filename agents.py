import json
import os
import re
import time
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from schemas import (
    AgentState, TriageResult, RetrievalResult, ResolutionDraft,
    ComplianceResult, ComplianceFlag, Citation, FinalResolution,
)
from retriever import retriever

load_dotenv()
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-PROVIDER LLM FALLBACK CHAIN
# ══════════════════════════════════════════════════════════════════════════════
#
# Provider priority order (all free):
#   1. Gemini 2.5 Flash   — primary
#   2. Gemini 2.5 Pro     — fallback 1
#   3. Groq Llama 3.3 70B     — fallback 2
#   4. Groq Llama 3.1 8B  — fallback 
#
# Add keys to .env:
#   GOOGLE_API_KEY=...
#   GROQ_API_KEY=...       ← free at console.groq.com
#
# The fallback fires ONLY on rate-limit or quota errors — not on JSON parse
# errors (those are retried on the same provider with a prompt tweak).
# ─────────────────────────────────────────────────────────────────────────────

# Errors that signal "try next provider" vs "something else is wrong"
_RATE_LIMIT_SIGNALS = (
    "429", "quota", "rate limit", "resource exhausted",
    "too many requests", "rate_limit_exceeded", "overloaded",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _RATE_LIMIT_SIGNALS)


def _build_provider_chain() -> list:
    chain = []

    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        chain.append((
            "gemini",  # ← vendor tag
            "Gemini 2.5 Flash",
            lambda: ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_key,
                temperature=0.0,
                convert_system_message_to_human=True,
            )
        ))

        chain.append((
            "gemini",
            "Gemini 2.5 Pro",
            lambda: ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=google_key,
                temperature=0.0,
                convert_system_message_to_human=True,
            )
        ))

    if groq_key:
        from langchain_groq import ChatGroq

        chain.append((
            "groq",
            "Groq Llama 3.3 70B",
            lambda: ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_key,
                temperature=0.0,
            )
        ))

        chain.append((
            "groq",
            "Groq Llama 3.1 8B",
            lambda: ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_key,
                temperature=0.0,
            )
        ))

    return chain


# Build the chain once at module load
_PROVIDER_CHAIN = _build_provider_chain()
logger.info(f"LLM provider chain: {[provider_name for _, provider_name, _ in _PROVIDER_CHAIN]}")

GLOBAL_BLOCKED_VENDORS = set()
def _call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    last_error = None
    global GLOBAL_BLOCKED_VENDORS  # ← KEY ADDITION

    for vendor, provider_name, factory in _PROVIDER_CHAIN:

        # 🚫 Skip blocked vendors (e.g. Gemini after rate limit)
        if vendor in GLOBAL_BLOCKED_VENDORS:
            continue

        llm = factory()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        for attempt in range(2):
            try:
                response = llm.invoke(messages)
                raw = response.content.strip()

                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw).strip()

                return json.loads(raw)

            except json.JSONDecodeError as e:
                if attempt == 0:
                    logger.warning(f"[{provider_name}] JSON error → retrying")
                    messages.append(HumanMessage(
                        content="Return ONLY valid JSON."
                    ))
                    continue
                else:
                    last_error = e
                    break

            except Exception as e:
                if _is_rate_limit_error(e):
                    logger.warning(f"[{provider_name}] Rate limited → blocking vendor '{vendor}'")

                    # 🚨 BLOCK ENTIRE VENDOR
                    GLOBAL_BLOCKED_VENDORS.add(vendor)

                    last_error = e
                    break  # move to next provider (different vendor)

                else:
                    raise

    raise RuntimeError(f"All providers exhausted. Last error: {last_error}")


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
    ticket = state.ticket
    order  = ticket.order_context

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
    """No LLM call — pure FAISS semantic search."""
    queries = state.triage.retrieval_queries
    state.retrieval = retriever.retrieve(queries)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3: RESOLUTION WRITER AGENT
# ══════════════════════════════════════════════════════════════════════════════
#
# KEY CHANGE: Citation format is now strictly human-readable.
# The prompt instructs the model to produce:
#   "Policy Name – Section N (Topic Description)"
# e.g. "Refund Policy – Section 1 (Eligibility Window)"
#
# Raw filenames like "refund_policy.txt (section 1)" are explicitly forbidden.
# ─────────────────────────────────────────────────────────────────────────────

RESOLUTION_SYSTEM_PROMPT = """You are a resolution writer for an e-commerce customer support system.

Your job is to write a complete resolution based STRICTLY on the provided policy context.

ABSOLUTE RULES — VIOLATION OF THESE WILL CAUSE SYSTEM FAILURE:
1. NEVER make any claim not directly supported by the provided policy context.
2. EVERY factual statement in rationale and customer_response MUST have a citation.
3. If the policy context does not cover the situation, set decision="escalate".
4. If information is missing, set decision="need_more_info" and list clarifying_questions.
5. Do NOT invent policy rules, numbers, or timeframes not present in the context.
6. Write INSUFFICIENT_CONTEXT in rationale if you cannot find policy support.
7. customer_response must be professional, empathetic, and clear. No jargon.
8. Output ONLY valid JSON. No explanation, no markdown, no extra text.

CITATION FORMAT RULES (strictly enforced):
- Source MUST follow this exact format:  "Policy Name – Section N (Topic Description)"
- Examples of CORRECT sources:
    "Refund Policy – Section 1 (Eligibility Window)"
    "Shipping Policy – Section 3 (Lost Packages)"
    "Cancellation Policy – Section 2 (Pre-Shipment Cancellations)"
    "Compensation Policy – Section 1 (Compensation for Our Errors)"
- NEVER include raw filenames (no ".txt", no underscores)
- NEVER include chunk IDs or internal identifiers
- Derive the Policy Name from the document title (e.g. "refund_policy" → "Refund Policy")
- Derive the Topic Description from the section heading in the policy text
- If a section number is not clear, use "Section 1"

OUTPUT FORMAT (strict JSON):
{
  "decision": "<approve|deny|partial|escalate|need_more_info>",
  "rationale": "<internal reasoning with policy references>",
  "citations": [
    {
      "claim": "<specific claim being supported>",
      "source": "<Policy Name – Section N (Topic Description)>",
      "chunk_id": "<copy chunk_id from the POLICY SOURCE label, e.g. refund_policy.txt::chunk_0>"
    }
  ],
  "customer_response": "<friendly customer-facing reply>",
  "internal_notes": "<notes for support team>",
  "clarifying_questions": [],
  "confidence_score": 0.95
}"""


def resolution_writer_agent(state: AgentState) -> AgentState:
    ticket   = state.ticket
    order    = ticket.order_context
    triage   = state.triage
    retrieval = state.retrieval

    policy_context = retriever.format_context(retrieval)

    if retrieval.total_retrieved == 0:
        state.draft = ResolutionDraft(
            decision="escalate",
            rationale="INSUFFICIENT_CONTEXT: No relevant policy was found for this issue type.",
            citations=[],
            customer_response=(
                f"Dear {order.customer_name},\n\n"
                "Thank you for reaching out. Your case requires review by our specialised team. "
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

Write the complete resolution JSON.
IMPORTANT: Every citation source MUST be formatted as "Policy Name – Section N (Topic)"."""

    result_dict = _call_llm_json(RESOLUTION_SYSTEM_PROMPT, user_prompt)

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
    draft    = state.draft
    retrieval = state.retrieval

    policy_context = retriever.format_context(retrieval)
    citations_str  = json.dumps(
        [{"claim": c.claim, "source": c.source} for c in draft.citations],
        indent=2,
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
# FINALIZER
# ══════════════════════════════════════════════════════════════════════════════

def finalizer_node(state: AgentState) -> AgentState:
    draft      = state.draft
    triage     = state.triage
    compliance = state.compliance
    retrieval  = state.retrieval

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
