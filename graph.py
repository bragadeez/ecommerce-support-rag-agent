from langgraph.graph import StateGraph, END
from typing import Literal

from schemas import AgentState
from agents import (
    triage_agent,
    policy_retriever_agent,
    resolution_writer_agent,
    compliance_agent,
    finalizer_node,
)

MAX_RETRIES = 2


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_after_triage(state: AgentState) -> Literal["policy_retriever", "finalizer"]:
    """Skip retrieval for spam/irrelevant tickets."""
    if state.triage.issue_type == "spam_or_irrelevant":
        return "finalizer"
    return "policy_retriever"


def route_after_compliance(
    state: AgentState,
) -> Literal["resolution_writer", "finalizer"]:
    """
    If compliance fails and we have retries left, send back to resolution writer.
    Otherwise proceed to finalizer regardless of compliance result.
    """
    compliance = state.compliance
    if compliance and not compliance.passed and compliance.requires_rewrite:
        if state.retry_count < MAX_RETRIES:
            state.retry_count += 1
            return "resolution_writer"
    return "finalizer"


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state machine.
    Returns a compiled graph ready to invoke.
    """

    # LangGraph requires dict-based state for its built-in state management.
    # We wrap our Pydantic AgentState into dicts at the boundaries.

    def triage_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = triage_agent(agent_state)
        return result.model_dump()

    def retriever_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = policy_retriever_agent(agent_state)
        return result.model_dump()

    def writer_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = resolution_writer_agent(agent_state)
        return result.model_dump()

    def compliance_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = compliance_agent(agent_state)
        return result.model_dump()

    def final_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = finalizer_node(agent_state)
        return result.model_dump()

    def _route_after_triage(state: dict) -> str:
        agent_state = AgentState(**state)
        return route_after_triage(agent_state)

    def _route_after_compliance(state: dict) -> str:
        agent_state = AgentState(**state)
        return route_after_compliance(agent_state)

    # Build the graph
    workflow = StateGraph(dict)

    workflow.add_node("triage", triage_node)
    workflow.add_node("policy_retriever", retriever_node)
    workflow.add_node("resolution_writer", writer_node)
    workflow.add_node("compliance", compliance_node)
    workflow.add_node("finalizer", final_node)

    # Entry point
    workflow.set_entry_point("triage")

    # Edges
    workflow.add_conditional_edges(
        "triage",
        _route_after_triage,
        {"policy_retriever": "policy_retriever", "finalizer": "finalizer"},
    )
    workflow.add_edge("policy_retriever", "resolution_writer")
    workflow.add_edge("resolution_writer", "compliance")
    workflow.add_conditional_edges(
        "compliance",
        _route_after_compliance,
        {"resolution_writer": "resolution_writer", "finalizer": "finalizer"},
    )
    workflow.add_edge("finalizer", END)

    return workflow.compile()


# Compiled graph singleton
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(ticket_dict: dict, order_dict: dict, ticket_id: str = "TICKET-001") -> dict:
    """
    Main entry point. Takes raw dicts (from API/UI) and returns final resolution dict.
    """
    from schemas import SupportTicket, OrderContext

    order = OrderContext(**order_dict)
    ticket = SupportTicket(
        ticket_id=ticket_id,
        customer_message=ticket_dict["message"],
        order_context=order,
    )

    initial_state = AgentState(ticket=ticket).model_dump()

    graph = get_graph()
    final_state_dict = graph.invoke(initial_state)
    final_state = AgentState(**final_state_dict)

    if final_state.final:
        return final_state.final.model_dump()
    else:
        return {"error": final_state.error or "Pipeline did not complete", "ticket_id": ticket_id}
