"""
ui.py — Streamlit interface for the E-Commerce Support Resolution Agent.
Run: streamlit run ui.py
"""

import json
import time
import streamlit as st
from datetime import datetime, timedelta

from graph import run_pipeline

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support Resolution Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .decision-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .approve { background: #d4edda; color: #155724; }
    .deny { background: #f8d7da; color: #721c24; }
    .partial { background: #fff3cd; color: #856404; }
    .escalate { background: #cce5ff; color: #004085; }
    .need_more_info { background: #e2e3e5; color: #383d41; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #4f46e5;
    }
    .citation-box {
        background: #f0f4ff;
        border-left: 3px solid #4f46e5;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .compliance-pass { color: #155724; font-weight: 600; }
    .compliance-fail { color: #721c24; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar: Sample Data Loader ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Sample Tickets")
    st.markdown("Click to load a pre-built test case:")

    SAMPLES = {
        "Refund - within window": {
            "message": "Hi, I received my order 5 days ago but the laptop stand is not what I expected. It feels really cheap and wobbly. I'd like a full refund please. Order was $89.99.",
            "order": {
                "order_id": "ORD-20241201",
                "customer_name": "Sarah Chen",
                "customer_email": "sarah@example.com",
                "order_date": (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d"),
                "items": [{"name": "Adjustable Laptop Stand", "qty": 1, "price": 89.99}],
                "total_amount": 89.99,
                "payment_status": "paid",
                "shipping_status": "delivered",
                "delivery_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                "carrier": "FedEx",
                "tracking_number": "FX123456789",
            },
        },
        "Damaged item claim": {
            "message": "My package arrived and the coffee maker inside was completely shattered. The box looked fine but the product is broken. I have photos. I need either a replacement or refund.",
            "order": {
                "order_id": "ORD-20241115",
                "customer_name": "James Rivera",
                "customer_email": "james@example.com",
                "order_date": (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"),
                "items": [{"name": "Premium Coffee Maker", "qty": 1, "price": 149.00}],
                "total_amount": 149.00,
                "payment_status": "paid",
                "shipping_status": "delivered",
                "delivery_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "carrier": "UPS",
                "tracking_number": "1Z999AA10123456784",
            },
        },
        "Lost package": {
            "message": "My order was supposed to arrive 3 weeks ago. The tracking hasn't updated in 18 days. I think it's lost. I need my money back or the items reshipped.",
            "order": {
                "order_id": "ORD-20241090",
                "customer_name": "Priya Patel",
                "customer_email": "priya@example.com",
                "order_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "items": [
                    {"name": "Wireless Headphones", "qty": 1, "price": 79.99},
                    {"name": "Phone Case", "qty": 2, "price": 14.99},
                ],
                "total_amount": 109.97,
                "payment_status": "paid",
                "shipping_status": "in_transit",
                "delivery_date": None,
                "carrier": "USPS",
                "tracking_number": "9400111899223387623000",
            },
        },
        "Refund denied - outside window": {
            "message": "I bought a set of kitchen knives 75 days ago. I know it's been a while but I'd really like to return them — I never ended up using them.",
            "order": {
                "order_id": "ORD-20240901",
                "customer_name": "Tom Baker",
                "customer_email": "tom@example.com",
                "order_date": (datetime.now() - timedelta(days=80)).strftime("%Y-%m-%d"),
                "items": [{"name": "Professional Kitchen Knife Set", "qty": 1, "price": 199.00}],
                "total_amount": 199.00,
                "payment_status": "paid",
                "shipping_status": "delivered",
                "delivery_date": (datetime.now() - timedelta(days=75)).strftime("%Y-%m-%d"),
                "carrier": "FedEx",
                "tracking_number": "FX987654321",
            },
        },
    }

    for label, data in SAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state.sample_message = data["message"]
            st.session_state.sample_order = json.dumps(data["order"], indent=2)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
**4-agent RAG pipeline:**
1. Triage Agent
2. Policy Retriever
3. Resolution Writer
4. Compliance Checker

**Free models:**
- LLM: Gemini 1.5 Flash
- Embeddings: all-MiniLM-L6-v2
- Vector DB: FAISS (local)
    """)

# ─── Main Layout ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛒 Support Resolution Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-agent RAG system · Gemini 1.5 Flash · FAISS · LangGraph</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### Customer Ticket")
    message = st.text_area(
        "Customer message",
        value=st.session_state.get("sample_message", ""),
        height=150,
        placeholder="Enter the customer's support message...",
    )

    st.markdown("#### Order Context (JSON)")
    order_json = st.text_area(
        "Order data",
        value=st.session_state.get("sample_order", json.dumps({
            "order_id": "ORD-12345",
            "customer_name": "Jane Doe",
            "customer_email": "jane@example.com",
            "order_date": "2024-11-15",
            "items": [{"name": "Product Name", "qty": 1, "price": 49.99}],
            "total_amount": 49.99,
            "payment_status": "paid",
            "shipping_status": "delivered",
            "delivery_date": "2024-11-20",
            "carrier": "FedEx",
            "tracking_number": "FX123456789",
        }, indent=2)),
        height=280,
    )

    ticket_id = st.text_input("Ticket ID", value="TICKET-001")

    run_btn = st.button("🚀 Run Resolution Pipeline", type="primary", use_container_width=True)

# ─── Output Column ────────────────────────────────────────────────────────────
with col2:
    if run_btn:
        try:
            order_data = json.loads(order_json)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in order context: {e}")
            st.stop()

        with st.spinner("Running 4-agent pipeline..."):
            start = time.time()
            try:
                result = run_pipeline(
                    ticket_dict={"message": message},
                    order_dict=order_data,
                    ticket_id=ticket_id,
                )
                elapsed = round(time.time() - start, 2)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        # ── Summary Row ──
        decision = result.get("decision", "unknown")
        badge_class = decision.replace("_", "")
        st.markdown(f"""
        <div style="display:flex; gap:12px; align-items:center; margin-bottom:16px">
            <span class="decision-badge {decision}">{decision.upper().replace('_',' ')}</span>
            <span style="color:#666; font-size:0.85rem">· {result.get('classification','').replace('_',' ')} · {result.get('severity','')} severity · {elapsed}s</span>
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        conf = result.get("confidence_score", 0)
        st.progress(conf, text=f"Confidence: {conf:.0%}")

        # Compliance badge
        if result.get("compliance_passed"):
            st.markdown('<span class="compliance-pass">✓ Compliance check passed</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="compliance-fail">⚠ Compliance flagged issues</span>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Tabs for output sections ──
        tab1, tab2, tab3, tab4 = st.tabs(["Customer Response", "Rationale & Citations", "Clarifying Questions", "Internal Notes"])

        with tab1:
            st.markdown("**Send to customer:**")
            st.info(result.get("customer_response", ""))

        with tab2:
            st.markdown("**Decision rationale:**")
            st.write(result.get("rationale", ""))

            st.markdown("**Policy citations:**")
            citations = result.get("citations", [])
            if citations:
                for c in citations:
                    st.markdown(f"""
                    <div class="citation-box">
                        <strong>Claim:</strong> {c.get('claim','')}<br>
                        <strong>Source:</strong> {c.get('source','')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No citations")

            st.markdown("**Policies retrieved:**")
            for p in result.get("retrieved_policies", []):
                st.caption(f"• {p}")

        with tab3:
            questions = result.get("clarifying_questions", [])
            if questions:
                for q in questions:
                    st.markdown(f"• {q}")
            else:
                st.caption("No clarifying questions needed.")

        with tab4:
            st.code(result.get("internal_notes", ""), language=None)

        # ── Raw JSON ──
        with st.expander("Raw JSON output"):
            st.json(result)
    else:
        st.markdown("""
        <div style="height:400px; display:flex; flex-direction:column; align-items:center;
                    justify-content:center; color:#aaa; text-align:center;">
            <div style="font-size:3rem; margin-bottom:12px">🤖</div>
            <div style="font-size:1.1rem; font-weight:500">Ready to resolve tickets</div>
            <div style="font-size:0.85rem; margin-top:6px">Load a sample from the sidebar or enter ticket details</div>
        </div>
        """, unsafe_allow_html=True)
