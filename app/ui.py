import json
import time
import streamlit as st
from datetime import datetime, timedelta, date

try:
    from .graph import run_pipeline
except ImportError:
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
    .main-header { font-size: 1.8rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-header { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .section-label { font-size: 0.8rem; font-weight: 600; color: #888;
                     text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; }
    .item-card { background: #f8f9fa; border-radius: 8px; padding: 12px 14px;
                 margin-bottom: 10px; border: 1px solid #e9ecef; }
    .decision-badge { display: inline-block; padding: 4px 14px;
                      border-radius: 20px; font-weight: 600; font-size: 0.9rem; }
    .approve   { background: #d4edda; color: #155724; }
    .deny      { background: #f8d7da; color: #721c24; }
    .partial   { background: #fff3cd; color: #856404; }
    .escalate  { background: #cce5ff; color: #004085; }
    .need_more_info { background: #e2e3e5; color: #383d41; }
    .citation-box { background: #f0f4ff; border-left: 3px solid #4f46e5;
                    padding: 10px 14px; border-radius: 0 6px 6px 0;
                    margin: 6px 0; font-size: 0.85rem; line-height: 1.6; }
    .citation-source { color: #4f46e5; font-weight: 600; font-size: 0.8rem;
                       text-transform: uppercase; letter-spacing: 0.03em; }
    .compliance-pass { color: #155724; font-weight: 600; }
    .compliance-fail { color: #721c24; font-weight: 600; }
    div[data-testid="stHorizontalBlock"] { align-items: flex-end; }
</style>
""", unsafe_allow_html=True)


# ─── Sample Data ──────────────────────────────────────────────────────────────
# Stored as structured dicts, not JSON strings — maps directly to form fields

SAMPLES = {
    "Refund – within window": {
        "message": "Hi, I received my order 5 days ago but the laptop stand is not what I expected. It feels really cheap and wobbly. I'd like a full refund please.",
        "order_id": "ORD-20241201",
        "customer_name": "Sarah Chen",
        "customer_email": "sarah@example.com",
        "order_date": datetime.now().date() - timedelta(days=12),
        "payment_status": "paid",
        "shipping_status": "delivered",
        "delivery_date": datetime.now().date() - timedelta(days=5),
        "carrier": "FedEx",
        "tracking_number": "FX123456789",
        "items": [
            {"name": "Adjustable Laptop Stand", "category": "Electronics",
             "fulfillment": "First-party", "qty": 1, "price": 89.99},
        ],
    },
    "Damaged item claim": {
        "message": "My package arrived and the coffee maker inside was completely shattered. The box looked fine but the product is broken. I have photos. I need a replacement or refund.",
        "order_id": "ORD-20241115",
        "customer_name": "James Rivera",
        "customer_email": "james@example.com",
        "order_date": datetime.now().date() - timedelta(days=4),
        "payment_status": "paid",
        "shipping_status": "delivered",
        "delivery_date": datetime.now().date() - timedelta(days=2),
        "carrier": "UPS",
        "tracking_number": "1Z999AA10123456784",
        "items": [
            {"name": "Premium Coffee Maker", "category": "Home & Kitchen",
             "fulfillment": "First-party", "qty": 1, "price": 149.00},
        ],
    },
    "Lost package": {
        "message": "My order was supposed to arrive 3 weeks ago. The tracking hasn't updated in 18 days. I think it's lost. I need my money back or the items reshipped.",
        "order_id": "ORD-20241090",
        "customer_name": "Priya Patel",
        "customer_email": "priya@example.com",
        "order_date": datetime.now().date() - timedelta(days=30),
        "payment_status": "paid",
        "shipping_status": "in_transit",
        "delivery_date": None,
        "carrier": "USPS",
        "tracking_number": "9400111899223387623000",
        "items": [
            {"name": "Wireless Headphones", "category": "Electronics",
             "fulfillment": "First-party", "qty": 1, "price": 79.99},
            {"name": "Phone Case", "category": "Electronics",
             "fulfillment": "Marketplace", "qty": 2, "price": 14.99},
        ],
    },
    "Refund denied – outside window": {
        "message": "I bought a set of kitchen knives 75 days ago. I know it's been a while but I'd really like to return them — I never ended up using them.",
        "order_id": "ORD-20240901",
        "customer_name": "Tom Baker",
        "customer_email": "tom@example.com",
        "order_date": datetime.now().date() - timedelta(days=80),
        "payment_status": "paid",
        "shipping_status": "delivered",
        "delivery_date": datetime.now().date() - timedelta(days=75),
        "carrier": "FedEx",
        "tracking_number": "FX987654321",
        "items": [
            {"name": "Professional Kitchen Knife Set", "category": "Home & Kitchen",
             "fulfillment": "First-party", "qty": 1, "price": 199.00},
        ],
    },
}

ITEM_CATEGORIES = [
    "Electronics", "Home & Kitchen", "Apparel & Clothing",
    "Books & Media", "Sports & Outdoors", "Toys & Games",
    "Beauty & Personal Care", "Food & Perishables",
    "Digital Download", "Gift Card", "Other",
]

FULFILLMENT_TYPES = ["First-party", "Marketplace", "Third-party seller"]

PAYMENT_STATUSES = ["paid", "pending", "refunded", "failed", "partially_refunded"]

SHIPPING_STATUSES = ["processing", "in_transit", "delivered", "returned", "lost", "cancelled"]

CARRIERS = ["FedEx", "UPS", "USPS", "DHL", "Amazon Logistics", "Other", "N/A"]


# ─── Session State Initialiser ────────────────────────────────────────────────

def _init_state():
    """Set default session state values on first load."""
    defaults = {
        "message": "",
        "order_id": "ORD-00001",
        "customer_name": "",
        "customer_email": "",
        "order_date": datetime.now().date() - timedelta(days=7),
        "payment_status": "paid",
        "shipping_status": "delivered",
        "has_delivery_date": True,
        "delivery_date": datetime.now().date() - timedelta(days=2),
        "carrier": "FedEx",
        "tracking_number": "",
        "ticket_id": "TICKET-001",
        # Items stored as a list of dicts; start with one blank item
        "items": [
            {"name": "", "category": "Electronics",
             "fulfillment": "First-party", "qty": 1, "price": 0.0}
        ],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_sample(data: dict):

    # Clear old item widget keys
    for k in list(st.session_state.keys()):
        if k.startswith("item_"):
            del st.session_state[k]

    # Set items FIRST
    st.session_state["items"] = [dict(item) for item in data["items"]]

    # 🔑 Populate widget state explicitly
    for idx, item in enumerate(st.session_state["items"]):
        st.session_state[f"item_name_{idx}"] = item["name"]
        st.session_state[f"item_cat_{idx}"] = item["category"]
        st.session_state[f"item_ful_{idx}"] = item["fulfillment"]
        st.session_state[f"item_qty_{idx}"] = item["qty"]
        st.session_state[f"item_price_{idx}"] = item["price"]

    # Other fields
    for key in ["message", "order_id", "customer_name", "customer_email",
                "order_date", "payment_status", "shipping_status",
                "carrier", "tracking_number"]:
        if key in data:
            st.session_state[key] = data[key]

    if data.get("delivery_date"):
        st.session_state["has_delivery_date"] = True
        st.session_state["delivery_date"] = data["delivery_date"]
    else:
        st.session_state["has_delivery_date"] = False


# ─── Order JSON Builder ───────────────────────────────────────────────────────

def _build_order_dict() -> dict:
    """
    Assemble the order dict from session state.
    This is the internal representation passed to run_pipeline() — same
    schema as before, so the pipeline needs zero changes.
    """
    items = st.session_state["items"]
    item_list = [
        {
            "name": f"{it['name']} ({it['category']}, {it['fulfillment']})",
            "qty": int(it["qty"]),
            "price": float(it["price"]),
        }
        for it in items
        if it["name"].strip()   # skip rows with no name
    ]
    total = sum(it["qty"] * it["price"] for it in items if it["name"].strip())

    delivery_date = (
        st.session_state["delivery_date"].strftime("%Y-%m-%d")
        if st.session_state.get("has_delivery_date") and st.session_state.get("delivery_date")
        else None
    )

    return {
        "order_id": st.session_state["order_id"],
        "customer_name": st.session_state["customer_name"],
        "customer_email": st.session_state["customer_email"],
        "order_date": st.session_state["order_date"].strftime("%Y-%m-%d"),
        "items": item_list,
        "total_amount": round(total, 2),
        "payment_status": st.session_state["payment_status"],
        "shipping_status": st.session_state["shipping_status"],
        "delivery_date": delivery_date,
        "carrier": st.session_state["carrier"] if st.session_state["carrier"] != "N/A" else None,
        "tracking_number": st.session_state["tracking_number"] or None,
    }


# ─── Citation Formatter ───────────────────────────────────────────────────────

def format_citation_display(citation: dict) -> str:
    """
    Render a single citation in clean human-readable format for the UI.
    Strips any residual raw filenames/chunk IDs that may slip through.

    Input:  {"claim": "...", "source": "Refund Policy – Section 1 (Eligibility Window)", ...}
    Output: HTML string for st.markdown(..., unsafe_allow_html=True)
    """
    source = citation.get("source", "Policy document")
    claim  = citation.get("claim", "")

    # Strip any leftover raw artefacts (e.g. "refund_policy.txt (section 3)")
    # in case an older run slips through before the prompt change takes effect
    source = _clean_source_label(source)

    return (
        f'<div class="citation-box">'
        f'<span class="citation-source">{source}</span><br>'
        f'{claim}'
        f'</div>'
    )


def _clean_source_label(source: str) -> str:
    """
    Normalise source strings:
    - 'refund_policy.txt (section 3)'  → 'Refund Policy – Section 3'
    - Already-clean strings pass through unchanged.
    """
    import re

    # If it already looks like "Policy Name – Section X (...)" leave it
    if "–" in source or "—" in source:
        return source

    # Strip .txt extension
    source = re.sub(r"\.txt", "", source, flags=re.IGNORECASE)
    # Replace underscores with spaces
    source = source.replace("_", " ")
    # Normalise "(section N)" → "– Section N"
    source = re.sub(r"\(section\s*(\d+)\)", r"– Section \1", source, flags=re.IGNORECASE)
    # Title-case the policy name part
    parts = source.split("–", 1)
    parts[0] = parts[0].strip().title()
    return " – ".join(p.strip() for p in parts)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

_init_state()

with st.sidebar:
    st.markdown("### Sample Tickets")
    st.caption("Click to pre-fill the form:")

    for label, data in SAMPLES.items():
        if st.button(label, use_container_width=True):
            _load_sample(data)
            st.rerun()

    st.markdown("---")
    

# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛒 Support Resolution Agent</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Input Form
# ══════════════════════════════════════════════════════════════════════════════

with left:

    # ── Ticket ────────────────────────────────────────────────────────────────
    st.markdown("#### Customer Message")
    st.session_state["message"] = st.text_area(
        "Message",
        value=st.session_state["message"],
        height=120,
        placeholder="Paste or type the customer's support message…",
        label_visibility="collapsed",
    )

    st.session_state["ticket_id"] = st.text_input(
        "Ticket ID", value=st.session_state["ticket_id"]
    )

    st.markdown("---")

    # ── Customer Info ─────────────────────────────────────────────────────────
    st.markdown("#### Customer")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["customer_name"] = st.text_input(
            "Full name", value=st.session_state["customer_name"],
            placeholder="Jane Doe",
        )
    with c2:
        st.session_state["customer_email"] = st.text_input(
            "Email", value=st.session_state["customer_email"],
            placeholder="jane@example.com",
        )

    st.markdown("---")

    # ── Order Info ────────────────────────────────────────────────────────────
    st.markdown("#### Order Details")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.session_state["order_id"] = st.text_input(
            "Order ID", value=st.session_state["order_id"]
        )
    with r1c2:
        st.session_state["order_date"] = st.date_input(
            "Order date", value=st.session_state["order_date"]
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.session_state["payment_status"] = st.selectbox(
            "Payment status",
            PAYMENT_STATUSES,
            index=PAYMENT_STATUSES.index(st.session_state["payment_status"]),
        )
    with r2c2:
        st.session_state["shipping_status"] = st.selectbox(
            "Shipping / order status",
            SHIPPING_STATUSES,
            index=SHIPPING_STATUSES.index(st.session_state["shipping_status"]),
        )

    # Delivery date — optional (hidden when order is not yet delivered)
    st.session_state["has_delivery_date"] = st.checkbox(
        "Package has been delivered",
        value=st.session_state["has_delivery_date"],
    )
    if st.session_state["has_delivery_date"]:
        st.session_state["delivery_date"] = st.date_input(
            "Delivery date",
            value=st.session_state.get("delivery_date") or datetime.now().date(),
        )

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        carrier_idx = CARRIERS.index(st.session_state["carrier"]) if st.session_state["carrier"] in CARRIERS else 0
        st.session_state["carrier"] = st.selectbox("Carrier", CARRIERS, index=carrier_idx)
    with r3c2:
        st.session_state["tracking_number"] = st.text_input(
            "Tracking number (optional)",
            value=st.session_state["tracking_number"],
        )

    st.markdown("---")

    # ── Items ─────────────────────────────────────────────────────────────────
    st.markdown("#### Order Items")
    st.caption("Add one row per item. Click ✕ to remove.")

    items = st.session_state["items"]
    to_remove = None

    for idx, item in enumerate(items):
        with st.container():
            st.markdown('<div class="item-card">', unsafe_allow_html=True)

            # ── Row 1: name + delete ─────────────────────────────
            name_col, del_col = st.columns([5, 1])

            with name_col:
                st.text_input(
                    "Item name",
                    key=f"item_name_{idx}",
                    placeholder="e.g. Wireless Headphones",
                    label_visibility="collapsed" if idx > 0 else "visible",
                )

            with del_col:
                if len(items) > 1:
                    if st.button("✕", key=f"del_{idx}", help="Remove item"):
                        to_remove = idx

            # ── Row 2: category / fulfillment / qty / price ─────
            cat_col, ful_col, qty_col, price_col = st.columns([2, 2, 1, 1])

            with cat_col:
                st.selectbox(
                    "Category",
                    ITEM_CATEGORIES,
                    key=f"item_cat_{idx}",
                )

            with ful_col:
                st.selectbox(
                    "Fulfillment",
                    FULFILLMENT_TYPES,
                    key=f"item_ful_{idx}",
                )

            with qty_col:
                st.number_input(
                    "Qty",
                    min_value=1,
                    max_value=999,
                    key=f"item_qty_{idx}",
                )

            with price_col:
                st.number_input(
                    "Price ($)",
                    min_value=0.0,
                    step=0.01,
                    key=f"item_price_{idx}",
                    format="%.2f",
                )

            # ── 🔥 CRITICAL: Sync widget → items ─────────────────
            items[idx]["name"] = st.session_state.get(f"item_name_{idx}", "")
            items[idx]["category"] = st.session_state.get(f"item_cat_{idx}", ITEM_CATEGORIES[0])
            items[idx]["fulfillment"] = st.session_state.get(f"item_ful_{idx}", FULFILLMENT_TYPES[0])
            items[idx]["qty"] = st.session_state.get(f"item_qty_{idx}", 1)
            items[idx]["price"] = st.session_state.get(f"item_price_{idx}", 0.0)

            st.markdown('</div>', unsafe_allow_html=True)

    # Handle removal outside the loop to avoid mutation during iteration
    if to_remove is not None:
        st.session_state["items"].pop(to_remove)
        st.rerun()

    # Add item button
    if st.button("＋ Add another item", use_container_width=True):
        st.session_state["items"].append(
            {"name": "", "category": "Electronics",
             "fulfillment": "First-party", "qty": 1, "price": 0.0}
        )
        st.rerun()

    # Show computed total
    total = sum(
        it["qty"] * it["price"]
        for it in st.session_state["items"]
        if it["name"].strip()
    )
    st.markdown(f"**Order total: ${total:.2f}**")

    st.markdown("---")
    run_btn = st.button(
        "🚀 Run Resolution Pipeline",
        type="primary",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Output
# ══════════════════════════════════════════════════════════════════════════════

with right:
    if run_btn:
        # Validate required fields
        errors = []
        if not st.session_state["message"].strip():
            errors.append("Customer message is required.")
        if not st.session_state["customer_name"].strip():
            errors.append("Customer name is required.")
        if not any(it["name"].strip() for it in st.session_state["items"]):
            errors.append("At least one item with a name is required.")
        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        order_data = _build_order_dict()

        with st.spinner("Running 4-agent pipeline…"):
            start = time.time()
            try:
                result = run_pipeline(
                    ticket_dict={"message": st.session_state["message"]},
                    order_dict=order_data,
                    ticket_id=st.session_state["ticket_id"],
                )
                elapsed = round(time.time() - start, 2)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        # ── Decision header ────────────────────────────────────────────────
        decision = result.get("decision", "unknown")
        st.markdown(f"""
        <div style="display:flex; gap:12px; align-items:center; margin-bottom:14px">
            <span class="decision-badge {decision}">
                {decision.upper().replace('_', ' ')}
            </span>
            <span style="color:#666; font-size:0.85rem">
                · {result.get('classification','').replace('_',' ')}
                · {result.get('severity','')} severity
                · {elapsed}s
            </span>
        </div>
        """, unsafe_allow_html=True)

        conf = result.get("confidence_score", 0)
        st.progress(conf, text=f"Confidence: {conf:.0%}")

        if result.get("compliance_passed"):
            st.markdown('<span class="compliance-pass">✓ Compliance passed</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="compliance-fail">⚠ Compliance flagged issues</span>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Output tabs ────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "Customer Response", "Rationale & Citations",
            "Clarifying Questions", "Internal Notes",
        ])

        with tab1:
            st.markdown("**Send to customer:**")
            st.info(result.get("customer_response", ""))

        with tab2:
            st.markdown("**Decision rationale:**")
            st.write(result.get("rationale", ""))

            citations = result.get("citations", [])
            if citations:
                st.markdown(f"**Policy citations** ({len(citations)}):")
                for c in citations:
                    st.markdown(format_citation_display(c), unsafe_allow_html=True)
            else:
                st.markdown("**Policy citations:**")
                st.caption("No citations returned.")

            retrieved = result.get("retrieved_policies", [])

        with tab3:
            questions = result.get("clarifying_questions", [])
            if questions:
                for q in questions:
                    st.markdown(f"• {q}")
            else:
                st.caption("No clarifying questions needed.")

        with tab4:
            st.code(result.get("internal_notes", ""), language=None)

        with st.expander("Raw JSON output"):
            st.json(result)

    else:
        st.markdown("""
        <div style="height:420px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    color:#aaa; text-align:center;">
            <div style="font-size:3rem; margin-bottom:12px">🤖</div>
            <div style="font-size:1.1rem; font-weight:500">Ready to resolve tickets</div>
            <div style="font-size:0.85rem; margin-top:6px">
                Load a sample from the sidebar, or fill in the form and click Run
            </div>
        </div>
        """, unsafe_allow_html=True)
