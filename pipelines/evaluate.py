"""
evaluate.py — Run 20 test cases and compute evaluation metrics.

Run: python evaluate.py
     python evaluate.py --delay 6     ← increase delay between cases
     python evaluate.py --fast        ← no delay (only safe with Groq keys)
"""

import json
import time
import argparse
import statistics
from datetime import datetime, timedelta
from pathlib import Path

from app.graph import run_pipeline

# ─── CLI Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=float, default=4.0,
                    help="Seconds between test cases (default: 4)")
parser.add_argument("--fast", action="store_true",
                    help="No inter-case delay (use only with Groq API key)")
args, _ = parser.parse_known_args()

INTER_CASE_DELAY = 0 if args.fast else args.delay

# ─── Helpers ──────────────────────────────────────────────────────────────────

def days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


# ─── 20 Test Cases ────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "TC-001",
        "description": "Refund within 30-day window — APPROVE",
        "expected_decision": "approve",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I received my order 10 days ago and I'm not satisfied with the quality. I'd like a full refund.",
        "order": {
            "order_id": "ORD-001", "customer_name": "Alice Johnson",
            "customer_email": "alice@test.com", "order_date": days_ago(15),
            "items": [{"name": "Ceramic Mug Set", "qty": 1, "price": 34.99}],
            "total_amount": 34.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(10),
            "carrier": "USPS", "tracking_number": "TRK001",
        },
    },
    {
        "id": "TC-002",
        "description": "Refund outside 60-day window — DENY",
        "expected_decision": "deny",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I bought a pair of headphones 70 days ago and I want to return them.",
        "order": {
            "order_id": "ORD-002", "customer_name": "Bob Smith",
            "customer_email": "bob@test.com", "order_date": days_ago(75),
            "items": [{"name": "Wireless Headphones", "qty": 1, "price": 79.99}],
            "total_amount": 79.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(70),
            "carrier": "FedEx", "tracking_number": "TRK002",
        },
    },
    {
        "id": "TC-003",
        "description": "Refund 31-60 days — store credit only (PARTIAL)",
        "expected_decision": "partial",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "My item arrived 40 days ago. It doesn't work properly. Can I get a refund?",
        "order": {
            "order_id": "ORD-003", "customer_name": "Carol White",
            "customer_email": "carol@test.com", "order_date": days_ago(45),
            "items": [{"name": "Electric Kettle", "qty": 1, "price": 44.99}],
            "total_amount": 44.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(40),
            "carrier": "UPS", "tracking_number": "TRK003",
        },
    },
    {
        "id": "TC-004",
        "description": "Digital download refund — DENY (non-returnable)",
        "expected_decision": "deny",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I accidentally bought the wrong digital software license. Can I get a refund?",
        "order": {
            "order_id": "ORD-004", "customer_name": "David Lee",
            "customer_email": "david@test.com", "order_date": days_ago(2),
            "items": [{"name": "Digital Software License (Download)", "qty": 1, "price": 59.99}],
            "total_amount": 59.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(2),
            "carrier": None, "tracking_number": None,
        },
    },
    {
        "id": "TC-005",
        "description": "Damaged item within 7-day report window — APPROVE",
        "expected_decision": "approve",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "My blender arrived yesterday and it's cracked. The glass pitcher has a large crack. I have photos. I want a replacement or full refund.",
        "order": {
            "order_id": "ORD-005", "customer_name": "Emma Davis",
            "customer_email": "emma@test.com", "order_date": days_ago(5),
            "items": [{"name": "Glass Blender", "qty": 1, "price": 89.99}],
            "total_amount": 89.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(1),
            "carrier": "FedEx", "tracking_number": "TRK005",
        },
    },
    {
        "id": "TC-006",
        "description": "Damaged item reported after 7 days — ESCALATE",
        "expected_decision": "escalate",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "My item arrived damaged but I only noticed it now, 12 days after delivery.",
        "order": {
            "order_id": "ORD-006", "customer_name": "Frank Brown",
            "customer_email": "frank@test.com", "order_date": days_ago(16),
            "items": [{"name": "Kitchen Scale", "qty": 1, "price": 29.99}],
            "total_amount": 29.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(12),
            "carrier": "USPS", "tracking_number": "TRK006",
        },
    },
    {
        "id": "TC-007",
        "description": "Lost package within 60 days — APPROVE",
        "expected_decision": "approve",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "My order was supposed to arrive 20 days ago. Tracking has not updated in 3 weeks. I think the package is lost.",
        "order": {
            "order_id": "ORD-007", "customer_name": "Grace Kim",
            "customer_email": "grace@test.com", "order_date": days_ago(35),
            "items": [{"name": "Fitness Tracker", "qty": 1, "price": 59.99}],
            "total_amount": 59.99, "payment_status": "paid",
            "shipping_status": "in_transit", "delivery_date": None,
            "carrier": "USPS", "tracking_number": "TRK007",
        },
    },
    {
        "id": "TC-008",
        "description": "Package marked delivered but not received — ESCALATE (carrier investigation)",
        "expected_decision": "escalate",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "The tracking shows my package was delivered yesterday but I never received it. I checked everywhere.",
        "order": {
            "order_id": "ORD-008", "customer_name": "Henry Wilson",
            "customer_email": "henry@test.com", "order_date": days_ago(8),
            "items": [{"name": "Smart Watch", "qty": 1, "price": 199.99}],
            "total_amount": 199.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(1),
            "carrier": "UPS", "tracking_number": "TRK008",
        },
    },
    {
        "id": "TC-009",
        "description": "Cancellation within 1 hour — APPROVE",
        "expected_decision": "approve",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I just placed an order 20 minutes ago by mistake. Can you please cancel it?",
        "order": {
            "order_id": "ORD-009", "customer_name": "Iris Martinez",
            "customer_email": "iris@test.com", "order_date": days_ago(0),
            "items": [{"name": "Yoga Mat", "qty": 1, "price": 39.99}],
            "total_amount": 39.99, "payment_status": "paid",
            "shipping_status": "processing", "delivery_date": None,
            "carrier": None, "tracking_number": None,
        },
    },
    {
        "id": "TC-010",
        "description": "Cancel already-shipped order — DENY, redirect to return",
        "expected_decision": "deny",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I want to cancel my order but it seems like it already shipped. Can you cancel it?",
        "order": {
            "order_id": "ORD-010", "customer_name": "Jack Thompson",
            "customer_email": "jack@test.com", "order_date": days_ago(3),
            "items": [{"name": "Gaming Mouse", "qty": 1, "price": 49.99}],
            "total_amount": 49.99, "payment_status": "paid",
            "shipping_status": "in_transit", "delivery_date": None,
            "carrier": "FedEx", "tracking_number": "TRK010",
        },
    },
    {
        "id": "TC-011",
        "description": "Wrong item shipped — our error, APPROVE full resolution",
        "expected_decision": "approve",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I ordered a blue shirt size M but received a red shirt size XL. This is clearly the wrong item.",
        "order": {
            "order_id": "ORD-011", "customer_name": "Karen Adams",
            "customer_email": "karen@test.com", "order_date": days_ago(7),
            "items": [{"name": "Classic T-Shirt (Blue, Size M)", "qty": 1, "price": 24.99}],
            "total_amount": 24.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(2),
            "carrier": "USPS", "tracking_number": "TRK011",
        },
    },
    {
        "id": "TC-012",
        "description": "Vague refund request, missing delivery date — NEED_MORE_INFO",
        "expected_decision": "need_more_info",
        "expected_has_citations": False,
        "expected_clarifying_questions": True,
        "ticket": "I want to return my order. Please process the refund.",
        "order": {
            "order_id": "ORD-012", "customer_name": "Liam Garcia",
            "customer_email": "liam@test.com", "order_date": days_ago(20),
            "items": [{"name": "Unknown Item", "qty": 1, "price": 45.00}],
            "total_amount": 45.00, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": None,
            "carrier": "FedEx", "tracking_number": "TRK012",
        },
    },
    {
        "id": "TC-013",
        "description": "Billing dispute, overcharged — ESCALATE for investigation",
        "expected_decision": "escalate",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I was charged $149 but my order total was $99. I've been overcharged by $50.",
        "order": {
            "order_id": "ORD-013", "customer_name": "Mia Robinson",
            "customer_email": "mia@test.com", "order_date": days_ago(10),
            "items": [{"name": "Wireless Speaker", "qty": 1, "price": 99.00}],
            "total_amount": 149.00, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(5),
            "carrier": "UPS", "tracking_number": "TRK013",
        },
    },
    {
        "id": "TC-014",
        "description": "High-value order ($750) damaged — ESCALATE per compensation policy",
        "expected_decision": "escalate",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "My laptop arrived with a cracked screen. I want a full refund of $750.",
        "order": {
            "order_id": "ORD-014", "customer_name": "Noah Clark",
            "customer_email": "noah@test.com", "order_date": days_ago(4),
            "items": [{"name": "Business Laptop 15 inch", "qty": 1, "price": 750.00}],
            "total_amount": 750.00, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(2),
            "carrier": "FedEx", "tracking_number": "TRK014",
        },
    },
    {
        "id": "TC-015",
        "description": "Spam message — handle gracefully",
        "expected_decision": "escalate",
        "expected_has_citations": False,
        "expected_clarifying_questions": False,
        "ticket": "FREE BITCOIN OFFER!! Click here to claim your prize $$$",
        "order": {
            "order_id": "ORD-015", "customer_name": "Olivia Lewis",
            "customer_email": "spam@test.com", "order_date": days_ago(1),
            "items": [{"name": "Test Item", "qty": 1, "price": 0.00}],
            "total_amount": 0.00, "payment_status": "pending",
            "shipping_status": "processing", "delivery_date": None,
            "carrier": None, "tracking_number": None,
        },
    },
    {
        "id": "TC-016",
        "description": "Partial order issue — one item damaged, rest fine (PARTIAL)",
        "expected_decision": "partial",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I ordered 3 items. Two arrived fine but the vase arrived broken. I only want a refund for the broken item.",
        "order": {
            "order_id": "ORD-016", "customer_name": "Peter Hall",
            "customer_email": "peter@test.com", "order_date": days_ago(6),
            "items": [
                {"name": "Decorative Vase", "qty": 1, "price": 35.00},
                {"name": "Candle Set", "qty": 1, "price": 22.00},
                {"name": "Photo Frame", "qty": 1, "price": 18.00},
            ],
            "total_amount": 75.00, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(3),
            "carrier": "USPS", "tracking_number": "TRK016",
        },
    },
    {
        "id": "TC-017",
        "description": "Wrong address entered by customer — PARTIAL (50% replacement cost)",
        "expected_decision": "partial",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I think I entered the wrong apartment number when I ordered. The package was delivered to the wrong unit. Can you reship it?",
        "order": {
            "order_id": "ORD-017", "customer_name": "Quinn Young",
            "customer_email": "quinn@test.com", "order_date": days_ago(9),
            "items": [{"name": "Desk Organizer", "qty": 1, "price": 29.99}],
            "total_amount": 29.99, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(4),
            "carrier": "FedEx", "tracking_number": "TRK017",
        },
    },
    {
        "id": "TC-018",
        "description": "General policy inquiry — informational / NEED_MORE_INFO",
        "expected_decision": "need_more_info",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "How long do I have to return a product? And what is your refund timeline?",
        "order": {
            "order_id": "ORD-018", "customer_name": "Rachel Scott",
            "customer_email": "rachel@test.com", "order_date": days_ago(1),
            "items": [{"name": "Book", "qty": 1, "price": 15.99}],
            "total_amount": 15.99, "payment_status": "paid",
            "shipping_status": "in_transit", "delivery_date": None,
            "carrier": "USPS", "tracking_number": "TRK018",
        },
    },
    {
        "id": "TC-019",
        "description": "Price adjustment within 7 days — APPROVE store credit",
        "expected_decision": "approve",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "I bought a jacket 5 days ago for $120, but I see it's now on sale for $90. Can I get the $30 difference back?",
        "order": {
            "order_id": "ORD-019", "customer_name": "Sam Turner",
            "customer_email": "sam@test.com", "order_date": days_ago(5),
            "items": [{"name": "Wool Jacket", "qty": 1, "price": 120.00}],
            "total_amount": 120.00, "payment_status": "paid",
            "shipping_status": "delivered", "delivery_date": days_ago(2),
            "carrier": "UPS", "tracking_number": "TRK019",
        },
    },
    {
        "id": "TC-020",
        "description": "Free shipping threshold inquiry — $55 order",
        "expected_decision": "need_more_info",
        "expected_has_citations": True,
        "expected_clarifying_questions": False,
        "ticket": "My order is $55. Do I qualify for free shipping? I was charged for shipping and I think I shouldn't have been.",
        "order": {
            "order_id": "ORD-020", "customer_name": "Tina Moore",
            "customer_email": "tina@test.com", "order_date": days_ago(1),
            "items": [{"name": "Skincare Bundle", "qty": 1, "price": 55.00}],
            "total_amount": 55.00, "payment_status": "paid",
            "shipping_status": "processing", "delivery_date": None,
            "carrier": None, "tracking_number": None,
        },
    },
]

# Decisions considered "close enough" to count as correct
_ADJACENT = {
    ("approve",        "partial"):       True,
    ("partial",        "approve"):       True,
    ("escalate",       "need_more_info"): True,
    ("need_more_info", "escalate"):       True,
}


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_evaluation():
    results = []
    decision_correct = 0
    citation_present = 0
    clarifying_when_needed = 0
    cases_needing_clarification = 0
    compliance_passes = 0
    times = []
    errors = []

    print("\n" + "=" * 70)
    print("RUNNING EVALUATION — 20 TEST CASES")
    if INTER_CASE_DELAY > 0:
        print(f"Inter-case delay: {INTER_CASE_DELAY}s  (use --fast to disable)")
    print("=" * 70)

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i:02d}/{len(TEST_CASES)}] {tc['id']}: {tc['description']}")

        # ── Rate-limit protection: pause between cases ────────────────────
        if i > 1 and INTER_CASE_DELAY > 0:
            time.sleep(INTER_CASE_DELAY)

        start = time.time()
        try:
            result = run_pipeline(
                ticket_dict={"message": tc["ticket"]},
                order_dict=tc["order"],
                ticket_id=tc["id"],
            )
            elapsed = round(time.time() - start, 2)
            times.append(elapsed)

            actual   = result.get("decision", "")
            expected = tc["expected_decision"]
            decision_match = (
                actual == expected
                or _ADJACENT.get((actual, expected), False)
            )
            has_citations  = len(result.get("citations", [])) > 0
            has_clarifying = len(result.get("clarifying_questions", [])) > 0
            compliance_ok  = result.get("compliance_passed", False)

            if decision_match:  decision_correct += 1
            if has_citations:   citation_present += 1
            if compliance_ok:   compliance_passes += 1
            if tc["expected_clarifying_questions"]:
                cases_needing_clarification += 1
                if has_clarifying:
                    clarifying_when_needed += 1

            status = "PASS" if decision_match else "FAIL"
            print(f"  {status} | expected={expected} | got={actual} | "
                  f"citations={has_citations} | compliance={compliance_ok} | {elapsed}s")

            results.append({
                "test_id":          tc["id"],
                "description":      tc["description"],
                "expected_decision": expected,
                "actual_decision":   actual,
                "decision_correct":  decision_match,
                "has_citations":     has_citations,
                "has_clarifying_questions": has_clarifying,
                "compliance_passed": compliance_ok,
                "confidence_score":  result.get("confidence_score", 0),
                "processing_time":   elapsed,
                "retrieved_policies": result.get("retrieved_policies", []),
                # Include first citation for quality audit
                "sample_citation":   result["citations"][0] if result.get("citations") else None,
            })

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            times.append(elapsed)
            errors.append({"test_id": tc["id"], "error": str(e)})
            print(f"  ERROR: {e}")
            results.append({
                "test_id": tc["id"],
                "description": tc["description"],
                "error": str(e),
                "decision_correct": False,
                "processing_time": elapsed,
            })

    # ── Summary ───────────────────────────────────────────────────────────
    n = len(TEST_CASES)
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Decision accuracy:      {decision_correct}/{n} = {decision_correct/n:.0%}")
    print(f"Citation presence rate: {citation_present}/{n} = {citation_present/n:.0%}")
    print(f"Compliance pass rate:   {compliance_passes}/{n} = {compliance_passes/n:.0%}")
    if cases_needing_clarification:
        print(f"Clarifying Q rate:      {clarifying_when_needed}/{cases_needing_clarification}"
              f" = {clarifying_when_needed/cases_needing_clarification:.0%}")
    print(f"Errors:                 {len(errors)}/{n}")
    if times:
        print(f"Avg time per case:      {statistics.mean(times):.2f}s")
        print(f"Median time per case:   {statistics.median(times):.2f}s")
        print(f"Total wall time:        {sum(times):.1f}s")

    summary = {
        "timestamp":               datetime.now().isoformat(),
        "total_cases":             n,
        "decision_accuracy":       decision_correct / n,
        "citation_presence_rate":  citation_present / n,
        "compliance_pass_rate":    compliance_passes / n,
        "error_count":             len(errors),
        "avg_processing_time_sec": statistics.mean(times) if times else 0,
        "inter_case_delay_sec":    INTER_CASE_DELAY,
        "results":                 results,
        "errors":                  errors,
    }

    out = Path("evaluation_results.json")
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results saved to: {out.resolve()}")
    return summary


if __name__ == "__main__":
    run_evaluation()
