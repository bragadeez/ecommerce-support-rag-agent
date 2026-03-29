import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

POLICY_DOCS_PATH = Path(os.getenv("POLICY_DOCS_PATH", "./data/policies"))
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", "./data/vectorstore"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 400       # characters (approx 80-100 tokens for English text)
CHUNK_OVERLAP = 80


def load_policy_documents() -> List[Tuple[str, str]]:
    """
    Load all .txt policy files from the policy directory.
    Returns list of (filename, content) tuples.
    """
    POLICY_DOCS_PATH.mkdir(parents=True, exist_ok=True)
    docs = []
    for filepath in POLICY_DOCS_PATH.glob("*.txt"):
        content = filepath.read_text(encoding="utf-8")
        docs.append((filepath.name, content))
        print(f"  Loaded: {filepath.name} ({len(content)} chars)")
    return docs


def chunk_document(filename: str, content: str) -> List[dict]:
    """
    Split a document into overlapping chunks.
    Each chunk carries its source metadata for citation.

    Strategy: character-based chunking on paragraph boundaries where possible.
    This keeps policy rules intact more often than blind character splits.
    """
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, save current and start new
        if len(current_chunk) + len(para) > CHUNK_SIZE and current_chunk:
            chunk_id = f"{filename}::chunk_{chunk_index}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": f"{filename} (section {chunk_index + 1})",
                "content": current_chunk.strip(),
            })
            # Overlap: keep last CHUNK_OVERLAP chars of current chunk
            current_chunk = current_chunk[-CHUNK_OVERLAP:] + "\n\n" + para
            chunk_index += 1
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunk_id = f"{filename}::chunk_{chunk_index}"
        chunks.append({
            "chunk_id": chunk_id,
            "source": f"{filename} (section {chunk_index + 1})",
            "content": current_chunk.strip(),
        })

    return chunks


def build_vectorstore():
    """
    Full pipeline: load docs → chunk → embed → build FAISS index → save.
    """
    print("\n[1/4] Loading policy documents...")
    docs = load_policy_documents()
    if not docs:
        print("  WARNING: No .txt files found in", POLICY_DOCS_PATH)
        print("  Run: python data_pipeline.py --create-sample-policies first")
        return

    print(f"\n[2/4] Chunking {len(docs)} documents...")
    all_chunks = []
    for filename, content in docs:
        chunks = chunk_document(filename, content)
        all_chunks.extend(chunks)
        print(f"  {filename} → {len(chunks)} chunks")
    print(f"  Total chunks: {len(all_chunks)}")

    print(f"\n[3/4] Embedding with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity (FAISS inner product = cosine when normalized)
    faiss.normalize_L2(embeddings)
    print(f"  Embedding shape: {embeddings.shape}")

    print("\n[4/4] Building FAISS index and saving...")
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)   # Inner Product = cosine after L2 norm
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(VECTOR_DB_PATH / "policy_index.faiss"))

    # Save chunk metadata (needed to return text + source at query time)
    with open(VECTOR_DB_PATH / "chunks_metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    # Save stats
    stats = {
        "total_documents": len(docs),
        "total_chunks": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL,
        "dimension": dimension,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open(VECTOR_DB_PATH / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved index to {VECTOR_DB_PATH}/policy_index.faiss")
    print(f"  Saved metadata to {VECTOR_DB_PATH}/chunks_metadata.pkl")
    print(f"\nVector store built successfully: {len(all_chunks)} chunks indexed.")


def create_sample_policies():
    """
    Create realistic sample policy documents so the system works out of the box.
    In production, replace these with your actual policy PDFs/docs.
    """
    POLICY_DOCS_PATH.mkdir(parents=True, exist_ok=True)

    policies = {
        "refund_policy.txt": """REFUND AND RETURN POLICY

Eligibility Window
Customers may request a full refund within 30 days of the delivery date. Requests made between 31-60 days from delivery are eligible for store credit only. Requests made after 60 days from delivery are not eligible for any refund or credit.

Condition Requirements
Items must be returned in their original, unused condition with all original packaging and tags. Items showing signs of use, damage caused by the customer, or missing components will be denied a full refund. A partial refund of up to 50% may be granted at the discretion of the support team for lightly used items.

Non-Returnable Items
The following items are non-returnable and non-refundable under any circumstances: digital downloads, gift cards, perishable goods, customized or personalized items, and items marked as "Final Sale" at the time of purchase.

Refund Processing Time
Once the returned item is received and inspected, refunds are processed within 5-7 business days to the original payment method. Store credit is applied within 1-2 business days.

Damaged or Defective Items
If an item arrives damaged or defective, customers must report the issue within 7 days of delivery with photographic evidence. Approved claims will receive a full refund or replacement at no cost, including return shipping. Claims submitted after 7 days for damage present at delivery will be evaluated on a case-by-case basis.

Shipping Costs
Original shipping charges are non-refundable except in cases where the return is due to our error (wrong item shipped, item arrived damaged). Customers are responsible for return shipping costs unless the return is due to our error.
""",

        "shipping_policy.txt": """SHIPPING AND DELIVERY POLICY

Standard Shipping Timeframes
Standard shipping orders are processed within 1-2 business days and delivered within 5-7 business days within the continental United States. Expedited shipping delivers within 2-3 business days. Overnight shipping delivers the next business day for orders placed before 2 PM EST.

International Shipping
International orders ship within 2-3 business days and arrive within 10-21 business days depending on destination country and customs processing. International customers are responsible for all customs duties and import taxes.

Lost Packages
A package is considered lost if it has not been delivered within 15 business days of the expected delivery date for domestic orders, or 30 business days for international orders. Customers must file a lost package claim within 60 days of the order date. Approved lost package claims result in a full refund or free replacement shipment.

Stolen Packages
For packages marked as delivered but not received, customers must file a claim within 7 days of the marked delivery date. We will initiate a carrier investigation. If the investigation confirms non-delivery, we will issue a replacement or refund. We are not responsible for packages stolen after confirmed delivery.

Address Errors
If a customer provides an incorrect shipping address, we will attempt to intercept the package. If interception is not possible and the package is delivered to the wrong address, a replacement will be sent at 50% of the original order cost. Full replacement will be issued if the address error was caused by a system error on our part.

Free Shipping Threshold
Orders over $50 qualify for free standard shipping within the continental United States. This threshold applies to the subtotal after discounts and before taxes.
""",

        "cancellation_policy.txt": """ORDER CANCELLATION POLICY

Cancellation Window
Orders may be cancelled for a full refund within 1 hour of placement, provided the order has not yet entered the fulfillment process. After 1 hour, cancellations may not be possible if the order is already being packed or shipped.

Pre-Shipment Cancellations
If an order has not yet shipped, customers may request a cancellation by contacting support. We will make every effort to cancel the order. If successful, a full refund is issued within 3-5 business days. If the order cannot be stopped, the customer must follow the standard return process once delivered.

Post-Shipment Cancellations
Once an order has shipped, it cannot be cancelled. Customers must wait for delivery and then initiate a return request under the Refund and Return Policy.

Subscription Cancellations
Subscription orders may be cancelled at any time. Cancellations take effect at the end of the current billing period. No partial refunds are issued for unused subscription time. Customers keep access to all subscription benefits until the period ends.

Bulk and Custom Orders
Bulk orders (10 or more units) and custom or personalized orders cannot be cancelled once confirmed and payment has been processed, as production begins immediately.
""",

        "compensation_policy.txt": """CUSTOMER COMPENSATION AND GOODWILL POLICY

Compensation for Our Errors
When an error is attributable to ShopEase (wrong item shipped, significant delay caused by our fulfillment center, damaged item, system error), customers are eligible for the following compensation tiers:

Tier 1 - Minor Inconvenience (delay of 1-3 days, minor packaging damage with no product damage): $5 store credit or 10% discount on next order.

Tier 2 - Moderate Issue (wrong item shipped, delay of 4-7 days, item damaged but still functional): Full refund of affected items plus $10 store credit.

Tier 3 - Major Issue (order completely lost, item arrived non-functional, delay over 7 days): Full refund plus free replacement shipping plus $20 store credit.

Goodwill Gestures
Support agents may offer goodwill gestures up to $15 in store credit for exceptional circumstances not covered by standard policy, subject to manager approval for amounts over $15. Goodwill store credit is non-transferable and expires after 90 days.

Escalation
Cases requiring compensation above Tier 3 thresholds, legal threats, or involving purchases over $500 must be escalated to the senior support team. Do not issue refunds exceeding the original purchase amount without senior approval.

Repeat Claims
Customers with more than 3 compensation claims in a 12-month period will be flagged for review. Subsequent claims require manager approval.
""",

        "payment_policy.txt": """PAYMENT AND BILLING POLICY

Accepted Payment Methods
We accept Visa, Mastercard, American Express, Discover, PayPal, Apple Pay, Google Pay, and ShopEase store credit. We do not accept cryptocurrency, cash, checks, or money orders.

Payment Security
All transactions are processed through PCI-DSS compliant payment processors. We do not store full credit card numbers. Customers will never be asked to provide payment information through email or chat.

Billing Disputes
Customers who believe they have been incorrectly charged must notify us within 60 days of the charge date. We will investigate and respond within 5 business days. If the charge is confirmed incorrect, a full refund will be issued within 7-10 business days.

Chargebacks
Filing a chargeback without first contacting our support team may result in account suspension. We encourage customers to resolve disputes directly with us first. If a chargeback is filed and we determine the original charge was valid, we reserve the right to provide evidence to the payment processor.

Failed Payments
If a payment fails, the order will not be processed. Customers will be notified immediately to update their payment method. Orders are held for 24 hours; after that, items are returned to inventory.

Price Adjustments
If an item goes on sale within 7 days of purchase, customers may request a one-time price adjustment for the difference as store credit. Price adjustments are not available during major sale events (Black Friday, Cyber Monday, Holiday Sale).
"""
    }

    for filename, content in policies.items():
        filepath = POLICY_DOCS_PATH / filename
        filepath.write_text(content, encoding="utf-8")
        print(f"  Created: {filepath}")

    print(f"\nCreated {len(policies)} sample policy documents in {POLICY_DOCS_PATH}")


if __name__ == "__main__":
    import sys
    if "--create-sample-policies" in sys.argv:
        print("Creating sample policy documents...")
        create_sample_policies()

    print("\nBuilding vector store...")
    build_vectorstore()
