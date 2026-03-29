# E-Commerce Support Resolution Agent

A production-quality multi-agent RAG system that automatically resolves customer support tickets using policy-grounded decisions.

## Architecture

```
Customer Ticket + Order JSON
        │
        ▼
┌─────────────────┐
│  Triage Agent   │  ← Classifies issue, extracts facts, generates retrieval queries
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Policy Retriever    │  ← Semantic search on FAISS vector DB (all-MiniLM-L6-v2)
└────────┬────────────┘
         │
         ▼
┌──────────────────────┐
│ Resolution Writer    │  ← Drafts decision with citations (Gemini 1.5 Flash)
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│ Compliance Agent │  ← Checks for hallucinations, missing citations
└────────┬─────────┘
    ┌────┴────┐
  FAIL      PASS (up to 2 retries)
    │          │
    └────┬─────┘
         ▼
   Final Resolution JSON
```

## Tech Stack (all free)

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 1.5 Flash (free tier) |
| Embeddings | `all-MiniLM-L6-v2` (local, HuggingFace) |
| Vector DB | FAISS (local) |
| Orchestration | LangGraph |
| Framework | LangChain |
| API | FastAPI |
| UI | Streamlit |

## Setup (3 steps)

### 1. Get a free Gemini API key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key" → Create API key
3. Copy the key

### 2. Install & configure

```bash
git clone <your-repo>
cd ecommerce-support-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your GOOGLE_API_KEY
```

### 3. Build the vector store

```bash
# Creates sample policy documents + builds FAISS index
python data_pipeline.py --create-sample-policies
```

You should see:
```
Created 5 sample policy documents
Building vector store...
Total chunks: ~42
Embedding shape: (42, 384)
Vector store built successfully.
```

## Running the System

### Option A: Streamlit UI (recommended for demo)
```bash
streamlit run ui.py
```
Open http://localhost:8501 — load a sample ticket from the sidebar and click "Run Resolution Pipeline".

### Option B: FastAPI REST
```bash
uvicorn api:app --reload --port 8000
```
Open http://localhost:8000/docs for Swagger UI.

Example request:
```bash
curl -X POST http://localhost:8000/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICKET-001",
    "message": "I received my order 10 days ago and want a full refund.",
    "order": {
      "order_id": "ORD-001",
      "customer_name": "Jane Doe",
      "customer_email": "jane@example.com",
      "order_date": "2024-11-15",
      "items": [{"name": "Laptop Stand", "qty": 1, "price": 49.99}],
      "total_amount": 49.99,
      "payment_status": "paid",
      "shipping_status": "delivered",
      "delivery_date": "2024-11-20",
      "carrier": "FedEx",
      "tracking_number": "FX123456"
    }
  }'
```

### Option C: Python directly
```python
from graph import run_pipeline

result = run_pipeline(
    ticket_dict={"message": "I want to return my order."},
    order_dict={...},
    ticket_id="TICKET-001"
)
print(result)
```

## Running Evaluation

```bash
python evaluate.py
```

Runs 20 test cases and prints:
- Decision accuracy
- Citation presence rate
- Compliance pass rate
- Processing time stats

Results saved to `evaluation_results.json`.

## Output Schema

```json
{
  "ticket_id": "TICKET-001",
  "classification": "refund_request",
  "severity": "medium",
  "decision": "approve",
  "rationale": "Customer is within the 30-day refund window...",
  "citations": [
    {
      "claim": "Refund eligible within 30 days of delivery",
      "source": "refund_policy.txt (section 1)",
      "chunk_id": "refund_policy.txt::chunk_0"
    }
  ],
  "customer_response": "Dear Jane, we are happy to process your refund...",
  "internal_notes": "Standard 30-day refund. Process immediately.",
  "clarifying_questions": [],
  "compliance_passed": true,
  "confidence_score": 0.92,
  "retrieved_policies": ["refund_policy.txt (section 1)"]
}
```

## Project Structure

```
ecommerce-support-agent/
├── schemas.py           # Pydantic models for all agent I/O
├── data_pipeline.py     # Build FAISS vector store from policy docs
├── retriever.py         # Semantic search wrapper
├── agents.py            # All 4 agent implementations
├── graph.py             # LangGraph state machine
├── api.py               # FastAPI REST endpoint
├── ui.py                # Streamlit frontend
├── evaluate.py          # 20 test cases + metrics
├── requirements.txt
├── .env.example
└── data/
    ├── policies/        # Your .txt policy documents
    └── vectorstore/     # FAISS index (auto-generated)
```

## Hallucination Prevention

Three layers of guardrails:

1. **Prompt-level**: Resolution Writer is explicitly instructed to write `INSUFFICIENT_CONTEXT` rather than guess. Every claim must have a citation from the retrieved context only.

2. **Retrieval threshold**: Chunks with cosine similarity below 0.35 are filtered out. No low-confidence policy gets into the prompt.

3. **Compliance Agent**: Reviews every claim in the draft against the retrieved passages. Triggers a rewrite (up to 2 retries) if it finds unsupported claims.

## Adding Your Own Policies

1. Add `.txt` files to `data/policies/`
2. Re-run `python data_pipeline.py` to rebuild the index
3. The system automatically picks them up
