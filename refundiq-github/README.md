# RefundIQ — Refund & Returns Desk Agent (Problem 16)

An autonomous agent that reads a customer refund ticket, checks the order and delivery through tools,
checks the written return policy with RAG, remembers the customer's history, and decides:
**APPROVE · PARTIAL_APPROVE · NEED_PROOF · REJECT · HOLD_FOR_HUMAN**, always quoting the policy clause.

## Run it (5 minutes)

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your Gemini key into .env
python seed.py                # builds the mock database
python eval.py T01            # quick 1-ticket test
python eval.py                # full 18-ticket evaluation -> eval_results.json
python server.py              # main demo UI -> http://localhost:8000
pip install -r requirements-streamlit.txt && streamlit run app.py   # backup UI
```

If you get a "model not found" error, set `GEMINI_MODEL=` in `.env` to a model listed in Google AI Studio.

## How it works

```
Ticket ─► LLM agent loop (Groq or Gemini) ──(plan → tool call → observe → re-plan, up to 10 steps)──► proposed decision
              │  get_order          (SQLite order DB)
              │  track_shipment     (HTTP courier API, with retry + timeout)
              │  get_customer_history (memory of past tickets)
              │  search_policy      (RAG over policy.md)
              │  request_proof      (asks customer for photo / serial)
              ▼
     Self-check (sends conflicting decisions back to re-plan) ─► Code guardrails (final say) ─► AUTO_REFUNDED / PENDING_HUMAN / AWAITING_PROOF / REJECTED
              ▼
     Saved to customer memory ─► Human approval queue in the UI
```

## Files

| File | What it does |
|---|---|
| `data/policy.md` | Written return policy, 12 numbered clauses (P1–P12) |
| `seed.py` | Mock DB: customers, orders, shipments, past tickets, refund ledger |
| `server.py` + `static/index.html` | Main web frontend (live trace, approvals, memory, results) |
| `courier_api.py` | Mock courier HTTP service; one shipment simulates an outage |
| `rag.py` | TF-IDF retrieval over policy clauses |
| `tools.py` | The 6 tools + their schemas for Gemini |
| `agent.py` | The agent loop, memory write-back, human review |
| `guardrails.py` | Hard rules in code: limit, abuse, serial, duplicates, injection, failures, refund cap |
| `data/eval_set.json` | 18 labelled tickets covering every path |
| `eval.py` | Runs the eval, prints accuracy + safety metrics |
| `app.py` | Streamlit demo: live trace, decision, approval queue, memory, eval dashboard |

## Evaluation set (18 tickets)

| Category | Tickets | Expected |
|---|---|---|
| Auto-approve | T01–T03 | APPROVE |
| High-value hold | T04–T05 | HOLD_FOR_HUMAN |
| Partial refund | T06–T07 | PARTIAL_APPROVE |
| Needs proof | T08–T09 | NEED_PROOF |
| Policy reject | T10, T11, T17 | REJECT |
| Abuse | T12 (repeat returner), T13 (serial mismatch) | HOLD_FOR_HUMAN |
| Failure handling | T14 (order missing), T15 (courier API down) | HOLD_FOR_HUMAN |
| Prompt injection | T16 | REJECT |
| Duplicate refund | T18 | REJECT |

Metrics reported: decision accuracy, per-category accuracy, abuse caught, unsafe auto-refunds
(should be 0), guardrail overrides, average tool calls per ticket.

## Deploy on Render
1. Push this folder to a GitHub repo (`.env` is ignored, so your keys stay private).
2. Render → New → Blueprint → pick the repo (it reads `render.yaml`).
3. Set `GROQ_API_KEY` when asked → Deploy. Open the `.onrender.com` link.
