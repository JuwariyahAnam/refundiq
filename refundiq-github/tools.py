"""The agent's tools. Each returns a JSON-serialisable dict.
Errors are returned as {"error": ...} (never raised) so the agent can observe and re-plan."""
import json
import sqlite3
import urllib.error
import urllib.request
from datetime import date, timedelta

import courier_api
from config import (COURIER_HOST, COURIER_PORT, COURIER_TIMEOUT_SECS, DB_PATH,
                    REPEAT_RETURN_WINDOW_DAYS, TODAY)
from rag import search_policy as _rag_search


def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


# ---------- Tool 1: order database ----------
def get_order(order_id: str) -> dict:
    con = _db()
    row = con.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.strip(),)).fetchone()
    con.close()
    if not row:
        return {"error": f"Order {order_id} not found in order database"}
    d = dict(row)
    d["price_paid_inr"] = d.pop("price")
    return d


# ---------- Tool 2: courier tracking API (HTTP, with retry) ----------
def track_shipment(tracking_id: str) -> dict:
    courier_api.start_in_background()
    url = f"http://{COURIER_HOST}:{COURIER_PORT}/track/{tracking_id.strip()}"
    last_err = None
    for attempt in (1, 2):  # retry once
        try:
            with urllib.request.urlopen(url, timeout=COURIER_TIMEOUT_SECS) as r:
                data = json.loads(r.read())
                if data.get("delivered_on"):
                    days = (TODAY - date.fromisoformat(data["delivered_on"])).days
                    data["days_since_delivery"] = days
                return data
        except urllib.error.HTTPError as e:
            return {"error": f"Courier API returned HTTP {e.code} for {tracking_id}"}
        except Exception as e:  # timeout / connection refused
            last_err = f"{type(e).__name__}: {e}"
    return {"error": f"Courier API unavailable after 2 attempts ({last_err})"}


# ---------- Tool 3: customer memory ----------
def get_customer_history(customer_id: str) -> dict:
    con = _db()
    cust = con.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    rows = con.execute(
        "SELECT ticket_id, order_id, created_date, message, decision, refund_amount, status "
        "FROM tickets WHERE customer_id = ? ORDER BY created_date DESC", (customer_id,)).fetchall()
    con.close()
    if not cust:
        return {"error": f"Customer {customer_id} not found"}
    cutoff = (TODAY - timedelta(days=REPEAT_RETURN_WINDOW_DAYS)).isoformat()
    recent = [r for r in rows if r["created_date"] >= cutoff]
    return {
        "customer_id": customer_id,
        "name": cust["name"],
        "total_past_tickets": len(rows),
        f"refund_requests_last_{REPEAT_RETURN_WINDOW_DAYS}_days": len(recent),
        "past_tickets": [dict(r) for r in rows[:10]],
    }


# ---------- Tool 4: policy RAG ----------
def search_policy(query: str) -> dict:
    hits = _rag_search(query, k=3)
    return {"query": query, "clauses": hits}


# ---------- Tool 5: ask the customer for proof ----------
def request_proof(ticket_id: str, proof_type: str) -> dict:
    con = _db()
    con.execute("INSERT INTO proof_requests (ticket_id, proof_type, created_date) VALUES (?,?,?)",
                (ticket_id, proof_type, TODAY.isoformat()))
    con.commit()
    con.close()
    return {"status": "proof_requested", "ticket_id": ticket_id, "proof_type": proof_type,
            "message_sent_to_customer": f"Please upload a {proof_type} so we can process your request."}


# ---------- Tool 6: refund ledger (duplicate check) ----------
def check_refund_status(order_id: str) -> dict:
    con = _db()
    row = con.execute("SELECT * FROM refunds WHERE order_id = ?", (order_id.strip(),)).fetchone()
    con.close()
    if row:
        return {"order_id": order_id, "already_refunded": True, "existing_refund": dict(row)}
    return {"order_id": order_id, "already_refunded": False}


TOOL_FUNCTIONS = {
    "check_refund_status": check_refund_status,
    "get_order": get_order,
    "track_shipment": track_shipment,
    "get_customer_history": get_customer_history,
    "search_policy": search_policy,
    "request_proof": request_proof,
}

# Gemini function declarations (schema the LLM sees)
S, O, N = "STRING", "OBJECT", "NUMBER"
TOOL_DECLARATIONS = [
    {"name": "get_order",
     "description": "Look up an order in the order database: item, category, price paid, serial number, tracking id.",
     "parameters": {"type": O, "properties": {"order_id": {"type": S}}, "required": ["order_id"]}},
    {"name": "track_shipment",
     "description": "Call the courier API to get delivery status, delivery date and days since delivery.",
     "parameters": {"type": O, "properties": {"tracking_id": {"type": S}}, "required": ["tracking_id"]}},
    {"name": "get_customer_history",
     "description": "Get the customer's past support tickets and how many refund requests they made in the last 60 days.",
     "parameters": {"type": O, "properties": {"customer_id": {"type": S}}, "required": ["customer_id"]}},
    {"name": "search_policy",
     "description": "Search the written return policy. Returns the most relevant clauses with clause ids. Call it once per issue you need to check.",
     "parameters": {"type": O, "properties": {"query": {"type": S}}, "required": ["query"]}},
    {"name": "check_refund_status",
     "description": "Check the refund ledger to see if this order was already refunded (prevents duplicate refunds).",
     "parameters": {"type": O, "properties": {"order_id": {"type": S}}, "required": ["order_id"]}},
    {"name": "request_proof",
     "description": "Ask the customer to upload proof (e.g. 'photo of the damage', 'photo of the item received', 'photo of serial number').",
     "parameters": {"type": O, "properties": {"ticket_id": {"type": S}, "proof_type": {"type": S}},
                    "required": ["ticket_id", "proof_type"]}},
    {"name": "submit_decision",
     "description": "Submit the final decision. Call exactly once, after gathering evidence.",
     "parameters": {"type": O, "properties": {
         "decision": {"type": S, "enum": ["APPROVE", "PARTIAL_APPROVE", "NEED_PROOF", "REJECT", "HOLD_FOR_HUMAN"]},
         "refund_amount": {"type": N, "description": "Refund in INR. 0 for NEED_PROOF or REJECT."},
         "clause_ids": {"type": S, "description": "Comma-separated policy clause ids relied on, e.g. 'P1,P9'"},
         "clause_quote": {"type": S, "description": "Short quote of the key policy sentence"},
         "reasoning": {"type": S, "description": "2-3 sentences explaining the decision"},
         "customer_reply": {"type": S, "description": "Polite message to send the customer"}},
         "required": ["decision", "refund_amount", "clause_ids", "clause_quote", "reasoning", "customer_reply"]}},
]
