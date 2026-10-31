"""Web server for the custom frontend.
Run:  python server.py   then open http://localhost:8000"""
import json
import re
import queue
import sqlite3
import threading
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from agent import human_review, run_ticket
from config import BASE_DIR, DB_PATH
from seed import reset_db

app = FastAPI(title="Refund Desk Agent")
if not DB_PATH.exists():
    reset_db()

INBOX_SCHEMA = """CREATE TABLE IF NOT EXISTS inbox (ticket_id TEXT PRIMARY KEY, customer_id TEXT,
  order_id TEXT, message TEXT, photo_attached INTEGER, serial_number TEXT, received_at TEXT,
  state TEXT, decision TEXT, refund_amount REAL, status TEXT, finished_at TEXT)"""


def ensure_inbox():
    con = sqlite3.connect(DB_PATH)
    con.execute(INBOX_SCHEMA)
    con.execute("UPDATE inbox SET state='QUEUED' WHERE state='PROCESSING'")  # recover after a crash
    con.commit()
    con.close()


ensure_inbox()

# ---------------- AUTOPILOT: the agent works through the inbox on its own ----------------
AUTOPILOT = {"on": True, "current": None, "steps": [], "last_error": None}


def _next_ticket():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM inbox WHERE state='QUEUED' ORDER BY received_at, ticket_id LIMIT 1").fetchone()
    if row:
        con.execute("UPDATE inbox SET state='PROCESSING' WHERE ticket_id=?", (row["ticket_id"],))
        con.commit()
    con.close()
    return dict(row) if row else None


def _finish(ticket_id, final, state="DONE"):
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE inbox SET state=?, decision=?, refund_amount=?, status=?, finished_at=? WHERE ticket_id=?",
                (state, final and final["final_decision"], final and final["refund_amount"],
                 final and final["status"], datetime.now().strftime("%H:%M:%S"), ticket_id))
    con.commit()
    con.close()


def autopilot_loop():
    while True:
        try:
            t = _next_ticket() if AUTOPILOT["on"] else None
            if not t:
                time.sleep(2)
                continue
            ticket = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"], "order_id": t["order_id"],
                      "message": t["message"], "photo_attached": bool(t["photo_attached"]),
                      "serial_number": t["serial_number"]}
            AUTOPILOT.update(current=ticket, steps=[])
            print(f"[autopilot] processing {t['ticket_id']}", flush=True)
            res = run_ticket(ticket, on_step=lambda s: AUTOPILOT["steps"].append(s))
            _finish(t["ticket_id"], res["final"])
            print(f"[autopilot] {t['ticket_id']} -> {res['final']['final_decision']}", flush=True)
        except Exception as e:
            AUTOPILOT["last_error"] = f"{type(e).__name__}: {e}"
            if AUTOPILOT.get("current"):
                _finish(AUTOPILOT["current"]["ticket_id"], None, state="FAILED")
        finally:
            AUTOPILOT["current"] = None
        time.sleep(1)


threading.Thread(target=autopilot_loop, daemon=True).start()


def rows(sql, args=()):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(sql, args).fetchall()]
    con.close()
    return out


# ---------------- LOGIN: support staff vs customers ----------------
import base64
import hashlib
import hmac
import os
import secrets
import time as _time

SESSION_SECRET = (os.getenv("SESSION_SECRET") or secrets.token_hex(32)).encode()
STAFF_USER = os.getenv("STAFF_USERNAME", "support")
STAFF_PASSWORD = os.getenv("STAFF_PASSWORD", "refundiq-staff")   # set STAFF_PASSWORD on Render to change it
SESSION_HOURS = 12


def _sign(payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(SESSION_SECRET, body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def _read_token(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    try:
        body, sig = auth[7:].strip().rsplit(".", 1)
        good = hmac.new(SESSION_SECRET, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(base64.urlsafe_b64decode(body.encode()))
        return data if data.get("exp", 0) > _time.time() else None
    except Exception:
        return None


def require(request: Request, *roles):
    """Returns the signed-in user, or stops the request. roles: 'staff', 'customer'."""
    user = _read_token(request)
    if not user:
        raise HTTPException(401, "Please sign in")
    if roles and user["role"] not in roles:
        raise HTTPException(403, "Support staff only" if roles == ("staff",) else "Not allowed")
    return user


def owns_order(customer_id: str, order_id: str) -> bool:
    return bool(rows("SELECT 1 FROM orders WHERE order_id = ? AND customer_id = ?", (order_id, customer_id)))


class Login(BaseModel):
    role: str
    username: str | None = Field(default=None, max_length=40)
    password: str | None = Field(default=None, max_length=100)
    customer_id: str | None = Field(default=None, max_length=20)
    order_id: str | None = Field(default=None, max_length=20)


@app.post("/api/login")
def login(body: Login):
    exp = _time.time() + SESSION_HOURS * 3600
    if body.role == "staff":
        ok = hmac.compare_digest((body.username or "").strip(), STAFF_USER) and \
             hmac.compare_digest(body.password or "", STAFF_PASSWORD)
        if not ok:
            raise HTTPException(401, "Wrong staff username or password")
        return {"token": _sign({"role": "staff", "sub": STAFF_USER, "exp": exp}), "role": "staff", "name": "Support staff"}
    if body.role == "customer":
        cid, oid = (body.customer_id or "").strip().upper(), (body.order_id or "").strip().upper()
        if not (cid and oid and owns_order(cid, oid)):
            raise HTTPException(401, "Customer ID and Order ID don't match")
        name = rows("SELECT name FROM customers WHERE customer_id = ?", (cid,))[0]["name"]
        return {"token": _sign({"role": "customer", "sub": cid, "exp": exp}), "role": "customer",
                "name": name, "customer_id": cid}
    raise HTTPException(400, "Unknown role")


@app.get("/api/me")
def me(request: Request):
    u = _read_token(request)
    return u or {"role": None}


def unique_ticket_id(tid: str) -> str:
    """Never reuse a ticket id: a repeat gets -2, -3 ... so no ticket overwrites another in memory."""
    base = (tid or "").strip() or f"W{datetime.now().strftime('%H%M%S')}"
    con = sqlite3.connect(DB_PATH)
    def taken(t):
        return con.execute("SELECT 1 FROM tickets WHERE ticket_id=? UNION SELECT 1 FROM inbox WHERE ticket_id=?",
                           (t, t)).fetchone() is not None
    cand, k = base, 2
    while taken(cand):
        cand, k = f"{base}-{k}", k + 1
    con.close()
    return cand


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/tickets")
def tickets():
    return json.loads((BASE_DIR / "data" / "eval_set.json").read_text())


class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    order_id: str
    message: str
    photo_attached: bool = False
    serial_number: str | None = None


@app.post("/api/run")
def run(t: Ticket, request: Request):
    """Streams each agent step as one JSON line, then a final 'done' line."""
    require(request, "staff")
    q: queue.Queue = queue.Queue()

    def work():
        try:
            ticket = t.model_dump()
            ticket["ticket_id"] = unique_ticket_id(ticket["ticket_id"])
            q.put({"type": "ticket_id", "ticket_id": ticket["ticket_id"]})
            res = run_ticket(ticket, on_step=lambda s: q.put(s))
            q.put({"type": "done", "final": res["final"], "proposal": res["proposal"],
                   "llm_calls": res["llm_calls"], "tool_calls": res["tool_calls"]})
        except Exception as e:
            q.put({"type": "error", "text": f"{type(e).__name__}: {e}"})
            q.put({"type": "done", "final": None})

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            yield json.dumps(item, default=str) + "\n"
            if item.get("type") == "done":
                break

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---------------- CHAT: talk to the agent like a customer ----------------
class ChatMsg(BaseModel):
    role: str
    text: str = Field(max_length=1000)


class ChatReq(BaseModel):
    conversation_id: str | None = None
    customer_id: str = Field(min_length=1, max_length=20)
    order_id: str = Field(min_length=1, max_length=20)
    message: str = Field(min_length=1, max_length=1000)
    photo_attached: bool = False
    serial_number: str | None = Field(default=None, max_length=30)
    history: list[ChatMsg] = Field(default_factory=list, max_length=30)


def _chat_reply(final, proposal, ticket_id):
    """The reply must match the FINAL decision (after guardrails), not just what the AI wanted."""
    d, amt = final["final_decision"], final["refund_amount"]
    ai_reply = (proposal or {}).get("customer_reply")
    if ai_reply and not final["overridden"] and d not in ("APPROVE", "PARTIAL_APPROVE", "HOLD_FOR_HUMAN"):
        return ai_reply
    if d in ("APPROVE", "PARTIAL_APPROVE"):
        part = "a partial refund" if d == "PARTIAL_APPROVE" else "your refund"
        return f"Good news: {part} of Rs {amt:,.0f} has been approved and issued (ref RF-{ticket_id})."
    if d == "NEED_PROOF":
        return "Thanks. To process this, please attach a photo of the item and the problem."
    if d == "REJECT":
        return "Sorry, we can't refund this order under our return policy."
    return ("Thanks for the details. Your request needs a quick review by our team, "
            "and we'll update you shortly. No further action is needed from you.")


@app.get("/api/orders")
def list_orders(request: Request):
    u = require(request, "staff", "customer")
    if u["role"] == "customer":
        return rows("SELECT o.order_id, o.customer_id, c.name, o.item, o.price FROM orders o "
                    "JOIN customers c ON c.customer_id = o.customer_id WHERE o.customer_id = ? ORDER BY o.order_id", (u["sub"],))
    return rows("SELECT o.order_id, o.customer_id, c.name, o.item, o.price FROM orders o "
                "JOIN customers c ON c.customer_id = o.customer_id ORDER BY o.order_id")


@app.post("/api/chat")
def chat(req: ChatReq, request: Request):
    u = require(request, "staff", "customer")
    if u["role"] == "customer" and (req.customer_id != u["sub"] or not owns_order(u["sub"], req.order_id)):
        raise HTTPException(403, "You can only chat about your own orders")
    if not owns_order(req.customer_id, req.order_id):
        raise HTTPException(400, "That order doesn't belong to that customer")
    """One conversation = one ticket. Each customer message re-runs the agent with the whole
    conversation, so it can react to new information (like a photo) without looking like a repeat returner."""
    conv = req.conversation_id or unique_ticket_id(f"CHAT-{req.order_id}")
    q: queue.Queue = queue.Queue()

    def work():
        try:
            prev = rows("SELECT status, refund_amount FROM tickets WHERE ticket_id = ?", (conv,))
            st = prev[0]["status"] if prev else None
            if st in ("AUTO_REFUNDED", "HUMAN_APPROVED"):
                q.put({"type": "done", "conversation_id": conv, "skipped": True, "final": None,
                       "reply": f"Your refund of Rs {prev[0]['refund_amount']:,.0f} has already been issued for this order. "
                                "Is there anything else I can help with?"})
                return
            if st == "PENDING_HUMAN":
                q.put({"type": "done", "conversation_id": conv, "skipped": True, "final": None,
                       "reply": "Your request is already with our team for review. We'll update you as soon as they decide."})
                return
            lines = [f"{'Customer' if m.role == 'user' else 'RefundDesk'}: {m.text}" for m in req.history[-8:]]
            lines.append(f"Customer (latest message): {req.message}")
            ticket = {"ticket_id": conv, "customer_id": req.customer_id, "order_id": req.order_id,
                      "message": "Chat conversation so far:\n" + "\n".join(lines),
                      "photo_attached": req.photo_attached, "serial_number": req.serial_number}
            res = run_ticket(ticket, on_step=lambda s: q.put(s))
            q.put({"type": "done", "conversation_id": conv, "final": res["final"], "proposal": res["proposal"],
                   "reply": _chat_reply(res["final"], res["proposal"], conv),
                   "llm_calls": res["llm_calls"], "tool_calls": res["tool_calls"]})
        except Exception as e:
            q.put({"type": "done", "conversation_id": conv, "final": None,
                   "reply": "Sorry, something went wrong on our side. Your request has been passed to our team.",
                   "error": f"{type(e).__name__}: {e}"})

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            yield json.dumps(item, default=str) + "\n"
            if item.get("type") == "done":
                break

    return StreamingResponse(stream(), media_type="application/x-ndjson")


CATEGORIES = {"electronics", "apparel", "footwear", "accessories", "home_appliance", "home_decor",
              "kitchen", "furniture", "fitness", "innerwear", "grocery"}
MAX_CUSTOM_ORDERS = 300


class NewOrder(BaseModel):
    customer_name: str = Field(min_length=1, max_length=40)
    item: str = Field(min_length=1, max_length=60)
    category: str = "electronics"
    price: float = Field(gt=0, le=1_000_000)          # Rs 1 to Rs 10 lakh
    delivered_days_ago: int = Field(default=3, ge=0, le=365)
    in_transit: bool = False
    serial_number: str | None = Field(default=None, max_length=30)
    customer_id: str | None = Field(default=None, max_length=20)   # optional: choose your own IDs
    order_id: str | None = Field(default=None, max_length=20)


ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{1,19}$")


@app.post("/api/orders")
def create_order(o: NewOrder, request: Request):
    require(request, "staff")
    """Lets anyone add their own customer + order to the demo database, then test the agent on it."""
    from datetime import timedelta
    from config import TODAY
    from guardrails import detect_injection
    if o.category not in CATEGORIES:
        raise HTTPException(400, "Unknown category")
    for field in (o.customer_name, o.item, o.serial_number or ""):
        if detect_injection(field) or any(ch in field for ch in "<>{}`"):
            raise HTTPException(400, "That text isn't allowed in an order field")
    want_c = (o.customer_id or "").strip().upper()
    want_o = (o.order_id or "").strip().upper()
    for v, label in ((want_c, "Customer ID"), (want_o, "Order ID")):
        if v and not ID_PATTERN.match(v):
            raise HTTPException(400, f"{label} can use letters, numbers and dashes only (2-20 characters)")
    con = sqlite3.connect(DB_PATH)
    n = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    if n >= MAX_CUSTOM_ORDERS:
        con.close()
        raise HTTPException(429, "Demo order limit reached. Use Reset demo to clear it")
    if want_o and con.execute("SELECT 1 FROM orders WHERE order_id = ?", (want_o,)).fetchone():
        con.close()
        raise HTTPException(409, f"Order ID {want_o} already exists. Choose another one")
    k = 5001 + n
    while con.execute("SELECT 1 FROM orders WHERE order_id = ?", (f"O{k}",)).fetchone() or \
            con.execute("SELECT 1 FROM customers WHERE customer_id = ?", (f"C{k}",)).fetchone() or \
            con.execute("SELECT 1 FROM shipments WHERE tracking_id = ?", (f"TRK{k}",)).fetchone():
        k += 1
    cid, oid, trk = want_c or f"C{k}", want_o or f"O{k}", f"TRK{k}"
    existing = con.execute("SELECT name FROM customers WHERE customer_id = ?", (cid,)).fetchone()
    if o.in_transit:
        expected, delivered = (TODAY + timedelta(days=2)).isoformat(), None
        status = "IN_TRANSIT"
    else:
        expected = delivered = (TODAY - timedelta(days=max(0, o.delivered_days_ago))).isoformat()
        status = "DELIVERED"
    if not existing:   # an existing customer ID keeps its name and simply gets another order
        con.execute("INSERT INTO customers VALUES (?,?)", (cid, o.customer_name.strip() or "Guest"))
    con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
                (oid, cid, o.item.strip(), o.category, round(float(o.price), 2),
                 (TODAY - timedelta(days=max(0, o.delivered_days_ago) + 4)).isoformat(),
                 (o.serial_number or "").strip() or None, "", trk))
    con.execute("INSERT INTO shipments VALUES (?,?,?,?,?)", (trk, status, expected, delivered, None))
    con.commit()
    con.close()
    return {"customer_id": cid, "order_id": oid, "tracking_id": trk, "orders_in_db": n + 1,
            "existing_customer": bool(existing)}


@app.get("/api/queue")
def pending(request: Request):
    require(request, "staff")
    return rows("SELECT ticket_id, customer_id, order_id, refund_amount, flags, reasoning, status, reviewer_note, reviewed_at "
                "FROM tickets WHERE status IN ('PENDING_HUMAN','HUMAN_APPROVED','HUMAN_REJECTED') "
                "ORDER BY status DESC, ticket_id")


class Review(BaseModel):
    ticket_id: str
    approve: bool
    note: str = ""


@app.post("/api/review")
def review(r: Review, request: Request):
    staff = require(request, "staff")
    return {"ok": True, "execution": human_review(r.ticket_id, r.approve, r.note, reviewer=staff["sub"])}


@app.get("/api/customer/{cid}")
def customer(cid: str, request: Request):
    u = require(request, "staff", "customer")
    if u["role"] == "customer" and u["sub"] != cid:
        raise HTTPException(403, "You can only see your own history")
    c = rows("SELECT * FROM customers WHERE customer_id = ?", (cid,))
    t = rows("SELECT ticket_id, created_date, order_id, message, decision, refund_amount, status "
             "FROM tickets WHERE customer_id = ? ORDER BY created_date DESC", (cid,))
    return {"customer": c[0] if c else None, "tickets": t}


@app.get("/api/eval")
def eval_results():
    p = BASE_DIR / "eval_results.json"
    return json.loads(p.read_text()) if p.exists() else JSONResponse({"missing": True})


class InboxTicket(BaseModel):
    ticket_id: str | None = None
    customer_id: str
    order_id: str
    message: str
    photo_attached: bool = False
    serial_number: str | None = None


def _enqueue(t: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR REPLACE INTO inbox (ticket_id, customer_id, order_id, message, photo_attached,"
                " serial_number, received_at, state) VALUES (?,?,?,?,?,?,?, 'QUEUED')",
                (t["ticket_id"], t["customer_id"], t["order_id"], t["message"], int(bool(t.get("photo_attached"))),
                 t.get("serial_number"), datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")))
    con.commit()
    con.close()


@app.post("/api/inbox")
def submit(t: InboxTicket, request: Request):
    u = require(request, "staff", "customer")
    if u["role"] == "customer" and (t.customer_id != u["sub"] or not owns_order(u["sub"], t.order_id)):
        raise HTTPException(403, "You can only raise tickets for your own orders")
    """A customer submits a ticket. The autopilot picks it up automatically."""
    d = t.model_dump()
    d["ticket_id"] = unique_ticket_id(d["ticket_id"])
    _enqueue(d)
    return {"ok": True, "ticket_id": d["ticket_id"]}


@app.post("/api/inbox/samples")
def load_samples(request: Request):
    require(request, "staff")
    for t in json.loads((BASE_DIR / "data" / "eval_set.json").read_text()):
        t = dict(t)
        t["ticket_id"] = unique_ticket_id(t["ticket_id"])
        _enqueue(t)
    return {"ok": True}


@app.get("/api/inbox")
def inbox(request: Request):
    require(request, "staff")
    return rows("SELECT ticket_id, customer_id, order_id, message, state, decision, refund_amount, status, finished_at "
                "FROM inbox ORDER BY received_at DESC LIMIT 60")


@app.get("/api/autopilot")
def autopilot_status(request: Request):
    is_staff = (_read_token(request) or {}).get("role") == "staff"
    counts = {r["state"]: r["n"] for r in rows("SELECT state, COUNT(*) n FROM inbox GROUP BY state")}
    dec = {r["decision"]: r["n"] for r in rows("SELECT decision, COUNT(*) n FROM inbox WHERE state='DONE' GROUP BY decision")}
    refunded = rows("SELECT COALESCE(SUM(refund_amount),0) s FROM inbox WHERE status='AUTO_REFUNDED'")[0]["s"]
    return {"on": AUTOPILOT["on"], "current": AUTOPILOT["current"] if is_staff else None,
            "steps": AUTOPILOT["steps"][-12:] if is_staff else [], "busy": AUTOPILOT["current"] is not None,
            "counts": counts, "decisions": dec, "auto_refunded_inr": refunded, "last_error": AUTOPILOT["last_error"]}


class Toggle(BaseModel):
    on: bool


@app.post("/api/autopilot")
def autopilot_toggle(t: Toggle, request: Request):
    require(request, "staff")
    AUTOPILOT["on"] = t.on
    return {"on": t.on}


@app.post("/api/reset")
def reset(request: Request):
    require(request, "staff")
    reset_db()
    ensure_inbox()
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM inbox")
    con.commit()
    con.close()
    return {"ok": True}


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8000"))           # Render sets PORT
    on_server = bool(os.getenv("RENDER") or os.getenv("PORT"))
    host = "0.0.0.0" if on_server else "127.0.0.1"
    print(f"RefundIQ running at http://localhost:{port}  (autopilot ON - leave this window open)")
    if not on_server:
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
