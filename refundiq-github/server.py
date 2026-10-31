"""Web server for the custom frontend.
Run:  python server.py   then open http://localhost:8000"""
import json
import queue
import sqlite3
import threading
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

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
def run(t: Ticket):
    """Streams each agent step as one JSON line, then a final 'done' line."""
    q: queue.Queue = queue.Queue()

    def work():
        try:
            res = run_ticket(t.model_dump(), on_step=lambda s: q.put(s))
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


@app.get("/api/queue")
def pending():
    return rows("SELECT ticket_id, customer_id, order_id, refund_amount, flags, reasoning, status, reviewer_note, reviewed_at "
                "FROM tickets WHERE status IN ('PENDING_HUMAN','HUMAN_APPROVED','HUMAN_REJECTED') "
                "ORDER BY status DESC, ticket_id")


class Review(BaseModel):
    ticket_id: str
    approve: bool
    note: str = ""


@app.post("/api/review")
def review(r: Review):
    return {"ok": True, "execution": human_review(r.ticket_id, r.approve, r.note)}


@app.get("/api/customer/{cid}")
def customer(cid: str):
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
def submit(t: InboxTicket):
    """A customer submits a ticket. The autopilot picks it up automatically."""
    d = t.model_dump()
    d["ticket_id"] = d["ticket_id"] or f"W{datetime.now().strftime('%H%M%S')}"
    _enqueue(d)
    return {"ok": True, "ticket_id": d["ticket_id"]}


@app.post("/api/inbox/samples")
def load_samples():
    for t in json.loads((BASE_DIR / "data" / "eval_set.json").read_text()):
        _enqueue(t)
    return {"ok": True}


@app.get("/api/inbox")
def inbox():
    return rows("SELECT ticket_id, customer_id, order_id, message, state, decision, refund_amount, status, finished_at "
                "FROM inbox ORDER BY received_at DESC LIMIT 60")


@app.get("/api/autopilot")
def autopilot_status():
    counts = {r["state"]: r["n"] for r in rows("SELECT state, COUNT(*) n FROM inbox GROUP BY state")}
    dec = {r["decision"]: r["n"] for r in rows("SELECT decision, COUNT(*) n FROM inbox WHERE state='DONE' GROUP BY decision")}
    refunded = rows("SELECT COALESCE(SUM(refund_amount),0) s FROM inbox WHERE status='AUTO_REFUNDED'")[0]["s"]
    return {"on": AUTOPILOT["on"], "current": AUTOPILOT["current"], "steps": AUTOPILOT["steps"][-12:],
            "counts": counts, "decisions": dec, "auto_refunded_inr": refunded, "last_error": AUTOPILOT["last_error"]}


class Toggle(BaseModel):
    on: bool


@app.post("/api/autopilot")
def autopilot_toggle(t: Toggle):
    AUTOPILOT["on"] = t.on
    return {"on": t.on}


@app.post("/api/reset")
def reset():
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
