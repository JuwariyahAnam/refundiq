"""Streamlit demo UI.  Run: streamlit run app.py"""
import json
import sqlite3

import pandas as pd
import streamlit as st

from agent import human_review, run_ticket
from config import BASE_DIR, DB_PATH
from seed import reset_db

st.set_page_config(page_title="Refund Desk Agent", page_icon="📦", layout="wide")
if not DB_PATH.exists():
    reset_db()

EVAL = json.loads((BASE_DIR / "data" / "eval_set.json").read_text())
COLORS = {"APPROVE": "green", "PARTIAL_APPROVE": "green", "NEED_PROOF": "orange",
          "REJECT": "red", "HOLD_FOR_HUMAN": "violet"}


def q(sql, args=()):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, con, params=args)
    con.close()
    return df


st.title("📦 E-Commerce Refund & Returns Desk Agent")
st.caption("Gemini agent · 5 tools (order DB, courier API, policy RAG, customer memory, proof request) · code guardrails · human approval")

with st.sidebar:
    st.header("Controls")
    if st.button("Reset demo database"):
        reset_db()
        st.success("Database reset")

tab1, tab2, tab3, tab4 = st.tabs(["🎫 Process ticket", "🧑‍⚖️ Human approval queue", "🧠 Customer memory", "📊 Evaluation"])

with tab1:
    labels = [f"{t['ticket_id']} · {t['category']} · {t['message'][:60]}" for t in EVAL] + ["✍️ Custom ticket"]
    choice = st.selectbox("Pick a sample ticket", labels)
    base = EVAL[labels.index(choice)] if choice != labels[-1] else \
        {"ticket_id": "T99", "customer_id": "C001", "order_id": "O1001", "photo_attached": False,
         "serial_number": "", "message": ""}
    c1, c2, c3 = st.columns(3)
    ticket = {
        "ticket_id": c1.text_input("Ticket ID", base["ticket_id"]),
        "customer_id": c2.text_input("Customer ID", base["customer_id"]),
        "order_id": c3.text_input("Order ID", base["order_id"]),
    }
    c4, c5 = st.columns(2)
    ticket["photo_attached"] = c4.checkbox("Photo attached", base.get("photo_attached", False))
    ticket["serial_number"] = c5.text_input("Serial number supplied", base.get("serial_number") or "")
    ticket["message"] = st.text_area("Customer message", base["message"], height=90)
    if base.get("expected"):
        st.caption(f"Expected decision for this eval ticket: **{base['expected']}**")

    if st.button("▶ Run agent", type="primary"):
        st.subheader("Agent trace (plan → act → observe → re-plan)")
        box = st.container()

        def show(step):
            if step["type"] == "thought":
                box.info(f"🧭 {step['text']}")
            elif step["type"] == "tool":
                bad = isinstance(step["result"], dict) and "error" in step["result"]
                with box.expander(f"{'⚠️' if bad else '🔧'} {step['name']}({json.dumps(step['args'])})", expanded=bad):
                    st.json(step["result"])
            elif step["type"] == "decision":
                box.write(f"🤖 LLM proposed: **{step['args'].get('decision')}** · Rs {step['args'].get('refund_amount')}")
            elif step["type"] == "error":
                box.error(step["text"])

        with st.spinner("Agent working..."):
            res = run_ticket(ticket, on_step=show)
        f, p = res["final"], res["proposal"] or {}
        st.subheader("Final decision")
        st.markdown(f"### :{COLORS[f['final_decision']]}[{f['final_decision']}] · Rs {f['refund_amount']:.0f} · {f['status']}")
        if f["overridden"]:
            st.warning(f"🛡️ Guardrail overrode the LLM's {f['llm_decision']}")
        for n in f["guardrail_notes"]:
            st.write(f"🛡️ {n}")
        for k, v in f["flags"].items():
            st.write(f"🚩 **{k}**: {v}")
        if p:
            st.markdown(f"**Policy cited:** {p.get('clause_ids')} — _{p.get('clause_quote')}_")
            st.markdown(f"**Reasoning:** {p.get('reasoning')}")
            st.markdown(f"**Reply to customer:** {p.get('customer_reply')}")
        st.caption(f"LLM calls: {res['llm_calls']} · tool calls: {res['tool_calls']}")

with tab2:
    st.subheader("Refunds waiting for a human")
    pending = q("SELECT ticket_id, customer_id, order_id, decision, refund_amount, flags, reasoning "
                "FROM tickets WHERE status = 'PENDING_HUMAN' ORDER BY ticket_id")
    if pending.empty:
        st.info("Nothing waiting. Run a high-value or flagged ticket first.")
    for _, r in pending.iterrows():
        with st.container(border=True):
            st.markdown(f"**{r.ticket_id}** · customer {r.customer_id} · order {r.order_id} · proposed Rs {r.refund_amount:.0f}")
            st.write(f"Flags: {r.flags}")
            st.caption(r.reasoning or "")
            note = st.text_input("Reviewer note", key=f"n{r.ticket_id}")
            a, b = st.columns(2)
            if a.button("✅ Approve refund", key=f"a{r.ticket_id}"):
                human_review(r.ticket_id, True, note)
                st.rerun()
            if b.button("❌ Reject", key=f"r{r.ticket_id}"):
                human_review(r.ticket_id, False, note)
                st.rerun()
    st.subheader("Reviewed")
    st.dataframe(q("SELECT ticket_id, customer_id, refund_amount, status, reviewer_note FROM tickets "
                   "WHERE status IN ('HUMAN_APPROVED','HUMAN_REJECTED')"), use_container_width=True)

with tab3:
    cid = st.text_input("Customer ID", "C004")
    st.dataframe(q("SELECT ticket_id, created_date, order_id, message, decision, refund_amount, status "
                   "FROM tickets WHERE customer_id = ? ORDER BY created_date DESC", (cid,)), use_container_width=True)
    st.caption("Every decision is saved here and read back by the agent on the customer's next ticket.")

with tab4:
    path = BASE_DIR / "eval_results.json"
    if not path.exists():
        st.info("Run `python eval.py` in the terminal, then refresh this tab.")
    else:
        data = json.loads(path.read_text())
        s = data["summary"]
        m = st.columns(5)
        m[0].metric("Decision accuracy", f"{s['accuracy']*100:.0f}%", f"{s['correct']}/{s['tickets']}")
        m[1].metric("Auto-refunded", s["auto_refunded"])
        m[2].metric("Held for human", s["held_for_human"])
        m[3].metric("Abuse caught", s["abuse_caught"])
        m[4].metric("Unsafe auto-refunds", s["unsafe_auto_refunds"])
        st.write("By category:", s["by_category"])
        st.dataframe(pd.DataFrame(data["results"])[["ticket_id", "category", "expected", "llm_decision",
                     "final_decision", "refund_amount", "correct", "overridden", "flags", "tool_calls"]],
                     use_container_width=True)
