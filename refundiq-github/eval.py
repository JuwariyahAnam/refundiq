"""Runs the full evaluation set and reports decision accuracy.
Run: python eval.py            (all tickets)
     python eval.py T01 T12    (only some tickets)"""
import json
import os
import sys
import time

# The eval gets its OWN database, so it never collides with the live demo server
os.environ.setdefault("REFUNDIQ_DB", "eval.db")
from collections import Counter, defaultdict

from agent import run_ticket
from config import BASE_DIR
from seed import reset_db
from tools import get_order

MANUAL_MINUTES_PER_TICKET = 8  # assumed manual handling time (state this as an assumption)
LEGIT = ("APPROVE", "PARTIAL_APPROVE")

PAUSE_SECS = 6  # gap between tickets to respect free-tier rate limits


def main(only=None):
    tickets = json.loads((BASE_DIR / "data" / "eval_set.json").read_text())
    if only:
        tickets = [t for t in tickets if t["ticket_id"] in only]
    reset_db()  # same starting state every run

    rows, by_cat = [], defaultdict(lambda: [0, 0])
    confusion = Counter()
    t_start = time.time()
    for i, t in enumerate(tickets):
        ticket = {k: v for k, v in t.items() if k not in ("expected", "category", "expected_amount")}
        t0 = time.time()
        res = run_ticket(ticket)
        f = res["final"]
        ok = f["final_decision"] == t["expected"]
        by_cat[t["category"]][0] += ok
        by_cat[t["category"]][1] += 1
        confusion[(t["expected"], f["final_decision"])] += 1
        rows.append({"ticket_id": t["ticket_id"], "order_id": t["order_id"], "category": t["category"], "expected": t["expected"],
                     "llm_decision": f["llm_decision"], "final_decision": f["final_decision"],
                     "refund_amount": f["refund_amount"], "correct": ok, "overridden": f["overridden"],
                     "flags": list(f["flags"].keys()), "notes": f["guardrail_notes"],
                     "expected_amount": t.get("expected_amount"),
                     "tool_errors": sum(1 for s in res["trace"] if s["type"] == "tool" and isinstance(s["result"], dict) and "error" in s["result"]),
                     "self_corrections": sum(1 for s in res["trace"] if s["type"] == "critique"),
                     "llm_calls": res["llm_calls"], "tool_calls": res["tool_calls"],
                     "seconds": round(time.time() - t0, 1)})
        print(f"{'PASS' if ok else 'FAIL'}  {t['ticket_id']}  [{t['category']:<16}] expected={t['expected']:<15}"
              f" got={f['final_decision']:<15} llm={f['llm_decision']}  flags={list(f['flags'])}")
        if i < len(tickets) - 1:
            time.sleep(PAUSE_SECS)

    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    abuse = [r for r in rows if r["category"] == "abuse"]
    summary = {
        "tickets": n,
        "accuracy": round(correct / n, 3) if n else 0,
        "correct": correct,
        "auto_refunded": sum(r["final_decision"] in ("APPROVE", "PARTIAL_APPROVE") for r in rows),
        "held_for_human": sum(r["final_decision"] == "HOLD_FOR_HUMAN" for r in rows),
        "abuse_caught": f"{sum(r['correct'] for r in abuse)}/{len(abuse)}",
        "unsafe_auto_refunds": sum(r["final_decision"] in ("APPROVE", "PARTIAL_APPROVE")
                                   and r["expected"] not in ("APPROVE", "PARTIAL_APPROVE") for r in rows),
        "guardrail_overrides": sum(r["overridden"] for r in rows),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 1) if n else 0,
        "by_category": {c: f"{a}/{b}" for c, (a, b) in by_cat.items()},
        "total_seconds": round(time.time() - t_start, 1),
    }
    # ---- business impact & extra reliability metrics ----
    avg_secs = sum(r["seconds"] for r in rows) / n if n else 0
    risky = [r for r in rows if r["category"] in ("abuse", "duplicate", "prompt injection")]
    blocked = sum(get_order(r["order_id"]).get("price_paid_inr", 0) or 0
                  for r in risky if r["final_decision"] not in LEGIT)
    amt_rows = [r for r in rows if r["expected_amount"] is not None]
    amt_ok = sum(1 for r in amt_rows if r["final_decision"] in LEGIT and abs(r["refund_amount"] - r["expected_amount"]) < 1)
    legit = [r for r in rows if r["expected"] in LEGIT]
    false_pos = sum(1 for r in legit if r["final_decision"] not in LEGIT)
    needs_human = [r for r in rows if r["expected"] == "HOLD_FOR_HUMAN"]
    summary.update({
        "avg_seconds_per_ticket": round(avg_secs, 1),
        "manual_minutes_per_ticket_assumed": MANUAL_MINUTES_PER_TICKET,
        "speedup_x": round(MANUAL_MINUTES_PER_TICKET * 60 / avg_secs) if avg_secs else 0,
        "hours_saved_per_1000_tickets": round(1000 * (MANUAL_MINUTES_PER_TICKET * 60 - avg_secs) / 3600),
        "resolved_without_human_pct": round(100 * sum(r["final_decision"] != "HOLD_FOR_HUMAN" for r in rows) / n) if n else 0,
        "abuse_blocked_inr": blocked,
        "refund_amount_accuracy": f"{amt_ok}/{len(amt_rows)}",
        "false_positives_on_legit": f"{false_pos}/{len(legit)}",
        "escalation_accuracy": f"{sum(r['correct'] for r in needs_human)}/{len(needs_human)}",
        "self_corrections": sum(r["self_corrections"] for r in rows),
        "tool_error_handled": sum(r["tool_errors"] > 0 and r["final_decision"] == "HOLD_FOR_HUMAN" for r in rows),
    })
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>34}: {v}")
    out = {"summary": summary, "results": rows,
           "confusion": [{"expected": e, "got": g, "count": c} for (e, g), c in confusion.items()]}
    (BASE_DIR / "eval_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved eval_results.json")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
