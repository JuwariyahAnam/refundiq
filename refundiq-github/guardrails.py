"""Hard guardrails enforced in CODE after the LLM decides.
The LLM proposes; these rules have the final say, so money never moves on an LLM mistake."""
import re

from config import AUTO_REFUND_LIMIT, REPEAT_RETURN_THRESHOLD, REPEAT_RETURN_WINDOW_DAYS
from tools import check_refund_status, get_customer_history, get_order

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"system prompt", r"admin mode", r"developer mode", r"you are now",
    r"override (the )?(policy|rules)", r"approve (it|this|my refund) (immediately|now)",
    r"disregard (the )?(policy|rules)",
]


def detect_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def compute_flags(ticket: dict) -> dict:
    """Independent checks straight from the database - does not trust the LLM."""
    flags = {}
    hist = get_customer_history(ticket["customer_id"])
    n = hist.get(f"refund_requests_last_{REPEAT_RETURN_WINDOW_DAYS}_days", 0)
    if n >= REPEAT_RETURN_THRESHOLD:
        flags["repeat_returner"] = f"{n} refund requests in last {REPEAT_RETURN_WINDOW_DAYS} days (P10)"
    order = get_order(ticket["order_id"])
    supplied = (ticket.get("serial_number") or "").strip()
    if "error" not in order and supplied and order.get("serial_number"):
        if supplied.upper() != order["serial_number"].upper():
            flags["serial_mismatch"] = f"customer gave {supplied}, record says {order['serial_number']} (P8)"
    dup = check_refund_status(ticket["order_id"])
    if dup.get("already_refunded") and dup["existing_refund"]["ticket_id"] != ticket.get("ticket_id"):
        r = dup["existing_refund"]
        flags["duplicate_refund"] = f"order already refunded Rs {r['amount']:.0f} on {r['created_at'][:10]} ({r['refund_id']})"
    if detect_injection(ticket.get("message", "")):
        flags["prompt_injection"] = "ticket text tries to override agent instructions"
    return flags, order


def apply_guardrails(ticket: dict, proposal: dict | None, tool_errors: list) -> dict:
    flags, order = compute_flags(ticket)
    notes = []
    if proposal is None:
        if tool_errors:
            flags["tool_failure"] = "; ".join(tool_errors)
        return _final("HOLD_FOR_HUMAN", 0, flags, ["Agent did not reach a decision -> escalated to human"], proposal,
                      risk_assessment(ticket, flags, order))

    decision = proposal.get("decision", "HOLD_FOR_HUMAN")
    amount = float(proposal.get("refund_amount") or 0)

    if tool_errors and decision != "NEED_PROOF":
        flags["tool_failure"] = "; ".join(tool_errors)
        notes.append("A system/tool failure occurred -> cannot verify facts, escalated to human")
        decision = "HOLD_FOR_HUMAN"

    if "duplicate_refund" in flags and decision != "REJECT" and not tool_errors:
        notes.append(f"Overrode {decision}: this order was already refunded - duplicate blocked")
        decision = "REJECT"

    if "serial_mismatch" in flags and decision != "HOLD_FOR_HUMAN":
        notes.append(f"Overrode {decision}: serial number mismatch must go to fraud review (P8)")
        decision = "HOLD_FOR_HUMAN"

    if decision in ("APPROVE", "PARTIAL_APPROVE"):
        price = order.get("price_paid_inr") if isinstance(order, dict) else None
        if price is not None and amount > price:
            notes.append(f"Refund capped from {amount} to price paid {price} (P11)")
            amount = float(price)
        if "repeat_returner" in flags:
            notes.append(f"Overrode {decision}: repeat returner needs human review (P10)")
            decision = "HOLD_FOR_HUMAN"
        elif "prompt_injection" in flags:
            notes.append(f"Overrode {decision}: ticket contained an instruction-injection attempt")
            decision = "HOLD_FOR_HUMAN"
        elif amount > AUTO_REFUND_LIMIT:
            notes.append(f"Rs {amount:.0f} is above auto-refund limit Rs {AUTO_REFUND_LIMIT} -> human approval (P9)")
            decision = "HOLD_FOR_HUMAN"
        elif not proposal.get("clause_ids"):
            notes.append("No policy clause cited -> escalated to human")
            decision = "HOLD_FOR_HUMAN"

    if decision in ("REJECT", "NEED_PROOF"):
        amount = 0.0
    return _final(decision, amount, flags, notes, proposal, risk_assessment(ticket, flags, order))


RISK_WEIGHTS = {"duplicate_refund": 40, "serial_mismatch": 40, "repeat_returner": 35,
                "prompt_injection": 30, "tool_failure": 15}
DAMAGE_WORDS = ("damaged", "broken", "cracked", "defective", "not working", "stopped working", "won't charge", "leaking")


def risk_assessment(ticket, flags, order):
    """Transparent risk score: every point comes from a named signal. Risk triggers review, never auto-reject."""
    signals = [(k, w, flags[k]) for k, w in RISK_WEIGHTS.items() if k in flags]
    price = order.get("price_paid_inr") if isinstance(order, dict) else None
    if price and price > AUTO_REFUND_LIMIT:
        signals.append(("high_value", 15, f"order value Rs {price:.0f} is above Rs {AUTO_REFUND_LIMIT}"))
    msg = ticket.get("message", "").lower()
    if any(w in msg for w in DAMAGE_WORDS) and not ticket.get("photo_attached"):
        signals.append(("unproven_damage_claim", 10, "damage claimed with no photo"))
    score = min(100, sum(w for _, w, _ in signals))
    level = "high" if score >= 35 else "medium" if score >= 15 else "low"
    return {"score": score, "level": level,
            "signals": [{"signal": n, "points": w, "reason": r} for n, w, r in signals]}


def _final(decision, amount, flags, notes, proposal, risk=None):
    status = {"APPROVE": "AUTO_REFUNDED", "PARTIAL_APPROVE": "AUTO_REFUNDED",
              "NEED_PROOF": "AWAITING_PROOF", "REJECT": "REJECTED",
              "HOLD_FOR_HUMAN": "PENDING_HUMAN"}[decision]
    llm_decision = proposal.get("decision") if proposal else None
    return {"final_decision": decision, "refund_amount": round(amount, 2), "status": status,
            "flags": flags, "guardrail_notes": notes, "llm_decision": llm_decision,
            "overridden": llm_decision is not None and llm_decision != decision, "risk": risk}
