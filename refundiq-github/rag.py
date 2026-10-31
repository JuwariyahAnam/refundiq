"""RAG over the written return policy.
Splits policy.md into clauses (P1..P12), builds a TF-IDF index, and returns
the most relevant clauses for a query. Pure Python: no extra API calls, fast, deterministic."""
import math
import re
from collections import Counter

from config import POLICY_PATH

STOP = set("""a an the and or of to in on for with is are be by as it its this that if
within must may can not no any than more less from at was were has have""".split())


def _tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOP:
            continue
        # tiny stemmer so "damaged"/"damage", "returns"/"return" match
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def load_clauses():
    text = POLICY_PATH.read_text(encoding="utf-8")
    clauses = []
    for block in text.split("\n## ")[1:]:
        header, _, body = block.partition("\n")
        cid, _, title = header.partition(" - ")
        clauses.append({"clause_id": cid.strip(), "title": title.strip(), "text": body.strip()})
    return clauses


class PolicyIndex:
    def __init__(self):
        self.clauses = load_clauses()
        docs = [_tokens(c["title"] + " " + c["title"] + " " + c["text"]) for c in self.clauses]
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in df}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, toks):
        tf = Counter(toks)
        v = {t: c * self.idf.get(t, 0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query, k=3):
        q = self._vec(_tokens(query))
        scored = [(sum(q.get(t, 0) * w for t, w in v.items()), c)
                  for v, c in zip(self.vecs, self.clauses)]
        scored.sort(key=lambda x: -x[0])
        return [{**c, "score": round(s, 3)} for s, c in scored[:k] if s > 0]


_index = None


def search_policy(query, k=3):
    global _index
    if _index is None:
        _index = PolicyIndex()
    return _index.search(query, k)
