"""Small local corpus and lexical retrieval; no external index required."""
import json
import math
import re
from collections import Counter
from pathlib import Path


def tokens(text: str) -> list[str]:
    # Character bigrams retain Chinese phrase order; single characters cover short terms.
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower())
    return [t for p in parts for t in ([p] if p.isascii() else list(p) + [p[i:i+2] for i in range(len(p)-1)])]


class Corpus:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.docs = json.loads(self.path.read_text(encoding="utf-8"))
        self.chunks = []
        for doc in self.docs:
            for chunk in doc["chunks"]:
                self.chunks.append({**chunk, "doc_id": doc["doc_id"], "version_date": doc["version_date"], "title": doc["title"], "scope": doc.get("scope", "")})
        ids = [(c["doc_id"], c["chunk_id"]) for c in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate document/chunk ID")
        self.index = {key: value for key, value in zip(ids, self.chunks)}
        self.terms = [Counter(tokens(c["title"] + " " + c["text"] + " " + c["scope"])) for c in self.chunks]
        self.df = Counter(t for row in self.terms for t in row)
        self.avg_length = sum(map(lambda x: sum(x.values()), self.terms)) / max(len(self.terms), 1)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        n = len(self.chunks)
        scored = []
        for c, terms in zip(self.chunks, self.terms):
            length = sum(terms.values())
            score = 0.0
            for term in set(tokens(query)):
                freq = terms[term]
                if freq:
                    idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / max(self.avg_length, 1)))
            # Tiny lexical overlaps such as 的/是 do not constitute evidence.
            if score >= 2.5:
                scored.append((score, c))
        scored.sort(key=lambda x: (-x[0], x[1]["doc_id"]))
        return [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "title": c["title"], "version_date": c["version_date"], "scope": c["scope"], "snippet": c["text"][:100], "score": round(score, 3)} for score, c in scored[:top_k]]

    def read(self, doc_id: str, chunk_id: str) -> dict:
        return dict(self.index[(doc_id, chunk_id)])
