"""Lightweight inverted-index search engine with cosine TF-IDF ranking."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math
import re
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
STOP_WORDS = frozenset('a an and are as at be been by for from has have in into is it its of on or that the their there these this to was were will with you your we our not'.split())


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    terms = TOKEN_RE.findall(text.lower())
    return [t for t in terms if t not in STOP_WORDS] if remove_stopwords else terms


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str
    source: str = "Sample collection"


@dataclass(frozen=True)
class Result:
    document: Document
    score: float
    snippet: str
    matched_terms: tuple[str, ...]


class SearchEngine:
    def __init__(self, documents: Iterable[Document] = ()) -> None:
        self.documents: dict[str, Document] = {}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.doc_lengths: dict[str, int] = {}
        self.doc_norms: dict[str, float] = {}
        self.idf: dict[str, float] = {}
        self._tokens: dict[str, list[str]] = {}
        self.rebuild(documents)

    def rebuild(self, documents: Iterable[Document]) -> None:
        self.documents = {}
        self.postings = defaultdict(dict)
        self.doc_lengths = {}
        self.doc_norms = {}
        self.idf = {}
        self._tokens = {}
        for doc in documents:
            if doc.doc_id in self.documents:
                raise ValueError(f"Duplicate document ID: {doc.doc_id}")
            self.documents[doc.doc_id] = doc
            terms = tokenize(f"{doc.title} {doc.content}")
            self._tokens[doc.doc_id] = terms
            self.doc_lengths[doc.doc_id] = len(terms)
            for term, count in Counter(terms).items():
                self.postings[term][doc.doc_id] = count
        n = len(self.documents)
        for term, postings in self.postings.items():
            self.idf[term] = math.log((1 + n) / (1 + len(postings))) + 1
        for doc_id, terms in self._tokens.items():
            counts = Counter(terms)
            self.doc_norms[doc_id] = math.sqrt(sum((1 + math.log(tf)) ** 2 * self.idf[t] ** 2 for t, tf in counts.items()))

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def search(self, query: str, top_k: int = 5, mode: str = "OR") -> list[Result]:
        if top_k < 1:
            return []
        raw_terms = Counter(tokenize(query))
        mode = mode.upper()
        if mode not in {"OR", "AND"}:
            raise ValueError("mode must be OR or AND")
        if mode == "AND" and any(t not in self.postings for t in raw_terms):
            return []
        terms = Counter({t: count for t, count in raw_terms.items() if t in self.postings})
        if not terms:
            return []
        sets = [set(self.postings[t]) for t in terms]
        candidates = set.intersection(*sets) if mode == "AND" else set.union(*sets)
        qweights = {t: (1 + math.log(tf)) * self.idf[t] for t, tf in terms.items()}
        qnorm = math.sqrt(sum(w * w for w in qweights.values()))
        scored = []
        for doc_id in candidates:
            numerator = sum(qw * (1 + math.log(self.postings[t][doc_id])) * self.idf[t]
                            for t, qw in qweights.items() if doc_id in self.postings[t])
            denominator = qnorm * self.doc_norms[doc_id]
            if denominator:
                score = numerator / denominator
                if score > 0:
                    scored.append((score, doc_id))
        best = heapq.nlargest(top_k, scored, key=lambda pair: (pair[0], pair[1]))
        results = []
        for score, doc_id in best:
            doc = self.documents[doc_id]
            hits = tuple(t for t in terms if doc_id in self.postings[t])
            results.append(Result(doc, score, make_snippet(doc.content, hits), hits))
        return results


def make_snippet(content: str, terms: tuple[str, ...], max_chars: int = 230) -> str:
    if len(content) <= max_chars:
        return content
    lowered = content.lower()
    positions = [match.start() for t in terms for match in re.finditer(r'\b' + re.escape(t) + r'\b', lowered)]
    start = max(0, min(positions) - 55) if positions else 0
    if start:
        next_space = content.find(' ', start)
        start = next_space + 1 if next_space != -1 else start
    end = min(len(content), start + max_chars)
    return ('…' if start else '') + content[start:end].rstrip() + ('…' if end < len(content) else '')
