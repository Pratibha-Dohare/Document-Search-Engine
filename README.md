# 🔎 IndexLab — Search Engine From Scratch

A Python search engine showcasing **inverted indexing, TF–IDF weighting, cosine-similarity ranking, dictionary/hash-map retrieval, and heap-based top-K results**. Includes an interactive Streamlit UI, sample corpus, and unit tests.
deployed at: https://document-search-engine-ywzdxbnp6r5mc4jf9u74gw.streamlit.app/ 



## Features

- Built-in collection of 10 example documents.
- Upload multiple `.txt`, `.md`, and **text-based** `.pdf` documents (max 5 MB per file).
- Boolean AND / OR query term matching.
- TF–IDF cosine ranking and efficient `heapq.nlargest` top-K retrieval.
- Index inspector with postings and term frequency.
- Search snippets, relevance scores, and full document preview.
- Unit tests with no cloud services or API keys required.

## Ranking algorithm

Documents are tokenized and normalized to lowercase (with a small English stop-word list). A hash-map inverted index stores `term -> {doc_id: frequency}`. Every term gets smoothed inverse document frequency:

`IDF(t) = ln((1 + N) / (1 + df(t))) + 1`

A term's TF–IDF weight is `(1 + ln(tf)) * IDF(t)` for positive tf. The query and document TF–IDF vectors are scored by cosine similarity. A heap chooses the `K` highest-scoring matches without requiring a complete sort of all candidate documents.

Building the initial postings is approximately **O(T)** for T total tokens, plus the scoring/normalization pass over unique document terms. Hash-map posting retrieval is average-case O(1) per term. For C candidate documents, scoring takes time proportional to query-term postings examined and top-K selection takes **O(C log K)**.

## Project layout

```
app.py                          Streamlit frontend
search_engine/engine.py         Tokenizer, indexer, cosine TF–IDF, heap ranking
search_engine/__init__.py       Package exports
data/sample_documents.json      Sample corpus
tests/test_engine.py            Unit tests
.streamlit/config.toml          UI theme
requirements.txt                Runtime dependencies
```

## Limitations and extensions

- Stop words are removed without stemming or lemmatization.
- Uploaded documents stay in memory only during the visitor's current session.
- Scanned/image-only PDFs need OCR before their text can be indexed.
- There is no persistent database, authentication, crawler, or distributed sharding.
- Extensions: BM25 ranking, typo tolerance, persistent SQLite storage, ingestion pipelines, caching, API endpoints, or distributed indexing.
