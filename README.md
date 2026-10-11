# 🔎 IndexLab — Search Engine From Scratch

A Python search engine showcasing **inverted indexing, TF–IDF weighting, cosine-similarity ranking, dictionary/hash-map retrieval, and heap-based top-K results**. Includes an interactive Streamlit UI, sample corpus, and unit tests.
deployes at: https://document-search-engine-ywzdxbnp6r5mc4jf9u74gw.streamlit.app/ 
## Run locally

Prerequisites: Python 3.10+ and pip.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown by Streamlit (usually http://localhost:8501).

## Publish for free using Streamlit Community Cloud

1. Extract this project, create a public GitHub repository (for example, `search-engine`), and upload **all project files and folders** to its root, including `app.py`, `requirements.txt`, `data`, and `search_engine`.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose your repository/branch, and set the main file path to `app.py`.
4. Click **Deploy** and use the generated public link in your resume.

Uploads are session-only. A free cloud instance may sleep when inactive; the sample collection is always available. This is an educational single-process search engine, not a distributed web crawler.

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

## Run tests

```bash
python -m unittest discover -s tests -v
```

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
