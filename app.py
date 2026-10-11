"""Streamlit frontend for an explainable, deployable TF-IDF search engine."""
import io
import json
from pathlib import Path
import uuid

import streamlit as st
from pypdf import PdfReader

from search_engine import Document, SearchEngine

st.set_page_config(page_title="IndexLab | Search Engine", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1060px; padding-top: 2rem;}
[data-testid="stMetric"] {background: rgba(110,120,140,.08); border-radius: 12px; padding: 14px;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def sample_documents():
    path = Path(__file__).parent / "data" / "sample_documents.json"
    return [Document(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def read_upload(upload):
    raw = upload.getvalue()
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError("Each document must be 5 MB or smaller.")
    ext = Path(upload.name).suffix.lower()
    if ext in (".txt", ".md"):
        content = raw.decode("utf-8-sig")
    elif ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")
        content = "\n".join(page.extract_text() or "" for page in reader.pages[:100])
    else:
        raise ValueError("Only TXT, MD and PDF documents are supported.")
    if not content.strip():
        raise ValueError("No extractable text found in this document.")
    return Document(doc_id=str(uuid.uuid4()), title=upload.name, content=content[:300_000], source="Uploaded document")


if "documents" not in st.session_state:
    st.session_state.documents = sample_documents()
if "revision" not in st.session_state:
    st.session_state.revision = 0

@st.cache_resource(show_spinner=False)
def cached_engine(documents_tuple, revision):
    return SearchEngine(Document(*row) for row in documents_tuple)


def get_engine():
    return cached_engine(tuple((d.doc_id, d.title, d.content, d.source) for d in st.session_state.documents), st.session_state.revision)


st.title("🔎 IndexLab Search Engine")
st.caption("Inverted index · TF–IDF · Cosine similarity · Hash-based lookup · Heap top-K ranking")
st.write("Explore a working search engine built from scratch in Python. Search sample documents or add your own files.")

with st.sidebar:
    st.header("Document collection")
    uploads = st.file_uploader("Add documents", type=["txt", "md", "pdf"], accept_multiple_files=True,
                               help="TXT, Markdown and text-based PDFs, up to 5 MB each")
    if st.button("Index uploaded files", disabled=not uploads, use_container_width=True):
        added = 0
        for upload in uploads or []:
            try:
                doc = read_upload(upload)
                st.session_state.documents.append(doc)
                added += 1
            except Exception as exc:
                st.error(f"{upload.name}: {exc}")
        if added:
            st.session_state.revision += 1
            st.success(f"Indexed {added} document(s).")
    if st.button("Reset to sample collection", use_container_width=True):
        st.session_state.documents = sample_documents()
        st.session_state.revision += 1
        st.rerun()
    st.caption("Uploaded documents are held in your current Streamlit session, not stored permanently.")
    st.divider()
    st.subheader("How search works")
    st.markdown("1. Tokenize and normalize text\n2. Build term → document postings\n3. Weight terms with TF–IDF\n4. Compute cosine similarity\n5. Select top-K using a heap")

engine = get_engine()
a, b, c = st.columns(3)
a.metric("Documents", len(engine.documents))
b.metric("Unique indexed terms", engine.vocabulary_size)
c.metric("Total indexed tokens", sum(engine.doc_lengths.values()))

query = st.text_input("Search your documents", placeholder="e.g. inverted index python priority queue")
col1, col2 = st.columns([2, 1])
with col1:
    mode = st.radio("Keyword matching", ["OR", "AND"], horizontal=True,
                    help="OR: match any query term; AND: require all known query terms")
with col2:
    top_k = st.slider("Number of results", 1, 20, 5)

if query.strip():
    results = engine.search(query, top_k=top_k, mode=mode)
    st.subheader(f"{len(results)} result(s)")
    if not results:
        st.info("No matching documents. Try a different term or use OR mode.")
    for rank, hit in enumerate(results, 1):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{rank}. {hit.document.title}**")
            right.metric("Relevance", f"{hit.score:.3f}")
            st.write(hit.snippet)
            st.caption(f"Source: {hit.document.source}  ·  Matching terms: {', '.join(hit.matched_terms)}")
            with st.expander("Read full document"):
                st.text(hit.document.content[:15_000])
else:
    st.info("Enter keywords to search the document collection.")

with st.expander("Inspect the inverted index"):
    term = st.text_input("Enter a term", key="inspect_term")
    if term:
        postings = engine.postings.get(term.strip().lower(), {})
        if postings:
            st.write(f"Document frequency: {len(postings)} | IDF: {engine.idf[term.strip().lower()]:.3f}")
            st.dataframe([{"Document": engine.documents[d].title, "Term frequency": freq}
                          for d, freq in sorted(postings.items())], use_container_width=True, hide_index=True)
        else:
            st.write("Term not in index.")

st.caption("Implementation uses Python dictionaries (hash maps) and heapq.nlargest for top-K retrieval; rankings use log-scaled TF–IDF cosine similarity.")
