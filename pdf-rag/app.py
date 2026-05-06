import os
from typing import Optional

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Personal PDF Q&A", page_icon="📄")
st.title("📄 Personal PDF Q&A")
st.caption("Upload a PDF and ask questions. Answers cite the page they came from.")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "pdf_chunks"


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def get_chroma_client():
    return chromadb.Client()


@st.cache_resource
def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "GOOGLE_API_KEY is not set.\n\n"
            "In Codespaces: Settings → Codespaces → New secret → "
            "name=GOOGLE_API_KEY, then rebuild the codespace.\n"
            "Locally: copy .env.example to .env and paste your key."
        )
        st.stop()
    return genai.Client(api_key=api_key)


def extract_pages(pdf_file) -> list[tuple[int, str]]:
    reader = PdfReader(pdf_file)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            for separator in ["\n\n", ". ", "\n", " "]:
                hit = text.rfind(separator, start + chunk_size // 2, end)
                if hit != -1:
                    end = hit + len(separator)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def index_pdf(pdf_file) -> Optional["chromadb.Collection"]:
    pages = extract_pages(pdf_file)
    embedder = get_embedding_model()
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []

    for page_num, page_text in pages:
        for chunk_idx, chunk in enumerate(chunk_text(page_text)):
            all_chunks.append(chunk)
            all_metadatas.append({"page": page_num})
            all_ids.append(f"p{page_num}_c{chunk_idx}")

    if not all_chunks:
        return None

    embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )
    return collection


def retrieve(
    question: str,
    collection: "chromadb.Collection",
    top_k: int = TOP_K,
) -> tuple[list[str], list[int]]:
    embedder = get_embedding_model()
    query_embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    chunks: list[str] = results["documents"][0]
    pages: list[int] = [m["page"] for m in results["metadatas"][0]]
    return chunks, pages


def build_prompt(question: str, chunks: list[str], pages: list[int]) -> str:
    context_blocks = [
        f"[Page {page}]\n{chunk}"
        for chunk, page in zip(chunks, pages)
    ]
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a helpful assistant answering questions about a specific PDF document.

Use ONLY the context below to answer the question.
If the answer is not contained in the context, reply exactly:
"I cannot find this information in the document."
Do not use outside knowledge. Do not guess.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def ask_gemini(prompt: str) -> str:
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text or "(Gemini returned an empty response.)"
    except Exception as e:
        return (
            "⚠️ Sorry, the Gemini API call failed. "
            "If you've been asking lots of questions quickly, the free tier "
            "rate-limit (~15/min) may have kicked in — wait ~60s and try again.\n\n"
            f"Technical detail: `{type(e).__name__}: {e}`"
        )


with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded_file = st.file_uploader(
        "PDF file", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file is not None:
        already_indexed = (
            st.session_state.get("indexed_filename") == uploaded_file.name
        )
        if not already_indexed:
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                collection = index_pdf(uploaded_file)
                if collection is None:
                    st.error(
                        "Could not extract any text from this PDF. "
                        "Is it a scanned image PDF? (OCR not supported.)"
                    )
                else:
                    st.session_state.collection = collection
                    st.session_state.indexed_filename = uploaded_file.name
                    st.session_state.messages = []
        st.success(f"Indexed: {uploaded_file.name}")

    st.markdown("---")
    st.header("2. Ask questions")
    st.markdown("Use the chat box on the right →")
    st.markdown("---")
    st.caption(
        f"Chunk size: {CHUNK_SIZE} chars · Overlap: {CHUNK_OVERLAP} · "
        f"Top-k: {TOP_K} · Embed: `{EMBED_MODEL_NAME}` · LLM: `{GEMINI_MODEL}`"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption(f"📍 Sources: page(s) {', '.join(map(str, msg['pages']))}")

if "collection" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    question = st.chat_input("Ask a question about your PDF…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and asking Gemini…"):
                chunks, pages = retrieve(question, st.session_state.collection)
                prompt = build_prompt(question, chunks, pages)
                answer = ask_gemini(prompt)
            st.markdown(answer)
            unique_pages = sorted(set(pages))
            st.caption(
                f"📍 Sources: page(s) {', '.join(map(str, unique_pages))}"
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "pages": unique_pages}
        )
