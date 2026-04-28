"""
Personal PDF Q&A — A beginner-friendly RAG (Retrieval-Augmented Generation) app.

## How this works (the RAG flow in 8 plain-English steps)
1. INDEX TIME — User uploads a PDF in the sidebar.
2. INDEX TIME — We extract the text from each page, keeping the page number.
3. INDEX TIME — Each page's text is split into small overlapping "chunks"
   of ~500 characters (a paragraph or two each).
4. INDEX TIME — Each chunk is converted into a 384-number vector ("embedding")
   using a small local model. Chunks with similar meaning end up close
   together in this 384-dimensional space.
5. INDEX TIME — Chunks + embeddings + page numbers are stored in ChromaDB
   (an in-memory vector database).
6. QUERY TIME — User types a question in the chat box.
7. QUERY TIME — We embed the question using the same model, then ask
   ChromaDB for the 4 chunks whose embeddings are most similar (cosine
   similarity) to the question's embedding.
8. QUERY TIME — Those 4 chunks become "context" inside a prompt sent to
   Gemini. The prompt forces Gemini to answer using ONLY that context,
   so the answer is grounded in the actual document.

The whole app is one file. Read it top-to-bottom to follow the flow.
"""

import os
from typing import Optional

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv

# Load GOOGLE_API_KEY from a local .env file when running outside Codespaces.
# In Codespaces, the key comes from a Codespaces "secret" (same env var name),
# which is automatically present in os.environ — load_dotenv() is a no-op then.
load_dotenv()

# ---------- Streamlit page setup ----------
st.set_page_config(page_title="Personal PDF Q&A", page_icon="📄")
st.title("📄 Personal PDF Q&A")
st.caption("Upload a PDF and ask questions. Answers cite the page they came from.")

# ---------- Configuration constants ----------
CHUNK_SIZE = 500           # characters per chunk
CHUNK_OVERLAP = 50         # characters of overlap between consecutive chunks
TOP_K = 4                  # number of chunks retrieved per question
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim local embedding model
GEMINI_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "pdf_chunks"

# INTERVIEW NOTE: Why CHUNK_SIZE=500 with CHUNK_OVERLAP=50?
# 500 chars (~80 words / ~120 tokens) is a sweet spot:
#   - Big enough to contain a complete thought (a sentence or two).
#   - Small enough that retrieval is precise — we get the part of the
#     document that actually answers the question, not a whole page of
#     unrelated text that confuses the LLM.
# If we picked 100: chunks would be too small. They'd lose context
#   ("It raises revenue by 20%" — what is "it"?), and we'd need many
#   more chunks to cover the same ground, which slows retrieval and
#   means each retrieved chunk carries less information.
# If we picked 2000: chunks would be too big. A single chunk might
#   cover 4 different topics, so retrieving it gives the LLM lots of
#   irrelevant text to wade through, hurting answer quality and
#   wasting tokens. Embedding quality also degrades because the
#   embedding has to "average" too many ideas into one vector.
# The 50-char overlap (~10% of chunk size) means consecutive chunks
# share a small tail/head, so an idea that straddles a chunk boundary
# isn't split in half and lost from retrieval.

# ---------- Cached resources (created ONCE, reused across reruns) ----------

@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    """Load the local embedding model. First call downloads ~90 MB; later calls are instant."""
    # INTERVIEW NOTE: Why a LOCAL embedding model instead of Gemini's embedding API?
    # 1. RATE LIMITS: Gemini free tier is ~15 req/min. A 50-page PDF can have
    #    hundreds of chunks — embedding each one via the API would burn through
    #    the quota immediately.
    # 2. SPEED + COST: Local model has zero network latency and is free. For
    #    bulk operations (indexing) this is a huge win.
    # 3. SEPARATION OF CONCERNS: Embedding (geometry of meaning) and generation
    #    (reasoning in natural language) are different jobs. Using a small
    #    specialized model for embeddings and saving the big LLM only for
    #    answer generation is the standard production pattern.
    # all-MiniLM-L6-v2 produces 384-dim vectors and is good enough for most
    # general-English semantic search.
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def get_chroma_client():
    """In-memory ChromaDB client. Data lives only while the app process runs."""
    return chromadb.Client()


@st.cache_resource
def get_gemini_client() -> genai.Client:
    """Create the Gemini client using the GOOGLE_API_KEY env var."""
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


# ---------- PDF parsing ----------
def extract_pages(pdf_file) -> list[tuple[int, str]]:
    """
    Read a PDF and return a list of (page_number, page_text) tuples.
    Page numbers are 1-indexed (the way humans count).
    """
    reader = PdfReader(pdf_file)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        # extract_text() can return None for image-only / scanned pages.
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages


# ---------- Chunking (a simple "recursive character splitter") ----------
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size characters.
    We try to break at natural boundaries (paragraph > sentence > newline >
    space) rather than mid-word. This is the core idea of LangChain's
    RecursiveCharacterTextSplitter, written from scratch so you can read it.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # If we're not at the end of the document, step back to a natural
        # break point so we don't slice through a sentence.
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

        # Move forward, but back up by `overlap` chars so consecutive chunks
        # share a small tail/head — preserves context across boundaries.
        start = max(end - overlap, start + 1)

    return chunks


# ---------- Indexing pipeline: PDF → chunks → embeddings → ChromaDB ----------
def index_pdf(pdf_file) -> Optional["chromadb.Collection"]:
    """Run the full indexing pipeline and return the populated Chroma collection."""
    pages = extract_pages(pdf_file)
    embedder = get_embedding_model()
    client = get_chroma_client()

    # Wipe any previous collection so a new upload doesn't mix with old data.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # INTERVIEW NOTE: Why COSINE similarity for retrieval?
    # Embeddings encode "meaning" as a *direction* in high-dim space.
    # Cosine similarity measures the angle between two vectors and ignores
    # their magnitude — so two sentences that mean the same thing but have
    # different lengths still score as very similar. Euclidean (L2) distance
    # would penalize length differences, which is the wrong thing for
    # semantic search. ChromaDB defaults to L2, so we explicitly request
    # cosine here.
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
        return None  # PDF had no extractable text (probably scanned images)

    # INTERVIEW NOTE: Why embed CHUNKS, not the whole document?
    # 1. Embedding models have a max input length (~256–512 tokens for
    #    MiniLM). A whole document literally won't fit.
    # 2. A single embedding for an entire 30-page PDF would have to "average"
    #    over many topics, losing specificity — every query would look
    #    "kind of similar" to it, defeating the point of retrieval.
    # 3. We want to retrieve THE relevant paragraph, not the whole document.

    # Encode all chunks in one batch — much faster than one-by-one because
    # the model can use vectorized matrix math.
    embeddings = embedder.encode(all_chunks, show_progress_bar=False).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )
    return collection


# ---------- Retrieval ----------
def retrieve(
    question: str,
    collection: "chromadb.Collection",
    top_k: int = TOP_K,
) -> tuple[list[str], list[int]]:
    """Embed the question and return (top-k chunks, their page numbers)."""
    embedder = get_embedding_model()
    query_embedding = embedder.encode([question]).tolist()

    # INTERVIEW NOTE: Why top_k=4?
    # This is the classic precision/recall tradeoff in retrieval:
    #   - Lower k (1–2): high precision but you may MISS the right chunk if
    #     your embedding model ranks it 3rd. The model can't answer even
    #     though the answer is in the document.
    #   - Higher k (10–20): high recall but you stuff the prompt with
    #     marginally-relevant chunks, which (a) costs more tokens and
    #     (b) can confuse the LLM with irrelevant context.
    # k=4 typically captures the right answer for a focused question while
    # keeping the prompt small. For a multi-part question
    # ("compare X to Y") you might bump it to 6–8.
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    chunks: list[str] = results["documents"][0]
    pages: list[int] = [m["page"] for m in results["metadatas"][0]]
    return chunks, pages


# ---------- Prompt construction (THIS is where "RAG" actually happens) ----------
def build_prompt(question: str, chunks: list[str], pages: list[int]) -> str:
    """Assemble the final prompt: retrieved chunks become 'context'."""
    # INTERVIEW NOTE: ⭐ THIS IS THE EXACT MOMENT WHERE "RAG" HAPPENS. ⭐
    # "Retrieval-Augmented Generation" = we *Augment* the *Generation*
    # prompt with text we *Retrieved* from our knowledge base. Without
    # this step, the model would have to answer from its training data
    # alone — and it has never seen the user's PDF. By pasting the
    # relevant chunks into the prompt, we give the LLM the facts it
    # needs to answer correctly, even though the model itself was
    # never trained on this document.
    context_blocks = [
        f"[Page {page}]\n{chunk}"
        for chunk, page in zip(chunks, pages)
    ]
    context = "\n\n---\n\n".join(context_blocks)

    # INTERVIEW NOTE: Why instruct Gemini to use ONLY the provided context?
    # Without this constraint, the model freely mixes its training
    # knowledge with the retrieved context, causing two failures:
    #   1. HALLUCINATION: It invents plausible-sounding facts that are
    #      not actually in the document.
    #   2. WRONG ATTRIBUTION: It answers from general knowledge but the
    #      user thinks the answer came from their PDF — destroying the
    #      whole point of a Q&A-over-docs app.
    # Saying "use ONLY the context" + "say I don't know if it's missing"
    # forces the model to ground its answer in the retrieved evidence
    # or admit ignorance. This is called "grounding".
    return f"""You are a helpful assistant answering questions about a specific PDF document.

Use ONLY the context below to answer the question.
If the answer is not contained in the context, reply exactly:
"I cannot find this information in the document."
Do not use outside knowledge. Do not guess.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ---------- Generation ----------
def ask_gemini(prompt: str) -> str:
    """Call Gemini with the prompt; return the answer or a friendly error."""
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text or "(Gemini returned an empty response.)"
    except Exception as e:
        # The free tier is rate-limited at ~15 req/min — most failures
        # users see are 429s after rapid-fire questions.
        return (
            "⚠️ Sorry, the Gemini API call failed. "
            "If you've been asking lots of questions quickly, the free tier "
            "rate-limit (~15/min) may have kicked in — wait ~60s and try again.\n\n"
            f"Technical detail: `{type(e).__name__}: {e}`"
        )


# ---------- Streamlit UI ----------

# Sidebar: upload + status
with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded_file = st.file_uploader(
        "PDF file", type="pdf", label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Re-index only when the uploaded file changes. Streamlit re-runs the
        # whole script on every interaction, so we cache by filename in
        # session_state to avoid re-embedding the same PDF over and over.
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
                    st.session_state.messages = []  # reset chat on new PDF
        st.success(f"Indexed: {uploaded_file.name}")

    st.markdown("---")
    st.header("2. Ask questions")
    st.markdown("Use the chat box on the right →")
    st.markdown("---")
    st.caption(
        f"Chunk size: {CHUNK_SIZE} chars · Overlap: {CHUNK_OVERLAP} · "
        f"Top-k: {TOP_K} · Embed: `{EMBED_MODEL_NAME}` · LLM: `{GEMINI_MODEL}`"
    )

# Main area: chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages on every rerun (Streamlit's chat is stateless by default)
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

        # Retrieve + Generate (the two halves of "RAG")
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
