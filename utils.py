import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
import os

# ── Directory to store vector DBs ─────────────────────────────────────────────
VECTOR_DB_DIR = "vector_dbs"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


def extract_text(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS L2 index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve(question_embedding: np.ndarray, index: faiss.IndexFlatL2,
             chunks: list, pdf_names: list, top_k: int = 5):
    """Retrieve top_k most relevant chunks for a question embedding."""
    distances, indices = index.search(
        question_embedding.astype(np.float32), top_k
    )
    results = [(chunks[i], pdf_names[i]) for i in indices[0] if i < len(chunks)]
    scores = distances[0][:len(results)]
    return results, scores


def save_vector_db(index: faiss.IndexFlatL2, chunks: list, pdf_name: str):
    """Save FAISS index and chunks to disk."""
    safe_name = pdf_name.replace(" ", "_").replace("/", "_")
    faiss.write_index(index, os.path.join(VECTOR_DB_DIR, f"{safe_name}.index"))
    with open(os.path.join(VECTOR_DB_DIR, f"{safe_name}.chunks"), "wb") as f:
        pickle.dump(chunks, f)


def load_vector_db(pdf_name: str):
    """Load FAISS index and chunks from disk. Returns (None, None) if not found."""
    safe_name = pdf_name.replace(" ", "_").replace("/", "_")
    index_path  = os.path.join(VECTOR_DB_DIR, f"{safe_name}.index")
    chunks_path = os.path.join(VECTOR_DB_DIR, f"{safe_name}.chunks")
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None