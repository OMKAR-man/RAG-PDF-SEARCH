# Import Libraries
import streamlit as st
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from utils import extract_text, chunk_text, build_index, retrieve, save_vector_db, load_vector_db


# Streamlit Title
st.title("📄 Multi-PDF RAG Chatbot")


# Load Embedding Model
@st.cache_resource
def load_embedding():
    return SentenceTransformer("intfloat/multilingual-e5-small")


# Load HuggingFace LLM Model (no API key needed)
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=512
    )


embedding_model = load_embedding()
llm = load_llm()


# Upload PDFs
uploaded_pdfs = st.file_uploader(
    "Upload multiple PDFs",
    type="pdf",
    accept_multiple_files=True
)


# Session Memory Initialization
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = {}

if "pdf_indexes" not in st.session_state:
    st.session_state.pdf_indexes = {}


# Process PDFs
if uploaded_pdfs:

    for pdf in uploaded_pdfs:

        # Skip if already processed
        if pdf.name in st.session_state.pdf_chunks:
            print(pdf.name, "already stored in session")
            continue

        print("Checking vector DB for:", pdf.name)

        index, chunks = load_vector_db(pdf.name)

        # If vector DB exists → load
        if index is not None:

            print(pdf.name, "vector DB loaded")

            st.session_state.pdf_chunks[pdf.name] = chunks
            st.session_state.pdf_indexes[pdf.name] = index

        # If vector DB does NOT exist
        else:

            print("extract_text running")

            text = extract_text(pdf)

            print("chunk_text running")

            chunks = chunk_text(text)

            print(chunks, "chunks created")

            print("embedding creation running")

            embeddings = embedding_model.encode(chunks)

            print("build_index running")

            index = build_index(np.array(embeddings))

            print(index, "vector index built")

            save_vector_db(index, chunks, pdf.name)

            print(pdf.name, "vector DB saved")

            st.session_state.pdf_chunks[pdf.name] = chunks
            st.session_state.pdf_indexes[pdf.name] = index


# Access stored data
pdf_chunks = st.session_state.pdf_chunks
pdf_indexes = st.session_state.pdf_indexes


# Layout: Chatbot | PDF Selector
if pdf_chunks:

    col1, col2 = st.columns([2, 1])

    # Right Panel → Select PDF
    with col2:

        st.subheader("📂 Select PDF")

        selected_pdf = st.selectbox(
            "Choose a document",
            list(pdf_chunks.keys())
        )

        st.write(f"Selected PDF: **{selected_pdf}**")


    # Left Panel → Chatbot
    with col1:

        question = st.text_input(
            "Ask a question from the selected PDF"
        )


        greetings = [
            "hi",
            "hello",
            "hey",
            "namaste",
            "good morning",
            "good evening",
            "नमस्कार"
        ]


        if question:

            if question.lower() in greetings:

                st.subheader("Answer")

                st.write(
                    "Hello 👋! I am your AI PDF assistant. "
                    "Select a document and ask me questions."
                )

            else:

                chunks = pdf_chunks[selected_pdf]
                index = pdf_indexes[selected_pdf]

                print("encoding question")

                question_embedding = embedding_model.encode([question])

                print("retrieval running")

                retrieved, scores = retrieve(
                    question_embedding,
                    index,
                    chunks,
                    [selected_pdf] * len(chunks)
                )

                print(retrieved, "retrieved")

                context = ""

                for chunk, name in retrieved:
                    context += chunk + "\n"


                prompt = f"""Answer the question using ONLY the context below.
If the answer is not found in the context, say: This information is not available in the uploaded document.

Context:
{context}

Question:
{question}

Answer:"""


                print("sending prompt to HuggingFace model")

                response = llm(prompt)

                answer = response[0]["generated_text"]


                st.subheader("Answer")
                st.write(answer)


                st.subheader("Source Information")

                for i, (chunk, pdf_name) in enumerate(retrieved):

                    similarity = round(1 / (1 + scores[i]), 2)

                    st.write(f"📄 PDF: **{pdf_name}**")
                    st.write(f"🔎 Accuracy Score: **{similarity}**")
                    st.write(f"📌 Text: {chunk}")
                    st.write("---")