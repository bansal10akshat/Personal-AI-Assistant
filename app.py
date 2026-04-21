import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ------------------ INIT ------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Load generator (only once)
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad"
)

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(file):
    text = ""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page_num, page in enumerate(doc):
            text += f"\n--- Page {page_num + 1} ---\n{page.get_text()}"
    except Exception:
        return None
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


def generate_answer(query, docs, threshold=0.3):
    if not docs:
        return "Not found in document."

    best_answer = ""
    best_score = 0.0

    for doc in docs:
        context = doc.page_content[:700]

        try:
            result = qa_pipeline(question=query, context=context)
            score = float(result.get("score", 0.0))
            answer = result.get("answer", "").strip()

            # ❌ Ignore useless answers
            if len(answer) < 3:
                continue

            if score > best_score:
                best_score = score
                best_answer = answer

        except:
            continue

    #Anti-hallucination guard
    if best_score < threshold:
        return "Not found in document."

    return best_answer


def classify_intent(query):
    q = query.lower()
    if "summary" in q:
        return "Summarization"
    elif "compare" in q:
        return "Comparison"
    elif "who" in q or "where" in q or "when" in q:
        return "Factual"
    return "General Query"


def export_chat(chat_history):
    md = ""
    for chat in chat_history:
        md += f"### Q: {chat['question']}\n"
        md += f"{chat['answer']}\n\n"
    return md


# ------------------ UI ------------------

st.set_page_config(page_title="AI Assistant", layout="wide")

st.title("📄 Personal AI Document Assistant")
st.write("Upload PDFs and ask questions based on them.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    key=f"file_upload_{st.session_state.uploader_key}"
)

# Query input (ALWAYS visible)
query = st.text_input("Ask a question", key="query_input")

# ------------------ FILE PROCESSING ------------------

if uploaded_files and st.session_state.vector_db is None:
    with st.spinner("Processing PDFs..."):
        all_text = ""

        for file in uploaded_files:
            extracted_text = extract_text_from_pdf(file)

            if extracted_text:
                all_text += extracted_text
            else:
                st.error(f"Failed to read {file.name}")

        chunks = chunk_text(all_text)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        vector_db = FAISS.from_texts(chunks, embeddings)
        st.session_state.vector_db = vector_db

    st.success("Documents processed successfully ✅")

# ------------------ QUERY HANDLING ------------------

if query:
    if st.session_state.vector_db is None:
        st.warning("⚠️ Please upload PDFs first.")
    else:
        docs = st.session_state.vector_db.similarity_search_with_score(query, k=5)

        # filter weak matches
        docs = [doc for doc, score in docs if score < 1.0]

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, docs)

        intent = classify_intent(query)

        # Show answer
        st.subheader("Answer")
        st.success(answer)

        # Show intent
        st.subheader("Intent")
        st.info(intent)

        # Show source
        st.subheader("Source (from document)")
        for doc in docs:
            st.write(doc.page_content[:300])

        # Save chat
        st.session_state.chat_history.append({
            "question": query,
            "answer": answer
        })

# ------------------ CHAT HISTORY ------------------

if st.session_state.chat_history:
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"Q: {chat['question']}")
        st.write(f"A: {chat['answer']}")

# ------------------ EXPORT ------------------

if st.session_state.chat_history:
    st.download_button(
        "Export Chat as Markdown",
        export_chat(st.session_state.chat_history),
        file_name="chat.md"
    )

# ------------------ CLEAR BUTTON ------------------

if st.button("Clear All Documents"):
    st.session_state.vector_db = None
    st.session_state.chat_history = []
    st.session_state.uploader_key += 1
    st.success("Cleared all data ✅")
    st.rerun()