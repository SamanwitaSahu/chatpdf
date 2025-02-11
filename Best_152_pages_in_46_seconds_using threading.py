import streamlit as st
import os
import fitz  # PyMuPDF
import pdfplumber
from paddleocr import PaddleOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from groq import Groq
import concurrent.futures  # For parallel processing
import faiss
import time
import gc

# Load environment variables
load_dotenv()

# Initialize Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize PaddleOCR (Faster than Tesseract)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Get Number of CPU Cores
NUM_CORES = os.cpu_count()  # Use all available cores (defaults to 4 if not detected)

# Set FAISS to Use All Cores
faiss.omp_set_num_threads(NUM_CORES)

# Increase Thread Count
MAX_THREADS = NUM_CORES*5  # Uses 3x CPU cores but limits to 15

def get_text_chunks(text):
    """Splits extracted text into chunks for FAISS indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)  # Larger chunks for speed
    return text_splitter.split_text(text)

# Split PDF Pages into Chunks for Parallel Processing
def split_pdf_pages(pdf, num_splits=10):
    """Splits a PDF into 'num_splits' smaller batches for parallel processing."""
    try:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        total_pages = len(doc)
        pages_per_split = max(1, total_pages // num_splits)  # Ensure at least 1 page per split

        return [doc[i : i + pages_per_split] for i in range(0, total_pages, pages_per_split)]
    except Exception as e:
        st.error(f"Error splitting PDF: {e}")
        return []

# Extract Text from PDF Pages
def extract_text_from_pdf_pages(pages):
    """Extracts text from a batch of PDF pages."""
    return "\n".join([page.get_text("text") for page in pages if page.get_text("text")])

# **Optimized OCR (Skip OCR if Text is Present)**
def extract_text_from_images(pages):
    """Extracts text from scanned pages using OCR but skips pages that already contain text."""
    text = ""
    for page in pages:
        if not page.get_text("text"):  # Only OCR if no selectable text
            image = page.get_pixmap()
            extracted_text = ocr.ocr(image, cls=True)
            text += " ".join([line[1][0] for result in extracted_text for line in result]) + "\n"
    return text

# Extract Tables from PDF
def extract_tables_from_pdf(pdf):
    """Extracts tables from a PDF using pdfplumber."""
    tables = []
    try:
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    tables.append("\n".join([" | ".join(map(str, row)) for row in table]))
    except Exception as e:
        st.error(f"Error extracting tables: {e}")
    return "\n".join(tables)

# **🚀 Optimized Parallel Processing of PDFs**
def process_pdfs_parallel(pdf_docs, num_splits=5):
    """Processes PDFs in parallel, splitting into 'num_splits' chunks."""
    combined_text = ""

    for pdf in pdf_docs:
        page_batches = split_pdf_pages(pdf, num_splits)

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            text_results = executor.map(extract_text_from_pdf_pages, page_batches)
            ocr_results = executor.map(extract_text_from_images, page_batches)
            

        # **Run Table Extraction in Parallel**
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     table_results = list(executor.map(extract_tables_from_pdf, [pdf] * num_splits))
        table_results = list()
        combined_text += "\n".join(text_results) + "\n" + "\n".join(ocr_results) + "\n" + "\n".join(table_results)

    return combined_text

# **Optimized FAISS Indexing (Parallel Processing)**
def get_vector_store(text_chunks):
    """Stores text chunks in FAISS using batch processing."""
    if not text_chunks:
        st.error("No text extracted. FAISS index not created.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small", model_kwargs={"device": "cpu"})

    batch_size = 8000  # 🚀 Increased batch size for speed
    vector_store = FAISS.from_texts(text_chunks[:batch_size], embedding=embeddings)

    def process_batch(start_idx):
        return FAISS.from_texts(text_chunks[start_idx:start_idx + batch_size], embedding=embeddings)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        sub_vector_stores = list(executor.map(process_batch, range(batch_size, len(text_chunks), batch_size)))

    for sub_vector_store in sub_vector_stores:
        vector_store.merge_from(sub_vector_store)

    vector_store.save_local("faiss_index")

# **🚀 Optimized Query Handling**
def retrieve_context(user_question):
    """Retrieves the most relevant document chunks using FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small", model_kwargs={"device": "cpu"})
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question, k=3)  # Increased to 5 for better accuracy
     # ✅ **Truncate context to fit within 4000 tokens**
    full_context = "\n".join([doc.page_content for doc in docs])
    if len(full_context.split()) > 6000:  # Ensure it doesn't exceed the Groq limit
        full_context = " ".join(full_context.split()[:6000])
    return full_context

# **Groq AI Query**
def ask_groq(question, context):
    """Uses Groq AI to generate fast answers based on retrieved context."""
    response = groq_client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": "You are an AI assistant answering based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=4096,  # 🚀 Increased for better responses
        top_p=0.9
    )
    return response.choices[0].message.content

# **🚀 Optimized User Query Handling**
def user_input(user_question):
    """Handles user queries by retrieving context and sending it to Groq AI."""
    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("FAISS index not found! Rebuilding...")

        if "pdf_docs" not in st.session_state or not st.session_state["pdf_docs"]:
            st.error("No PDFs uploaded! Please upload a file before asking questions.")
            return

        raw_text = process_pdfs_parallel(st.session_state["pdf_docs"], num_splits=MAX_THREADS//3)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    context = retrieve_context(user_question)

    if not context.strip():
        st.write("❌ No relevant information found in the document.")
        return

    response = ask_groq(user_question, context)
    st.write("💡 **Reply:** \n" + str(response))

# **🚀 Streamlit UI**
def main():
    gc.collect()
    st.set_page_config(page_title="Chat PDF with Groq AI 💡")
    st.header("Chat with PDF using Groq AI 💡")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            st.write("🕒 **Processing started...**")
            start_time = time.time()

            with st.spinner("Processing..."):
                raw_text = process_pdfs_parallel(pdf_docs, num_splits=5)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

            st.success(f"✅ Processing Complete! 🕒 Time: {time.time() - start_time:.2f} sec")

if __name__ == "__main__":
    main()
