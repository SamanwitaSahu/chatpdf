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
import math

# Load environment variables
load_dotenv()

# Initialize Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize PaddleOCR (Faster and More Accurate than Tesseract)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Set FAISS threading optimization
faiss.omp_set_num_threads(1)  # Reduce FAISS threads

# Split PDF Pages into Chunks for Parallel Processing
def split_pdf_pages(pdf, num_splits=5):
    """Splits a PDF into 'num_splits' smaller batches for parallel processing."""
    try:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        total_pages = len(doc)
        pages_per_split = math.ceil(total_pages / num_splits)
        
        return [doc[i : i + pages_per_split] for i in range(0, total_pages, pages_per_split)]
    except Exception as e:
        st.error(f"Error splitting PDF: {e}")
        return []

# Extract Text from PDF Pages
def extract_text_from_pdf_pages(pages):
    """Extracts text from a batch of PDF pages."""
    return "\n".join([page.get_text("text") for page in pages if page.get_text("text")])

# Extract Text from Images using OCR
def extract_text_from_images(pages):
    """Extracts text from scanned pages using OCR."""
    text = ""
    for page in pages:
        if not page.get_text("text"):  # Only use OCR if no selectable text
            image = page.get_pixmap()
            extracted_text = ocr.ocr(image, cls=True)
            text += " ".join([line[1][0] for result in extracted_text for line in result]) + "\n"
    return text

# Extract Tables from PDF
def extract_tables_from_pdf(pages):
    """Extracts tables from a batch of pages using pdfplumber."""
    tables = []
    for page in pages:
        try:
            with pdfplumber.open(page.parent) as pdf_file:  # Open PDF again (pdfplumber requires full PDF)
                extracted_tables = pdf_file.pages[page.number].extract_tables()  # Extract table from this page only
                for table in extracted_tables:
                    tables.append("\n".join([" | ".join(map(str, row)) for row in table]))
        except Exception as e:
            st.error(f"Error extracting tables: {e}")
    return "\n".join(tables)


import asyncio

async def process_pdfs_parallel(pdf_docs, num_splits=5):
    """Processes PDFs in parallel using asyncio for compatibility with Streamlit."""
    combined_text = ""

    for pdf in pdf_docs:
        page_batches = split_pdf_pages(pdf, num_splits)

        loop = asyncio.get_running_loop()

        # Run extraction functions in parallel using run_in_executor
        text_results = await asyncio.gather(
            *[loop.run_in_executor(None, extract_text_from_pdf_pages, batch) for batch in page_batches]
        )

        ocr_results = await asyncio.gather(
            *[loop.run_in_executor(None, extract_text_from_images, batch) for batch in page_batches]
        )

        table_results = await asyncio.gather(
            *[loop.run_in_executor(None, extract_tables_from_pdf, batch) for batch in page_batches]
        )

        combined_text += "\n".join(text_results) + "\n" + "\n".join(ocr_results) + "\n" + "\n".join(table_results)

    return combined_text




# **NEW: Define Missing Function get_text_chunks()**
def get_text_chunks(text):
    """Splits extracted text into chunks for FAISS indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)  # Adjusted for speed
    return text_splitter.split_text(text)

# **NEW: Optimized FAISS Vector Storage**
def get_vector_store(text_chunks):
    """Stores text chunks in FAISS using batch processing."""
    if not text_chunks:
        st.error("No text extracted. FAISS index not created.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small", model_kwargs={"device": "cpu"})
    
    batch_size = 300
    vector_store = FAISS.from_texts(text_chunks[:batch_size], embedding=embeddings)

    def process_batch(start_idx):
        return FAISS.from_texts(text_chunks[start_idx:start_idx + batch_size], embedding=embeddings)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        sub_vector_stores = list(executor.map(process_batch, range(batch_size, len(text_chunks), batch_size)))

    for sub_vector_store in sub_vector_stores:
        vector_store.merge_from(sub_vector_store)

    vector_store.save_local("faiss_index")

# **Groq AI Query**
def ask_groq(question, context):
    """Uses Groq AI to generate fast answers based on retrieved context."""
    response = groq_client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b",
        messages=[
            {"role": "system", "content": "You are an AI assistant answering based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=1024,
        top_p=0.9
    )
    return response.choices[0].message.content

# **NEW: Retrieve Context from FAISS**
def retrieve_context(user_question):
    """Retrieves the most relevant document chunks using FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small", model_kwargs={"device": "cpu"})
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question, k=3)  # Reduced to top 3 results for speed
    return "\n".join([doc.page_content for doc in docs])

# **Handle User Query**
def user_input(user_question):
    """Handles user queries by retrieving context and sending it to Groq AI."""
    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("FAISS index not found! Rebuilding...")

        if "pdf_docs" not in st.session_state or not st.session_state["pdf_docs"]:
            st.error("No PDFs uploaded! Please upload a file before asking questions.")
            return

        raw_text = process_pdfs_parallel(st.session_state["pdf_docs"], num_splits=5)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    context = retrieve_context(user_question)

    if not context.strip():
        st.write("Reply: \n No relevant information found in the document.")
        return

    response = ask_groq(user_question, context)
    st.write("Reply: \n" + str(response))

import asyncio
import time

def main():
    st.set_page_config(page_title="Chat PDF with Groq AI 💡")
    st.header("Chat with PDF using Groq AI 💡")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

        if pdf_docs:
            st.session_state["pdf_docs"] = pdf_docs  

        if st.button("Submit & Process"):
            st.write("🕒 **Processing started...**")
            start_time = time.time()  # ✅ Start Timer
            with st.spinner("Processing..."):
                loop = asyncio.new_event_loop()  # ✅ Manually create an event loop
                asyncio.set_event_loop(loop)

                # Run process_pdfs_parallel() asynchronously
                raw_text = loop.run_until_complete(process_pdfs_parallel(pdf_docs, num_splits=5))

                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            end_time = time.time()  # ✅ End Timer
            elapsed_time = end_time - start_time  # ✅ Calculate Time Taken

            st.success(f"✅ Processing Complete! 🕒 Time taken: **{elapsed_time:.2f} seconds**")


if __name__ == "__main__":
    main()
