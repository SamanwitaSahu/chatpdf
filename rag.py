import streamlit as st
import os
import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
from paddleocr import PaddleOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from groq import Groq
import concurrent.futures  # For parallel processing
import faiss

# Load environment variables
load_dotenv()

# Initialize Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize PaddleOCR (Faster and More Accurate than Tesseract)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Set FAISS threading optimization
faiss.omp_set_num_threads(1)  # Reduce FAISS threads

# PDF Processing Functions
def get_pdf_text(pdf):
    """Extracts text from a single PDF."""
    try:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        return "\n".join([page.get_text("text") for page in doc if page.get_text("text")])
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_images(pdf):
    """Extracts text from scanned PDFs using OCR (PaddleOCR)."""
    text = ""
    try:
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
                else:
                    image = page.to_image()
                    pil_image = image.original
                    extracted_text = ocr.ocr(pil_image, cls=True)
                    text += " ".join([line[1][0] for result in extracted_text for line in result]) + "\n"
    except Exception as e:
        st.error(f"Error extracting text from images: {e}")
    return text

def process_pdfs_parallel(pdf_docs):
    """Processes multiple PDFs in parallel to improve speed."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        text_results = list(executor.map(get_pdf_text, pdf_docs))
        ocr_results = list(executor.map(extract_text_from_images, pdf_docs))

    combined_text = "\n".join(text_results) + "\n" + "\n".join(ocr_results)
    return combined_text

# Text Processing & FAISS Indexing
def get_text_chunks(text):
    """Splits extracted text into fewer but larger chunks for FAISS indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, embeddings):
    """Store text in FAISS and generate embeddings at query time."""
    if not text_chunks:
        st.error("No text extracted. FAISS index not created.")
        return

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Store embeddings
    vector_store.save_local("faiss_index")

# Groq AI Integration
def ask_groq(question, context):
    """Uses Groq AI to generate fast answers based on retrieved context."""
    response = groq_client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",  # Optimized model for fast response
        messages=[
            {"role": "system", "content": "You are an AI assistant answering based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        temperature=0.5,
        max_tokens=1024,  # Reduced max tokens for faster response
        top_p=0.9
    )
    return response.choices[0].message.content

def retrieve_context(user_question):
    """Retrieves the most relevant document chunks using FAISS at query time."""
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small", model_kwargs={"device": "cpu"})  # Fastest model
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Compute embeddings only for the query
    query_embedding = embeddings.embed_query(user_question)
    
    docs = new_db.similarity_search_by_vector(query_embedding, k=3)  # Retrieve top 3 chunks
    return "\n".join([doc.page_content for doc in docs])

def user_input(user_question):
    """Handles user queries by first retrieving relevant context."""
    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("FAISS index not found! Upload a document first.")

        if "pdf_docs" not in st.session_state or not st.session_state["pdf_docs"]:
            st.error("No PDFs uploaded! Please upload a file before asking questions.")
            return

        raw_text = process_pdfs_parallel(st.session_state["pdf_docs"])
        text_chunks = get_text_chunks(raw_text)
        
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small", model_kwargs={"device": "cpu"})  # Ensure embeddings exist
        get_vector_store(text_chunks, embeddings)

    context = retrieve_context(user_question)  # Query-first retrieval

    if not context.strip():
        st.write("Reply: \n No relevant information found in the document.")
        return

    response = ask_groq(user_question, context)  # Only send retrieved chunks to LLM
    st.write("Reply: \n" + str(response))

# Streamlit UI
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
            with st.spinner("Processing..."):
                raw_text = process_pdfs_parallel(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small", model_kwargs={"device": "cpu"})  # Ensure embeddings exist
                get_vector_store(text_chunks, embeddings)

                st.success("Processing Complete!")

if __name__ == "__main__":
    main()
