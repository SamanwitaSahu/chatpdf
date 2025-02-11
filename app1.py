import streamlit as st
import os
import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
from paddleocr import PaddleOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Gemini Embeddings (Can be replaced)
from langchain.embeddings import HuggingFaceEmbeddings  # Alternative Embeddings
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize PaddleOCR (Faster and More Accurate than Tesseract)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Set FAISS threading optimization
import faiss
faiss.omp_set_num_threads(1)  # Reduce FAISS threads

# PDF Processing Functions
def get_pdf_text(pdf_docs):
    """Extracts text from PDFs (text-based)."""
    text = ""
    for pdf in pdf_docs:
        try:
            doc = fitz.open(stream=pdf.read(), filetype="pdf")
            for page in doc:
                text += page.get_text("text") + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            continue
    return text

def extract_text_from_images(pdf_docs):
    """Extracts text from scanned PDFs using OCR (PaddleOCR)."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.read()
            if not pdf_bytes:
                st.error("Uploaded PDF ....")
                continue  
            
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

def extract_tables_from_pdf(pdf_docs):
    """Extracts tables from PDFs using pdfplumber."""
    tables = []
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_file:
                for page in pdf_file.pages:
                    extracted_tables = page.extract_tables()
                    for table in extracted_tables:
                        tables.append("\n".join([" | ".join(map(str, row)) for row in table]))
        except Exception as e:
            st.error(f"Error extracting tables: {e}")
    return "\n".join(tables)

# Text Processing & FAISS Indexing
def get_text_chunks(text):
    """Splits extracted text into small chunks for FAISS indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)  # Optimized for large PDFs
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Stores text chunks in FAISS using efficient embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    
    batch_size = 100
    vector_store = FAISS.from_texts(text_chunks[:batch_size], embedding=embeddings)
    
    for i in range(batch_size, len(text_chunks), batch_size):
        sub_vector_store = FAISS.from_texts(text_chunks[i:i+batch_size], embedding=embeddings)
        vector_store.merge_from(sub_vector_store)

    vector_store.save_local("faiss_index")

# Groq AI Integration
def ask_groq(question, context):
    """Uses Groq AI to generate answers based on retrieved document context."""
    response = groq_client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",  # Alternative: "llama3-70b" or "mistral-7b"
        messages=[
            {"role": "system", "content": "You are an AI assistant answering based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=4096,  # Corrected parameter
        top_p=0.95
        
    )
    return response.choices[0].message.content

def retrieve_context(user_question):
    """Retrieves the most relevant document chunks using FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Retrieve top 5 most relevant chunks
    docs = new_db.similarity_search(user_question, k=5)
    return "\n".join([doc.page_content for doc in docs])

def user_input(user_question):
    """Handles user queries by retrieving context and sending it to Groq AI."""
    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("FAISS index not found! Rebuilding...")

        if "pdf_docs" not in st.session_state or not st.session_state["pdf_docs"]:
            st.error("No PDFs uploaded! Please upload a file before asking questions.")
            return

        # Rebuild FAISS index
        raw_text = get_pdf_text(st.session_state["pdf_docs"])
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    context = retrieve_context(user_question)  

    if not context.strip():  
        st.write("Reply: \n No relevant information found in the document.")
        return

    response = ask_groq(user_question, context)  

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
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])

        if pdf_docs:  
            st.session_state["pdf_docs"] = pdf_docs  

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                ocr_text = extract_text_from_images(pdf_docs)
                table_text = extract_tables_from_pdf(pdf_docs)
                combined_text = raw_text + "\n" + ocr_text + "\n" + table_text
                text_chunks = get_text_chunks(combined_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete!")

if __name__ == "__main__":
    main()
