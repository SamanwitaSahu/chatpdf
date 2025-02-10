import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_bytes
import fitz
import tempfile
import pdfplumber
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_bytes
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
import os

# Force OpenMP to allow duplicate libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Reduce OpenMP threading to prevent crashes
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import faiss
faiss.omp_set_num_threads(1)  # Reduce FAISS threads

# Set the Poppler path (Change this to your actual Poppler installation path)
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            doc = fitz.open(stream=pdf.read(), filetype="pdf")  # Load PDF safely
            for page in doc:
                text += page.get_text("text") + "\n"  # Extract text properly
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            continue
    return text

def extract_text_from_images(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_file:
                for page in pdf_file.pages:
                    # If page has text, extract it normally
                    if page.extract_text():
                        text += page.extract_text() + "\n"
                    else:
                        # Convert scanned page image to text using OCR
                        image = page.to_image()
                        text += pytesseract.image_to_string(image.original) + "\n"
        except Exception as e:
            st.error(f"Error extracting text from images: {e}")
    return text

# Load Donut OCR Model
# processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
# model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# def extract_text_from_images(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             images = convert_from_bytes(pdf.read(), poppler_path=POPPLER_PATH)
#             for img in images:
#                 inputs = processor(images=img, return_tensors="pt")
#                 outputs = model.generate(**inputs)
#                 extracted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
#                 text += extracted_text + "\n"
#         except Exception as e:
#             st.error(f"Error extracting text from images: {e}")
#     return text

def extract_tables_from_pdf(pdf_docs):
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

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "answer is not available in the context."

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: \n" + str(response["output_text"]))

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                ocr_text = extract_text_from_images(pdf_docs)
                table_text = extract_tables_from_pdf(pdf_docs)
                combined_text = raw_text + "\n" + ocr_text + "\n" + table_text
                text_chunks = get_text_chunks(combined_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
