# 📚 Chat with PDF using Gemini 💬  
🚀 **Still working on making it support 100+ page PDFs and asynchronous processing!**  
🛠️ **Better OCR methods are also being tested for improved accuracy.**

## 🌟 Features  
- 📖 **Extracts text from PDFs**  
  - Supports both **text-based** and **scanned PDFs**  
  - Uses **PyMuPDF (fitz)** for **accurate text extraction**  
- 🔍 **AI-powered Q&A**  
  - Uses **Google Gemini AI** to provide **detailed and context-aware answers**  
- 🖼️ **OCR for Scanned PDFs**  
  - Uses **Tesseract & pdfplumber** for OCR (Testing **Donut OCR for better accuracy**)  
- 📊 **Table Extraction**  
  - Extracts **tables** from PDFs using **pdfplumber**  
- ⚡ **FAISS Vector Store for Fast Search**  
  - Converts PDF content into **vector embeddings** for efficient retrieval  
- 🚀 **Optimized for Performance**  
  - Uses **chunked text splitting** to improve vector search speed  
- 🔥 **Work in Progress:**  
  - ✅ **Handling 100+ page PDFs efficiently**  
  - ✅ **Async PDF processing for better performance**  
  - ✅ **Testing improved OCR models for better accuracy**  

---

## 🛠️ **How the Project Works**
### **1️⃣ Extract Text from PDFs**
- **For text-based PDFs:**  
  - Uses **PyMuPDF (fitz)** to extract structured text accurately.  
- **For scanned PDFs (images inside PDFs):**  
  - Uses **pdfplumber & Tesseract OCR** to extract text from images.  
  - Testing **Donut OCR for better accuracy** in future versions.

### **2️⃣ Split Text into Chunks for Better Processing**
- Uses **LangChain's RecursiveCharacterTextSplitter**  
- Splits text into **small manageable chunks** (~10,000 characters)  
- Helps with **efficient vector search in FAISS**

### **3️⃣ Store PDF Content in FAISS Vector Database**
- **Embeds extracted text** using **Google Generative AI Embeddings**  
- **FAISS Index** stores the embeddings for **fast and efficient retrieval**

### **4️⃣ Process User Queries**
- When a user asks a question:  
  1. FAISS finds the **most relevant text chunks**  
  2. The **Gemini AI model** generates a **detailed response**  
  3. Streamlit displays the answer  

### **5️⃣ User Interface (Streamlit)**
- Simple **web UI** built with **Streamlit**  
- Allows users to **upload PDFs** and **ask questions interactively**  

---

## 🔧 **Installation Instructions**
### **1️⃣ Clone the Repository**
requirements.txt will be uploaded
