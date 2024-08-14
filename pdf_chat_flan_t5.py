#import streamlit as st
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
#from langchain.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
#from langchain.llms import HuggingFaceHub
#from PyPDF2 import PdfReader



import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceApi

huggingface_token = st.secrets["api_key"]

# Function to create and load models (SentenceTransformer)
def create_llm_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    flan_api = InferenceApi(repo_id="google/flan-t5-large", token=huggingface_token)
    return embedding_model, flan_api

# Initialize models
embedding_model, flan_api = create_llm_models()

# Vector store (FAISS)
index = None
pdf_texts = []

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def create_vector_store(texts):
    global index
    # Generate embeddings for each text chunk
    embeddings = embedding_model.encode(texts)
    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

def handle_user_input(query, texts):
    # Search vector store for relevant text chunks
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 matches
    relevant_texts = [texts[i] for i in I[0]]
    combined_text = "\n\n".join(relevant_texts)
    
    # Prepare input for Flan-T5
    input_text = f"Context: {combined_text}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate a response using Flan-T5 via Hugging Face API
    response = flan_api(inputs=input_text)
    
    return response.get("generated_text", "")

# Streamlit app layout
st.title("Chat with Your PDFs")

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    process_button = st.button("Process PDFs")

    if process_button:
        try:
            pdf_texts = []
            for uploaded_file in uploaded_files:
                with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                    text = extract_text_from_pdf(uploaded_file)
                    pdf_texts.append(text)

            with st.spinner("Generating embeddings and creating vector store..."):
                create_vector_store(pdf_texts)
                st.success("PDFs processed and vector store created successfully!")

            user_input = st.text_input("Ask something about the PDFs:")
            if user_input:
                with st.spinner("Generating response..."):
                    response = handle_user_input(user_input, pdf_texts)
                    st.write("### Response:")
                    st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please upload PDF files to proceed.")
    
    # Add some custom CSS to auto-scroll to the bottom
    st.markdown("""
        <style>
            .element-container {
                overflow: auto;
                max-height: 500px;
            }
        </style>
    """, unsafe_allow_html=True)

