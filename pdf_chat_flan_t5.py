#import streamlit as st
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
#from langchain.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
#from langchain.llms import HuggingFaceHub
#from PyPDF2 import PdfReader

#huggingface_token = st.secrets["api_key"]

import streamlit as st
import PyPDF2
import requests

# Configuration for Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_API_KEY = st.secrets["api_key"]

# Function to process PDF
def process_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to chat with model
def chat_with_model(user_input, document_text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": f"Context: {document_text}\n\nQuestion: {user_input}"
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json().get('generated_text', 'No response')
    else:
        return f"Error: {response.status_code}, {response.text}"

# Setup layout with two columns
file_uploader, chat_column = st.columns(2)

# File uploader section
uploaded_file = file_uploader.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        document_text = process_pdf(uploaded_file)
        st.session_state['document_text'] = document_text
        file_uploader.success("File uploaded and processed successfully!")

# Chat section
if 'document_text' in st.session_state:
    user_input = chat_column.text_input("You: ", key="input")
    if user_input:
        with st.spinner("Generating response..."):
            response = chat_with_model(user_input, st.session_state['document_text'])
            chat_column.write(f"Assistant: {response}")
