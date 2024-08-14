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
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large"
HF_API_KEY = st.secrets["api_key"]

# Function to process PDF
def process_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def truncate_text(text, max_length):
    # This is a basic truncation; you may need to adjust based on tokenization
    if len(text) > max_length:
        return text[:max_length]
    return text

# Function to chat with model
def chat_with_model(user_input, document_text):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    # Combine context and question
    combined_text = f"Context: {document_text}\nQuestion: {user_input}"
    
    # Truncate combined text to fit model input size
    truncated_text = truncate_text(combined_text, MAX_INPUT_LENGTH)
    
    payload = {"inputs": truncated_text}

    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        response_json = response.json()
        # Extract response assuming it is a single string
        return response_json[0]['generated_text'] if isinstance(response_json, list) and len(response_json) > 0 else 'No response'
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
