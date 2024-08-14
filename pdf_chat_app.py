import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import logging
import requests

#API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
#headers = {"Authorization": "Bearer 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"}

def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
  
# Function to get text from pdf
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get vectorstore
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore
  
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token = st.secrets["api_key"]
    )

def process_query(knowledge_base, query):
    llm = get_llm()
    docs = knowledge_base.similarity_search(query)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.invoke(input={"question": query, "input_documents": docs})
    return response["output_text"]



# Main function
if __name__ == '__main__':
  st.set_page_config(page_title="Chat with PDF using FLAN-T5", page_icon=":books:", layout="wide")
  st.header("Chat with your PDF💬")

  col1, col2 = st.columns([2, 1])

  with col1:
        # Chat interface
    chat_container = st.container()
        
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
      response = process_query(vstore,user_question)
    with chat_container:
      st.write(response)
    
    with col2:
    # Sidebar for PDF upload
    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
          "Upload your PDFs here and click on 'Process'",
          accept_multiple_files=True,
          type="pdf"
        )
        
    if st.button("Process"):
          if not pdf_docs:
              st.error("Please upload at least one PDF file before processing.")
          else:
              with st.spinner("Processing PDFs..."):
                # Get PDF text
                txt = ""
                for pdf in pdf_docs:
                  txt += get_pdf_text(pdf)
                  # Get the text chunks
                text_chunks = get_text_chunks(txt)
                #get knowledge base
                vstore = get_vectorstore(text_chunks)
                
