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
def get_knowledge_base(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return knowledge_base
  
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

     with col2:
        st.header("Chat")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What would you like to know about the PDF?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            if 'knowledge_base' in st.session_state:
                response = process_query(st.session_state.knowledge_base, prompt)
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.chat_message("assistant"):
                    st.markdown("Please upload a PDF first.")
    
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
            if 'knowledge_base' not in st.session_state:
                st.session_state.knowledge_base = get_knowledge_base(text_chunks)
                
