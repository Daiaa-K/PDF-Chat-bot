#import streamlit as st
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
#from langchain.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
#from langchain.llms import HuggingFaceHub
#from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import logging
from huggingface_hub import InferenceApi

# Set up logging
logging.basicConfig(level=logging.INFO)

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

# Function to create an Inference API client
def get_inference_api():
    huggingface_api_token = st.secrets["api_key"]
    return InferenceApi(repo_id="google/flan-t5-large", token=huggingface_api_token)

# Function to get answer from FLAN-T5
def get_flan_t5_response(inference_api, prompt, max_length=512):
    parameters = {
        "max_length": max_length,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True
    }
    response = inference_api(inputs=prompt, parameters=parameters)
    return response[0]['generated_text']

# Function to handle user questions and get answer
def handle_user_input(user_question, vectorstore, inference_api, memory):
    # Retrieve relevant chunks from the vectorstore
    relevant_docs = vectorstore.similarity_search(user_question, k=3)
    
    # Construct the prompt
    context = "\n".join([doc.page_content for doc in relevant_docs])
    chat_history = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages[-4:]])  # Last 4 messages
    prompt = f"Answer the question based on the context and chat history. If you can't answer, say 'I don't know'.\n\nContext: {context}\n\nChat History:\n{chat_history}\n\nQuestion: {user_question}\n\nAnswer:"
    
    # Get response from FLAN-T5
    response = get_flan_t5_response(inference_api, prompt)
    
    # Update memory
    memory.chat_memory.add_user_message(user_question)
    memory.chat_memory.add_ai_message(response)
    
    return response

# Main function
if __name__ == '__main__':
    st.set_page_config(page_title="Chat with PDF using FLAN-T5", page_icon=":books:", layout="wide")
    st.header("Chat with your PDF💬")
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    inference_api = get_inference_api()

    # Create a two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Chat interface
        chat_container = st.container()
        
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            if st.session_state.vectorstore is None:
                st.error("Please upload and process a PDF before asking questions.")
            else:
                with st.spinner("Thinking..."):
                    response = handle_user_input(user_question, st.session_state.vectorstore, inference_api, st.session_state.memory)
                    st.session_state.chat_history.append(("Human", user_question))
                    st.session_state.chat_history.append(("AI", response))

        # Display chat history
        with chat_container:
            for sender, message in st.session_state.chat_history:
                if sender == "Human":
                    st.markdown(f"**Question:** {message}")
                else:
                    st.markdown(f"**Answer:** {message}")

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
                    # Create vector store
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                st.success("Processing complete! You can now ask questions about your PDFs.")

    # Add some custom CSS to auto-scroll to the bottom
    st.markdown("""
        <style>
            .element-container {
                overflow: auto;
                max-height: 500px;
            }
        </style>
    """, unsafe_allow_html=True)
