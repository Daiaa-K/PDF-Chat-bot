import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader

huggingface_token = st.secrets["Llama_key"]

#function to get text from pdf
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
  
#function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        
    )
    chunks = text_splitter.split_text(text)
    return chunks

# function to get vectorstore
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore
  
# function to create a LLM pipeline
def get_llm_pipeline():
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat-hf",
        huggingfacehub_api_token = huggingface_token,
        model_kwargs={
            "temperature": 0.7,
            "max_length": 1128,
            "max_new_tokens": 256
        }
    )
    
    return llm
  
# function to get conversation chain
def get_conversation_chain(vectorstore):
    llm = get_llm_pipeline()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
# function to handle user questions and get answer
def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process a PDF before asking questions.")
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history.append(("Human", user_question))
            st.session_state.chat_history.append(("AI", response['answer']))

# Main function
if __name__ == '__main__':
    st.set_page_config(page_title="Chat with PDF using BLOOMZ", page_icon=":books:", layout="wide")
    st.header("Chat with your PDF💬")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Create a two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Chat interface
        chat_container = st.container()
        
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            handle_user_input(user_question)

        # Display chat history
        with chat_container:
            for sender, message in st.session_state.chat_history:
                if sender == "Human":
                    st.markdown(f"**Question:** {message}")
                else:
                    st.markdown(f"Answer: {message}")

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
                    vstore = get_vectorstore(text_chunks)
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vstore)
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

