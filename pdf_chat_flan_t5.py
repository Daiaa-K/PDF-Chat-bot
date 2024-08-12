import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationalBufferMemory
from langchain.llms import HuggingFacePipeline
from PyPDF2 import PdfReader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
  
# function to create a FLAN-T5 pipeline
def flan_t5_pipeline():
    model_id = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=512,
        temperature=0.7,
        top_p=0.95
    )
    
  # wraping the pipeline with huggingface pipline for compatibility
    return HuggingFacePipeline(pipeline=pipe)
  
# function to get conversation chain
def get_conversation_chain():
    llm = get_flan_t5_pipeline()
    memory = ConversationalBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# function to handle user input
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("AI: ", message.content)
          
# Main function
if __name__ == '__main__':
    st.set_page_config(page_title="Chat with PDF using FLAN-T5", page_icon=":books:")
    st.header("Chat with your PDF 💬")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
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
                with st.spinner("Processing"):
                    # Get PDF text
                    raw_text = ""
                    for pdf in pdf_docs:
                        raw_text += get_pdf_text(pdf)
                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # Create vector store
                    vstore = get_vectorstore(text_chunks)
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vstore)
                st.success("Processing complete! You can now ask questions about your PDFs.")
