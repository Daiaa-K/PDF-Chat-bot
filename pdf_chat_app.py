from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings,HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

def get_llm():
    return HuggingFaceHub(
        repo_id="gpt2-xl",
        model_kwargs={"temperature": 0.5, "max_length": 1024},
        huggingfacehub_api_token = st.secrets["api_key"]
    )

def process_query(knowledge_base, query, llm):
    docs = knowledge_base.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm(prompt)
    answer_start = response.find("Answer")
    return response[answer_start:].strip()
    
if __name__ == '__main__':
    st.set_page_config(page_title="Chat with PDF using FLAN-T5", page_icon=":books:", layout="wide")
    st.title("Chat with your PDF💬")

    # Create two columns with different widths
    col1, col2 = st.columns([1, 2])  # col1 is 1/3 width, col2 is 2/3 width

    with col1:
        st.header("Upload PDF")
        pdf = st.file_uploader("Upload your PDF File", type="pdf")
        
        if pdf is not None:
            with st.spinner("Processing..."):
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

            

            # Only process text and get LLM if not already done
                if 'knowledge_base' not in st.session_state:
                    st.session_state.knowledge_base = process_text(text)
                    st.session_state.llm = get_llm()
                
            st.success("PDF successfully uploaded and processed!")

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
                response = process_query(st.session_state.knowledge_base, prompt, st.session_state.llm)
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.chat_message("assistant"):
                    st.markdown("Please upload a PDF first.")
