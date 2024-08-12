import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorscores import FAIS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationalBufferMemory
from langchain.llms import HuggingFacePipeline
from PyPDF2 import PDFReader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

def get_pdf_text(pdf):
  text = ''
  pdf_read = PDFReader(pdf)
  for page in pdf_read.pages:
    text += page.extract_text()
  return text

def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
  chunks = text_splitter.split_text(text)
  return chunks

def get_vectorscore(chunks):
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectorscore =  FAISS.from_texts(texts = chunks, embedding = embeddings)
  return vectorscore

def flan_t5_pipeline():
  model_id = "google/flan_t5-large"
  tokenizer = T5Tokenizer.from_pretrained(model_id)
  model = T5ForConditionalGeneration.from_pretrained(model_id)
  pipe = pipeline(
    "text2text-generation",
    model = model,
    tokenizer = tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.95
  )
  return HuggingFacePipline(pipeline = pipe)

def get_conversation_chain():
  llm = get_t5_pipeline()
  memory = ConversationalBufferMemory(memory_key='chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
  return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("AI: ", message.content)

if __name__ == '__main__':
  st.set_page_config("Chat with your pdf using FLAN-T5", page_icon = ":books:")
  st.header("PDF Chat")

  if "conversation" not in st.session_state:
    st.session_state.conversation = "None"
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = "None"

question = st.text_input("Ask a question about your PDF:")
if question:
  handle_user_input(user_question)

with st.sidebar:
  st.subheader("Your PDF Documents")
  pdf_docs = st.file_uploader("Upload your PDFs and click Process",accept_multiple_files=True, type = "pdf")
  if st.button("process"):
    if not pdf_docs:
      st.error("please upload atleast 1 pdf file")
  else:
    with st.spinner("Processing"):
      # getting text from pdfs
      txt = ""
      for pdf in pdf_docs:
        txt += get_pdf_text(pdf)
      # getting chunks 
      chunks = get_text_chunks(txt)
      # getting vectore score for the pdf chunks
      vscore = get_vectorescore(chunks)
      # Creating conversation
      st.session_state.conversation = get_conversation_chain()
   st.success("Processing complete! You can now ask questions about your PDFs.")

