import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
import time

from dotenv import load_dotenv
groq_api_key = os.environ['GROQ_API_KEY']

st.title("Objectbox VectorsstoreDB with Llama3")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide te most accurate response based on the question
    <<context>>
    {context}
    <<context>>
    Questions: {input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
            st.session_state.embeddings=OllamaEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./us_census") # Data Ingestion
            st.session_state.docs = st.session_state.loader.load() # Document loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) # Splitting
            st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=768) # Vector Ollama embeddings

input_prompt = st.text_input("Input Prompt")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

if input_prompt:
    vector_embedding()
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input":input_prompt})
    print("Response time:",time.process_time()-start)
    st.write(response['answer'])

    # Streamlit expander
    with st.expander("Doc similarity search"):
        # finding relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------")
