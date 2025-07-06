import streamlit as st 
import os
from langchain_groq import ChatGroq 
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit title
st.title("📄 RAG Document Q&A with Groq and LLaMA3")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    <context>

    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Load PDFs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

# User input
user_prompt = st.text_input("💬 Enter your query from the research papers:")

# Button to embed documents
if st.button("🔍 Create Document Embeddings"):
    create_vector_embeddings()
    st.success("✅ Vector database is ready.")

# Response generation
if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write(f"🕒 Response time: {round(time.process_time() - start, 2)} seconds")

    # Show answer
    st.subheader("✅ Answer:")
    st.write(response['answer'])

    # Show similar documents
    with st.expander("📚 Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.markdown(f"**Document {i+1}:**")
            st.write(doc.page_content)
