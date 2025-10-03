import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

# LangChain + Community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

# ----------------------
# Load secrets from Streamlit Cloud (st.secrets)
# ----------------------
groq_api_key = st.secrets["GROQ_API_KEY"]
hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]

# ----------------------
# Load environment variables
# ----------------------
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HUGGINGFACE_API_TOKEN'] = os.getenv("HUGGINGFACE_API_TOKEN")

groq_api_key = os.getenv('GROQ_API_KEY')

# ----------------------
# Embeddings
# ----------------------
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# ----------------------
# LLM
# ----------------------
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

# ----------------------
# Prompt Template
# ----------------------
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
<context>

Question: {input}
""")

# ----------------------
# Function: Create Vector Store from Uploaded PDFs
# ----------------------
def create_vector_embeddings(uploaded_files):
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF.")
        return
    
    all_docs = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    final_documents = text_splitter.split_documents(all_docs)

    # Create FAISS vector DB
    vectorstore_db = FAISS.from_documents(final_documents, embeddings)

    # Save to session
    st.session_state.vectorstore_db = vectorstore_db
    st.success("✅ Vector Database created for uploaded PDFs!")

# ----------------------
# Streamlit App
# ----------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="💬")
st.title("📄 RAG Document Q&A")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload your PDF(s)", type=["pdf"], accept_multiple_files=True
)

if st.button('📥 Create / Update Document Embeddings'):
    create_vector_embeddings(uploaded_files)

user_prompt = st.text_input("Enter your query from the uploaded document(s):")

if user_prompt:
    if "vectorstore_db" not in st.session_state:
        st.warning("⚠️ Please upload PDFs and create the vector database first!")
    else:
        # Create retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectorstore_db.as_retriever(
            search_kwargs={"k": 8}  # retrieve more chunks
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed_time = time.process_time() - start_time
        st.write(f"⏱ Response Time: {elapsed_time:.2f} seconds")

        # Display answer
        st.subheader("💡 Answer:")
        st.write(response['answer'])

        # Show retrieved context chunks
        with st.expander("📚 Document Chunks Used"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Chunk {i+1}:**\n\n{doc.page_content}")
                st.write(f"Source: {doc.metadata}")
