import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

#Tracking performance using Langsmith
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Chat with Document"

# Main App
st.title("🦜🔗 LangChain and NVIDIA NIM RAG App")
st.subheader("Document Query using Meta Llama 3")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'nvidia_api_key' not in st.session_state:
    st.session_state.nvidia_api_key = None

# Sidebar
with st.sidebar:
    st.header("📁 Document Upload")
    
    # NVIDIA API Key input
    st.subheader("NVIDIA API Key")
    nvidia_api_key = st.text_input("Enter your NVIDIA API key:", type="password")
    
    if nvidia_api_key and nvidia_api_key != st.session_state.nvidia_api_key:
        st.session_state.nvidia_api_key = nvidia_api_key
        st.success("NVIDIA API key has been updated!")
    
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.session_state.nvidia_api_key:
        if st.button("Process Documents", key="process_docs"):
            with st.spinner("Processing documents..."):
                try:
                    # Set the API key
                    os.environ['NVIDIA_API_KEY'] = st.session_state.nvidia_api_key
                    
                    all_documents = []
                    for uploaded_file in uploaded_files:
                        # Process PDF
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_file_path = temp_file.name

                        loader = PyPDFLoader(temp_file_path)
                        documents = loader.load()
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                        split_documents = text_splitter.split_documents(documents)
                        
                        os.unlink(temp_file_path)
                        all_documents.extend(split_documents)
                    
                    # Create vector store
                    embeddings = NVIDIAEmbeddings()
                    st.session_state.vectors = FAISS.from_documents(all_documents, embeddings)
                    st.session_state.documents = all_documents
                    st.success("Vector Store DB with NVIDIA Embedding is Ready!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    if st.button("Clear Documents"):
        st.session_state.documents = None
        st.session_state.vectors = None
        st.success("Documents cleared! You can upload new files.")

# Main content
if st.session_state.vectors is not None and st.session_state.nvidia_api_key:
    st.header("💬 Chat with your PDFs")
    
    # Set the API key before creating the LLM
    os.environ['NVIDIA_API_KEY'] = st.session_state.nvidia_api_key
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct") #Meta Llama 3 
    
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
    )
    
    prompt1 = st.text_input("Enter your question about the documents:")
    
    if prompt1:
        with st.spinner("Generating response..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': prompt1})
            
        st.subheader("Answer:")
        st.write(response['answer'])
        
        with st.expander("📚 Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Relevant Document {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")
else:
    if not st.session_state.nvidia_api_key:
        st.warning("Please enter your NVIDIA API key in the sidebar.")
    if st.session_state.vectors is None:
        st.info("👈 Please upload and process documents using the sidebar to start chatting.")
