---
license: apache-2.0
title: Document_Query_LLM
sdk: streamlit
emoji: 🏆
colorFrom: green
colorTo: red
short_description: Document_Query_LLM
---
# Document Query RAG Application

A Streamlit-based application for document querying using NVIDIA NIM and Meta Llama 3 through LangChain integration. An effective application to query single or multiple pdf documents using RAG capabilities leveraging the inferencing of NVIDIA NIM

## Features
- PDF document processing and analysis
- NVIDIA NIM embeddings integration
- Interactive Q&A with Meta Llama 3 (70B parameter model)
- Document similarity search
- Multi-document support
- Session state management

## Prerequisites
- Python 3.10
- NVIDIA API Key

## Create an environment and Install dependencies
### Environment creation
```bash
conda create -p name python==3.10 -y
```
### Environment activation
```bash
conda activate name/
```
### Install dependencies from requirements.txt
```bash
pip install -r requirments.txt
```
## Create '.env' file
NVIDIA_API_KEY=your_api_key_here

## Running the application in a terminal<br>
```bash
streamlit run app.py
```
## Application Flow
- Uploaded documents are split into 700-character chunks with 50-character overlap
- NVIDIA NIM creates embeddings stored in FAISS Index vector database
- RAG pipeline retrieves relevant contexts which is fed to LLM
- Meta Llama 3 processes user queries along with retrieved contexts to generate accurate responses
### Document processing pipeline
```bash
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, 
    chunk_overlap=50
)
split_documents = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = NVIDIAEmbeddings()
vectors = FAISS.from_documents(all_documents, embeddings)
```
### LLM and RAG pipeline
```bash
# Initialize LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Create RAG chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Generate response
response = retrieval_chain.invoke({'input': prompt1})
```
## Technical Specifications
- Text Chunking: 700 characters with 50 character overlap
- Vector Store: FAISS implementation
- LLM: meta/llama3-70b-instruct
- Embeddings: NVIDIA NIM
## Built With
- Streamlit - Web framework
- LangChain - LLM framework
- NVIDIA NIM - AI endpoints
- FAISS - Vector database
## Libraries used
- python-dotenv - to load environment variables from .env file
- pypdf - reading and extracting text from pdf
- faiss-cpu - for similarity search and clustering dense vectors
- langchain_nvidia_ai_endpoints - Provides integration with nvidia-ai_endpoints for embeddings and LLM capbilities
- langchain_community -  for buiding applications using LLMs, including document loaders and vector stores
- streamlit - for creating web-based user interface
## Note
- Requires valid NVIDIA API key with sufficient credits for embeddings and inference.

