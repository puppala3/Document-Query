# LangChain and NVIDIA NIM RAG Application

A Streamlit-based application for document querying using NVIDIA NIM and Meta Llama 3 through LangChain integration. Upload PDFs and interact with their content using advanced RAG (Retrieval Augmented Generation) capabilities.

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
- Upload PDF documents via sidebar
- Click "Process Documents" to initialize vector store
- Enter questions in the chat interface
- View AI-generated answers and relevant document sections

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
