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

"Document Query RAG Application" is an end-to-end Generative AI application that enables efficient querying of single or multiple PDF documents. Built using NVIDIA NIM, LangChain and Meta's Llama 3 70B model, the application leverages Retrieval-Augmented Generation (RAG) capabilities to provide accurate and context-aware responses.

## Features
-   Multi-document querying capability, allowing users to extract insights across various PDFs simultaneously
-	Integration with NVIDIA NIM for optimized inference performance
-	Utilization of Meta's Llama 3 70B model for high-quality language understanding and generation
-	Langchain integration for seamless orchestration of the RAG pipeline
-	LangSmith integration for robust productionization, enabling advanced monitoring, debugging, and optimization
-	User-friendly interface deployed on Hugging Face for easy access and interaction
## Key components
- **Retrieval Augmented Generation (RAG)**:
RAG is a technique that enhances language models by retrieving relevant information from a knowledge base before generating responses. In this project, RAG is used to provide context-aware answers to user queries about the uploaded documents. It combines the power of information retrieval with natural language generation
- **Meta Llama 3**:
In this application, Meta Llama 3 is used as the core language model for generating responses. The specific version used is "meta/llama3-70b-instruct", which is a 70 billion parameter model fine-tuned for instruction following tasks.
- **NVIDIA NIM**:
NVIDIA NIM (NVIDIA Inference Microservices) is a cloud AI service that provides access to state-of-the-art AI models. In this project, it's used for both embeddings (NVIDIAEmbeddings) and text generation (ChatNVIDIA), leveraging NVIDIA's infrastructure for efficient AI computations.
- **LangChain**:
LangChain is a framework for developing applications powered by language models. It's extensively used in this project for document loading, text splitting, creating embeddings, vector storage, and constructing the retrieval chain. LangChain simplifies the process of building complex LLM-powered applications.
- **LangSmith**:
LangSmith is a platform for debugging, testing, and monitoring LLM applications. In this project, it's used for performance tracking, as indicated by the environment variables set for LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT.
- **Streamlit**:
Streamlit offers a quick way to turn python code into web apps. In this project, it's used to build the entire user interface, handling document uploads, user inputs, and displaying responses. Streamlit enables rapid development of interactive data applications.
- **Hugging Face**:
 Hugging Face is a platform that provides access to a wide range of pre-trained models and datasets, which can be easily integrated into AI applications. It is used  for the GenAI app deployment
## Technical Specifications
- Text Chunking: 700 characters with 50 character overlap
- Vector Store: FAISS Index
- Embeddings: NVIDIA
- LLM: meta/llama3-70b-instruct
- Web framewoke: Streamlit
- Production: LangSmith
- Deployment: Hugging Face
- Language: Python

## Prerequisites
- Python 3.10
- NVIDIA, LangChain and Hugging Face API keys

## Libraries used
- python-dotenv - to load environment variables from .env file
- pypdf - reading and extracting text from pdf
- faiss-cpu - for similarity search and clustering dense vectors
- langchain_nvidia_ai_endpoints - Provides integration with nvidia-ai_endpoints for embeddings and LLM capbilities
- langchain_community -  for buiding applications using LLMs, including document loaders and vector stores
- streamlit - for creating web-based user interface

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
## Built With
- Streamlit - Web framework
- LangChain - LLM framework
- NVIDIA NIM - AI endpoints
- FAISS - Vector database
## Note
- Requires valid NVIDIA API key with sufficient credits for embeddings and inference.

