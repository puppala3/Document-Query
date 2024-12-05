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
- **python-dotenv** - to load environment variables from .env file
- **pypdf** - reading and extracting text from pdf
- **faiss-cpu** - for similarity search and clustering dense vectors
- **langchain_nvidia_ai_endpoints** - Provides integration with nvidia-ai_endpoints for embeddings and LLM capbilities
- **langchain_community** -  for buiding applications using LLMs, including document loaders and vector stores
- **streamlit** - for creating web-based user interface
## User Interface
![image alt](https://github.com/puppala3/Document-Query/blob/main/User%20Interface.PNG?raw=true)
## Application Flow
- User can upload pdf documents with browse option and click process
- Uploaded documents will be split into 700-character chunks with 50-character overlap
- NVIDIA NIM creates NVIDIA embeddings which will be stored in FAISS Index vector database
- User will be notified once the vector base is ready and then user can start querying
- Retriever retrieves relevant contexts based on similarity match between query embeddings and vector store embeddings
- Meta Llama 3 processes actual user query along with retrieved contexts to generate accurate responses

## Code Execution
### Create an environment and Install dependencies
- Environment creation
```bash
conda create -p name python==3.10 -y
```
- Environment activation
```bash
conda activate name/
```
- Install dependencies from requirements.txt
```bash
pip install -r requirments.txt
```
- Create '.env' file
NVIDIA_API_KEY=your_api_key_here

### Running the application in a terminal<br>
```bash
streamlit run app.py
```
## Code Walk
### LangSmith for tracing
- Naming the project for tracing and API key loading
```bash
#Tracking performance using Langsmith
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Chat with Document"
```
### Document processing pipeline
- Converting the uploaded pdfs to LangChain readable format
```bash
#processing uploaded documents
all_documents = []
for uploaded_file in uploaded_files:
# Process PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
         temp_file.write(uploaded_file.getvalue())
         temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
```
- Splitting documents into chucks

```bash
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, 
    chunk_overlap=50
)
split_documents = text_splitter.split_documents(documents)
```

### Create embeddings and vector store
- Creating NVIDIA Embeddings and storing in a vector store
```bash
embeddings = NVIDIAEmbeddings()
vectors = FAISS.from_documents(all_documents, embeddings)
```
### LLM and RAG pipeline
- ChatNVIDIA is to access the META Llama 3 model.
- Prompt1 is the actual query from the user.
- Retriever retrieves relevant contexts based on similarity match between query embeddings and vector store embeddings.
- Retrieval chain then feeds the actual user query and relevant contexts to the LLM for accurate reponse.

```bash
# Initialize LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
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
# Create RAG chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Generate response
response = retrieval_chain.invoke({'input': prompt1})
```
# Deployment in Hugging Face
- Create a new space in spaces of Hugging Face and also generate HF token
- Create a main.yml file in the github with an integration code in managing spaces with github actions
- Then add the hf token in the github. Settings->Secrets and variables->Actions->New repository secret
- Commit the changes and the files get pushed to hugging face
- Add the LangChain API key to Hugging Face space.Settings->Variables and secrets->secrets
# Limitations
- It fails to answer questions if a context is not found. Let’s say, you upload 10 different short stories and query for number of storing with happy endings. This is because RAG couldn’t get the relevant context for the query
- It can’t read texts from the images. Documents come with both texts and images and a multimodal model will help in that case.

# Conclusion
- Document Query RAG App is an elegant solution making complex document analysis and information retrieval accessible and efficient for users across different domains. 
## Note
- User Requires valid NVIDIA API key with sufficient credits for embeddings and inference.

