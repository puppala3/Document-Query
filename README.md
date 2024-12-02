# Document Query RAG Application

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
- Uploaded documents are split into 700-character chunks with 50-character overlap
- NVIDIA NIM creates embeddings stored in FAISS vector database
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



# LangChain Integration with Python 
# Streamlit Dashboard UI
Turns python scripts into sharable web Apps <br>
It uses [streamlit](https://streamlit.io) library for creating this UI dashboard example. <br>
Streamlit Galery of Vidgets: https://streamlit.io/gallery 

## Running the application in a terminal<br>
```bash
streamlit run app.py
```
### The app will open in your browser
![img_1.png](img.png)

The default content of the input Post object will be displayed in app's sidebar<br>
```json
{
    "English Text to Translate:": "How old is your car?",
    "The translation is:": "Wie alt ist Ihr Auto?""
}
```

The input (the “English Text to Translate” and the “Language”) are passed to the function “hf_langchain()” 
```python
def hf_langchain(text):
    with open("../my_keys/my_huggingface_key.txt", mode='r', encoding='utf8') as f:
        my_token = f.readlines()[0]  # Read Saved Huggingface Key

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = my_token

    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-xxl", max_length=128, temperature=0.5, token=my_token)
    return llm
```
which, through LangChain, uses LLM to perform the text translation.

## Built With
- Python 3.10

## Python Libraries Used
- streamlit - library for creating UI
- pandas 0.24.2 -  Data Analysis Library
- langchain_community - LangChain Platform


## License
NA
