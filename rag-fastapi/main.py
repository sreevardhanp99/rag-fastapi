import os
import uuid
import tempfile
import io
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import azure.storage.blob as azureblob
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    SearchField
)
import openai
from pydantic import BaseModel
import PyPDF2
from docx import Document
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Pipeline API",
    description="A RAG (Retrieve-Augment-Generate) pipeline implementation with Azure services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_UPLOAD = "uploaded-files"
AZURE_CONTAINER_CHUNKS = "processed-chunks"
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

# Initialize clients
try:
    # Blob Service Client
    blob_service_client = azureblob.BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    
    # Create containers if they don't exist
    try:
        blob_service_client.create_container(AZURE_CONTAINER_UPLOAD)
        logger.info(f"Created container: {AZURE_CONTAINER_UPLOAD}")
    except Exception as e:
        logger.info(f"Container {AZURE_CONTAINER_UPLOAD} already exists or error: {str(e)}")
    
    try:
        blob_service_client.create_container(AZURE_CONTAINER_CHUNKS)
        logger.info(f"Created container: {AZURE_CONTAINER_CHUNKS}")
    except Exception as e:
        logger.info(f"Container {AZURE_CONTAINER_CHUNKS} already exists or error: {str(e)}")
        
except Exception as e:
    logger.error(f"Error initializing blob service client: {str(e)}")
    raise

# OpenAI client configuration
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = OPENAI_API_KEY

# Search clients
search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
search_index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=search_credential
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class ProcessResponse(BaseModel):
    message: str
    chunks_created: int
    files_processed: List[str]

class EmbedResponse(BaseModel):
    message: str
    documents_embedded: int

class Response(BaseModel):
    answer: str
    sources: List[str]

# Utility functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing DOCX: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of approximately chunk_size words"""
    try:
        words = text.split()
        chunks = []
        current_chunk = []
        current_count = 0
        
        for word in words:
            current_chunk.append(word)
            current_count += 1
            
            if current_count >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error chunking text: {str(e)}")

def generate_embeddings(text: str) -> List[float]:
    """Generate embeddings using Azure OpenAI"""
    try:
        # Clean text to avoid API issues
        clean_text = text.replace("\x00", "").strip()
        if not clean_text:
            return [0.0] * 1536  # Return zero vector for empty text
            
        response = openai.Embedding.create(
            input=clean_text,
            engine=EMBEDDING_DEPLOYMENT
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

def create_search_index():
    """Create or update the search index with proper vector field configuration"""
    try:
        # Delete index if it exists
        try:
            search_index_client.delete_index(AZURE_SEARCH_INDEX_NAME)
            logger.info(f"Deleted existing index: {AZURE_SEARCH_INDEX_NAME}")
            # Wait a moment for deletion to complete
            import time
            time.sleep(3)
        except Exception as e:
            logger.info(f"Index {AZURE_SEARCH_INDEX_NAME} does not exist or could not be deleted: {str(e)}")
        
        # Define fields with proper vector configuration
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, searchable=True),
            SearchField(
                name="embedding", 
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # OpenAI embedding dimensions
                vector_search_profile_name="my-vector-config"
            )
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="my-vector-config",
                    algorithm_configuration_name="my-algorithm-config"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-algorithm-config",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE
                    )
                )
            ]
        )
        
        # Create the index
        index = SearchIndex(
            name=AZURE_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search
        )
        
        search_index_client.create_index(index)
        logger.info(f"Created search index: {AZURE_SEARCH_INDEX_NAME}")
        
        # Wait for index to be ready
        import time
        time.sleep(5)
        
    except Exception as e:
        logger.error(f"Error creating search index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating search index: {str(e)}")

def search_with_embeddings(search_client, query_embedding, top_k=3):
    """
    Perform vector search with proper error handling and compatibility
    """
    try:
        # Try different approaches based on SDK version
        try:
            # Approach 1: New vector queries format (for newer SDK versions)
            from azure.search.documents.models import VectorizedQuery
            
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k=top_k,
                fields="embedding"
            )
            
            results = search_client.search(
                search_text="",
                vector_queries=[vector_query],
                select=["id", "content", "source"],
                top=top_k
            )
            return results
            
        except (ImportError, AttributeError) as e:
            logger.info(f"VectorizedQuery not available: {str(e)}")
            
            # Approach 2: Raw vector query format
            try:
                results = search_client.search(
                    search_text="",
                    vector_queries=[{
                        "vector": query_embedding,
                        "k": top_k,
                        "fields": "embedding"
                    }],
                    select=["id", "content", "source"],
                    top=top_k
                )
                return results
                
            except Exception as e2:
                logger.info(f"Vector queries format failed: {str(e2)}")
                
                # Approach 3: Fallback to hybrid search
                results = search_client.search(
                    search_text="*",
                    select=["id", "content", "source"],
                    top=top_k
                )
                return results
                
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        # Return empty results
        return []

# API Routes
@app.post("/upload", response_model=dict)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload PDF and DOCX files to Azure Blob Storage
    """
    uploaded_files = []
    
    for file in files:
        if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.docx')):
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is not a PDF or DOCX file. Only PDF and DOCX files are allowed."
            )
        
        try:
            # Read file content
            content = await file.read()
            
            # Upload to blob storage
            blob_client = blob_service_client.get_blob_client(
                container=AZURE_CONTAINER_UPLOAD,
                blob=file.filename
            )
            
            blob_client.upload_blob(content, overwrite=True)
            uploaded_files.append(file.filename)
            logger.info(f"Uploaded file: {file.filename}")
            
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error uploading file {file.filename}: {str(e)}"
            )
    
    return {
        "message": "Files uploaded successfully",
        "uploaded_files": uploaded_files
    }

@app.post("/process", response_model=ProcessResponse)
async def process_files():
    """
    Process uploaded files: extract text, chunk, and store chunks
    """
    try:
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_UPLOAD)
        chunks_container_client = blob_service_client.get_container_client(AZURE_CONTAINER_CHUNKS)
        
        total_chunks = 0
        processed_files = []
        
        # List all blobs in the upload container
        blobs = container_client.list_blobs()
        
        for blob in blobs:
            try:
                # Download blob content
                blob_client = container_client.get_blob_client(blob.name)
                content = blob_client.download_blob().readall()
                
                # Extract text based on file type
                if blob.name.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(content)
                elif blob.name.lower().endswith('.docx'):
                    text = extract_text_from_docx(content)
                else:
                    logger.warning(f"Skipping non-PDF/DOCX file: {blob.name}")
                    continue
                
                # Skip empty files
                if not text.strip():
                    logger.warning(f"Skipping empty file: {blob.name}")
                    continue
                
                # Chunk the text
                chunks = chunk_text(text)
                
                # Upload each chunk as a separate blob
                for i, chunk in enumerate(chunks):
                    chunk_blob_name = f"{blob.name}_chunk_{i}.txt"
                    chunk_client = chunks_container_client.get_blob_client(chunk_blob_name)
                    chunk_client.upload_blob(chunk, overwrite=True)
                    total_chunks += 1
                
                processed_files.append(blob.name)
                logger.info(f"Processed file: {blob.name}, created {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing file {blob.name}: {str(e)}")
                continue
        
        if not processed_files:
            return ProcessResponse(
                message="No files to process or all files failed processing",
                chunks_created=0,
                files_processed=[]
            )
        
        return ProcessResponse(
            message="Files processed successfully",
            chunks_created=total_chunks,
            files_processed=processed_files
        )
        
    except Exception as e:
        logger.error(f"Error in process_files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/embed", response_model=EmbedResponse)
async def embed_files():
    """
    Generate embeddings for all chunks and index them in Azure AI Search
    """
    try:
        # Create search index with proper vector field configuration
        create_search_index()
        
        # Get search client
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=search_credential
        )
        
        # Process chunks
        chunks_container_client = blob_service_client.get_container_client(AZURE_CONTAINER_CHUNKS)
        chunks = list(chunks_container_client.list_blobs())
        
        if not chunks:
            return EmbedResponse(
                message="No chunks found to process. Please run /process first.",
                documents_embedded=0
            )
        
        documents = []
        embedded_count = 0
        batch_size = 5  # Smaller batch size to avoid timeouts
        
        for chunk in chunks:
            try:
                # Download chunk content
                blob_client = chunks_container_client.get_blob_client(chunk.name)
                content = blob_client.download_blob().readall().decode('utf-8', errors='ignore')
                
                # Skip empty chunks
                if not content.strip():
                    logger.warning(f"Skipping empty chunk: {chunk.name}")
                    continue
                
                # Generate embedding
                embedding = generate_embeddings(content)
                
                # Create document for search index with proper field names and types
                document = {
                    "id": str(uuid.uuid4()),
                    "content": content[:32000],  # Limit content length
                    "source": chunk.name[:1000],  # Limit source length
                    "embedding": embedding
                }
                
                documents.append(document)
                embedded_count += 1
                
                # Upload in smaller batches to avoid timeouts
                if len(documents) >= batch_size:
                    result = search_client.upload_documents(documents=documents)
                    logger.info(f"Uploaded {len(documents)} documents to search index")
                    documents = []
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.name}: {str(e)}")
                continue
        
        # Upload remaining documents
        if documents:
            result = search_client.upload_documents(documents=documents)
            logger.info(f"Uploaded {len(documents)} remaining documents to search index")
        
        return EmbedResponse(
            message="Embeddings generated and indexed successfully",
            documents_embedded=embedded_count
        )
        
    except Exception as e:
        logger.error(f"Error in embed_files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/response", response_model=Response)
async def generate_response(request: QueryRequest):
    """
    Generate response based on user query using RAG pipeline
    """
    try:
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=search_credential
        )
        
        # Generate embedding for query
        query_embedding = generate_embeddings(request.query)
        
        # Search for relevant documents
        results = search_with_embeddings(search_client, query_embedding, top_k=3)
        
        # Collect relevant content
        relevant_content = []
        sources = []
        
        try:
            # Handle different result formats
            if hasattr(results, '__iter__'):
                for result in results:
                    # Handle both dictionary and object formats
                    if isinstance(result, dict):
                        content = result.get('content', '')
                        source = result.get('source', '')
                    else:
                        content = getattr(result, 'content', '')
                        source = getattr(result, 'source', '')
                    
                    if content and content.strip():
                        relevant_content.append(content)
                    if source and source.strip():
                        sources.append(source)
                        
                    # Limit to top 3 results
                    if len(relevant_content) >= 3:
                        break
        except Exception as e:
            logger.warning(f"Error processing search results: {str(e)}")
        
        if not relevant_content:
            return Response(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                sources=[]
            )
        
        # Prepare prompt for GPT
        context = "\n\n".join(relevant_content[:3])
        
        prompt = f"""
        Based on the following context extracted from documents, please answer the user's question.
        If the context doesn't contain information needed to answer the question, say so.
        
        Context:
        {context}
        
        Question: {request.query}
        
        Answer:
        """
        
        # Generate response using GPT
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If you don't know the answer based on the context, say so."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        
        return Response(
            answer=answer,
            sources=list(set(sources))  # Remove duplicates
        )
        
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "services": {
            "azure_storage": "connected" if blob_service_client else "disconnected",
            "azure_search": "connected" if search_index_client else "disconnected",
            "openai": "configured" if OPENAI_API_KEY else "not configured"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload (POST) - Upload PDF/DOCX files",
            "process": "/process (POST) - Process uploaded files",
            "embed": "/embed (POST) - Generate embeddings",
            "response": "/response (POST) - Generate responses to queries",
            "health": "/health (GET) - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)