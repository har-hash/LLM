import os
import shutil
import httpx
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Import our custom modules
from src.document_parser import parse_document
from src.chunking import chunk_document
from src.vector_store import get_vector_store
from src.llm_handler import parse_query_with_llm, generate_final_answer
from src.schemas import QueryRequest, Answer # Re-using schemas for the UI part

# --- Configuration & Setup ---
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(
    title="IntelliQueryAI",
    description="LLM-Based Intelligent Document Query & Decision System"
)

# --- CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key Security ---
API_KEY = "24c0b18c08d53ed45670770484a01479994d7aa84b8d9755db17a7a93701a445"
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Validates the API key from the Authorization header.
    """
    try:
        bearer, key = api_key.split()
        if bearer.lower() != "bearer" or key != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API Key")
        return key
    except (ValueError, IndexError):
        raise HTTPException(status_code=403, detail="Invalid Authorization header format. Expected 'Bearer <key>'.")

# --- Pydantic Schemas ---
class HackRxRequest(BaseModel):
    """
    Defines the structure for the Hackathon submission request.
    'documents' is now a single string URL.
    """
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    """
    Defines the structure for the Hackathon submission response.
    """
    answers: List[str]

# ==============================================================================
# === INTERACTIVE UI ENDPOINTS                                               ===
# ==============================================================================

@app.post("/upload_document/", summary="Upload and process a document for the UI")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    """
    Endpoint for the frontend UI to upload a document.
    It parses, chunks, and indexes the document in a vector store for the given session.
    """
    try:
        file_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        raw_text = parse_document(file_path)
        chunks = chunk_document(raw_text, document_name=file.filename)
        vector_store = get_vector_store(session_id)
        vector_store.build_index(chunks)
        
        os.remove(file_path) # Clean up the uploaded file
        return {"session_id": session_id, "filename": file.filename, "message": f"Document processed and indexed successfully. Total chunks: {len(chunks)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during document upload: {e}")

@app.post("/query/", response_model=Answer, summary="Query a processed document from the UI")
async def query_document(request: QueryRequest):
    """
    Endpoint for the frontend UI to ask a question about documents in a session.
    """
    try:
        vector_store = get_vector_store(request.session_id)
        
        # Use the LLM to understand the user's query
        parsed_query = parse_query_with_llm(request.question)
        search_query = f"Intent: {parsed_query.intent}. Details: {parsed_query.details}"
        
        # Search for relevant clauses in the vector store
        relevant_clauses = vector_store.search(search_query, top_k=5)
        if not relevant_clauses:
            raise HTTPException(status_code=404, detail="Could not find relevant clauses for your query.")
            
        # Generate the final structured answer
        final_answer = generate_final_answer(request.question, relevant_clauses)
        return final_answer
    except Exception as e:
        print(f"An error occurred during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# === FORMAL API ENDPOINT (for Hackathon Submission)                         ===
# ==============================================================================

@app.post("/hackrx/run",
          response_model=HackRxResponse,
          summary="Process a document from a URL and answer questions (Submission Endpoint)",
          dependencies=[Depends(get_api_key)])
async def hackrx_run(request: HackRxRequest):
    """
    This endpoint meets the formal project specifications.
    It fetches a document from a URL, processes it, and answers a list of questions.
    """
    print("==> /hackrx/run endpoint initiated. Processing request...")
    session_id = f"hackrx_session_{datetime.now().timestamp()}"
    vector_store = get_vector_store(session_id)
    all_chunks = []

    print(f"==> Found 1 document URL and {len(request.questions)} question(s).")

    # 1. Download and process the single document from the URL
    print("==> Starting document download and processing...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        doc_url = request.documents
        try:
            print(f"    -> Downloading document: {doc_url[:70]}...")
            response = await client.get(doc_url)
            response.raise_for_status()

            # Clean the file name from the URL
            file_name = doc_url.split("/")[-1].split("?")[0]
            file_path = os.path.join(UPLOADS_DIR, file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            print(f"    -> Parsing and chunking document...")
            raw_text = parse_document(file_path)
            chunks = chunk_document(raw_text, document_name=file_name)
            all_chunks.extend(chunks)
            os.remove(file_path) # Clean up
            print(f"    -> Successfully processed document. Chunks added: {len(chunks)}")
        except httpx.RequestError as e:
            print(f"    -> ERROR downloading document {doc_url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download document from URL: {doc_url}")
        except Exception as e:
            print(f"    -> ERROR processing document {doc_url}: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing the document.")

    if not all_chunks:
        print("==> ERROR: No chunks were created from the document.")
        raise HTTPException(status_code=400, detail="No documents could be processed from the provided URLs.")
    
    print(f"==> Building vector store index with {len(all_chunks)} total chunks...")
    vector_store.build_index(all_chunks)
    print("==> Index built successfully.")

    # 2. Answer all questions based on the processed document
    print("==> Starting question answering...")
    final_answers = []
    for i, question in enumerate(request.questions):
        try:
            print(f"    -> Answering question {i+1}: '{question}'")
            relevant_clauses = vector_store.search(question, top_k=5)
            
            if not relevant_clauses:
                print("    -> No relevant clauses found.")
                final_answers.append("Could not find relevant information in the provided documents to answer this question.")
                continue

            print(f"    -> Found {len(relevant_clauses)} relevant clauses. Generating final answer...")
            structured_answer = generate_final_answer(question, relevant_clauses)
            # Per spec, the final output is a list of justification strings
            final_answers.append(structured_answer.justification)
            print(f"    -> Successfully answered question {i+1}.")
        except Exception as e:
            print(f"    -> ERROR answering question '{question}': {e}")
            final_answers.append(f"An error occurred while processing the question: '{question}'")

    print("==> Finished processing all questions.")
    return HackRxResponse(answers=final_answers)

# ==============================================================================
# === ROOT ENDPOINT                                                          ===
# ==============================================================================

@app.get("/")
def read_root():
    """
    Root endpoint to check if the service is running.
    """
    return {"status": "IntelliQueryAI is running"}
