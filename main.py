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

# Create the uploads directory if it doesn't exist
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="IntelliQueryAI",
    description="LLM-Based Intelligent Document Query & Decision System"
)

# --- CORS Middleware ---
# This is crucial for allowing the frontend (running on a different origin) to communicate with the backend.
origins = [
    "*",  # Allows all origins. For production, restrict this to your frontend's domain.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods
    allow_headers=["*"], # Allows all headers
)

# --- API Key Security for /hackrx/run endpoint ---
API_KEY = "24c0b18c08d53ed45670770484a01479994d7aa84b8d9755db17a7a93701a445"  # CHANGE THIS to a secure, random key
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Validates the API key from the Authorization header."""
    # The key is expected to be in the format "Bearer <key>"
    try:
        bearer, key = api_key.split()
        if bearer.lower() != "bearer" or key != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API Key")
        return key
    except (ValueError, IndexError):
        raise HTTPException(status_code=403, detail="Invalid Authorization header format. Expected 'Bearer <key>'.")

# --- Pydantic Schemas for /hackrx/run endpoint ---
class HackRxRequest(BaseModel):
    documents: List[str] # List of URLs
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]


# ==============================================================================
# === INTERACTIVE UI ENDPOINTS (for your index.html)                         ===
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
        os.remove(file_path)

        return {
            "session_id": session_id,
            "filename": file.filename,
            "message": f"Document processed and indexed successfully. Total chunks: {len(chunks)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", response_model=Answer, summary="Query a processed document from the UI")
async def query_document(request: QueryRequest):
    """
    Endpoint for the frontend UI to ask a question about documents in a session.
    """
    try:
        vector_store = get_vector_store(request.session_id)
        parsed_query = parse_query_with_llm(request.question)
        search_query = f"Intent: {parsed_query.intent}. Details: {parsed_query.details}"
        relevant_clauses = vector_store.search(search_query, top_k=5)

        if not relevant_clauses:
            raise HTTPException(status_code=404, detail="Could not find relevant clauses for your query.")

        final_answer = generate_final_answer(request.question, relevant_clauses)
        return final_answer
        
    except Exception as e:
        print(f"An error occurred during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# === FORMAL API ENDPOINT (for Hackathon/Project Submission)                 ===
# ==============================================================================

@app.post("/hackrx/run",
          response_model=HackRxResponse,
          summary="Process documents from URLs and answer questions (Submission Endpoint)",
          dependencies=[Depends(get_api_key)])
async def hackrx_run(request: HackRxRequest):
    """
    This endpoint meets the formal project specifications.
    It fetches documents from URLs, processes them, and answers a list of questions.
    """
    session_id = f"hackrx_session_{datetime.now().timestamp()}"
    vector_store = get_vector_store(session_id)
    all_chunks = []

    # 1. Download and process all documents from the provided URLs
    async with httpx.AsyncClient(timeout=30.0) as client:
        for doc_url in request.documents:
            try:
                response = await client.get(doc_url)
                response.raise_for_status()

                file_name = doc_url.split("/")[-1]
                file_path = os.path.join(UPLOADS_DIR, file_name)
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                raw_text = parse_document(file_path)
                chunks = chunk_document(raw_text, document_name=file_name)
                all_chunks.extend(chunks)
                os.remove(file_path)

            except httpx.RequestError as e:
                raise HTTPException(status_code=400, detail=f"Failed to download or process document: {e.url}")

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No documents could be processed from the provided URLs.")
    
    vector_store.build_index(all_chunks)

    # 2. Answer all questions sequentially
    final_answers = []
    for question in request.questions:
        try:
            relevant_clauses = vector_store.search(question, top_k=5)
            if not relevant_clauses:
                final_answers.append("Could not find relevant information in the provided documents to answer this question.")
                continue

            structured_answer = generate_final_answer(question, relevant_clauses)
            final_answers.append(structured_answer.justification)
        except Exception:
            # If one question fails, provide a generic error and move to the next
            final_answers.append(f"An error occurred while processing the question: '{question}'")

    return HackRxResponse(answers=final_answers)


# ==============================================================================
# === ROOT ENDPOINT (for Health Check)                                       ===
# ==============================================================================

@app.get("/")
def read_root():
    return {"status": "IntelliQueryAI is running"}

# To run the app from your terminal:
# uvicorn main:app --reload
