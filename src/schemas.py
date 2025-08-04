# src/schemas.py

from pydantic import BaseModel
from typing import List, Optional, Any

# --- Request Schemas ---

class QueryRequest(BaseModel):
    """
    Defines the structure for an incoming query request.
    """
    session_id: str # To track a user's session and their uploaded documents
    question: str


# --- Response Schemas ---

class ReferencedClause(BaseModel):
    """
    Details of a clause referenced in the justification.
    """
    clause_number: Optional[str] = "N/A"
    text: str
    document_name: str

class Answer(BaseModel):
    """
    The structured answer to a single query.
    """
    decision: str
    justification: str
    amount: Optional[float] = None
    referenced_clauses: List[ReferencedClause]
    conditions: Optional[str] = None


# --- Internal Data Structures ---

class DocumentChunk(BaseModel):
    """
    Represents a single chunk of a document with metadata.
    """
    page_content: str
    metadata: dict

class ParsedQuery(BaseModel):
    """
    Structured data extracted from a natural language query by the LLM.
    """
    intent: str
    details: dict[str, Any]