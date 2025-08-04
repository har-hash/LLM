# src/chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from .schemas import DocumentChunk
import re

def chunk_document(text: str, document_name: str) -> List[DocumentChunk]:
    """
    Splits a document's text into semantic chunks.
    Here, we use a simple text splitter, but more advanced methods
    (e.g., splitting by clause numbers like "3.2.1") can be implemented.
    """
    # A simple regex to identify potential clause headers
    # This can be customized for specific document formats
    # E.g., "(Clause|Section|Article) \d+(\.\d+)*"
    
    # Using LangChain's text splitter for effective chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    texts = text_splitter.split_text(text)
    
    chunks = []
    for i, chunk_text in enumerate(texts):
        # Attempt to find a clause number at the beginning of the chunk
        clause_match = re.search(r"^\s*(\d+(\.\d+)*)\s+", chunk_text)
        clause_number = clause_match.group(1) if clause_match else f"Part_{i+1}"
        
        metadata = {
            "document_name": document_name,
            "clause_number": clause_number
        }
        chunk = DocumentChunk(page_content=chunk_text, metadata=metadata)
        chunks.append(chunk)
        
    return chunks