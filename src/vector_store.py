# src/vector_store.py

import faiss
import numpy as np
import os
from typing import List, Dict, Any
from .schemas import DocumentChunk
from .llm_handler import get_embeddings

# A simple in-memory cache for vector stores per session
VECTOR_STORE_CACHE = {}

class VectorStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.index_file = f"vector_store_{session_id}.index"
        self.metadata_file = f"metadata_{session_id}.json"
        
        self.index = None
        self.metadata = [] # List of DocumentChunk objects

    def build_index(self, chunks: List[DocumentChunk]):
        """Creates a FAISS index from document chunks."""
        self.metadata = [chunk.dict() for chunk in chunks]
        
        # Get embeddings for all chunk contents
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = get_embeddings(chunk_texts)
        
        # FAISS index requires a numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Create the FAISS index
        d = embeddings_np.shape[1] # dimension of vectors
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings_np)
        
        print(f"[{self.session_id}] FAISS index built with {self.index.ntotal} vectors.")
        
        # Save to our global cache
        VECTOR_STORE_CACHE[self.session_id] = self

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the index for the most relevant chunks."""
        if self.index is None:
            raise Exception(f"Index not built for session {self.session_id}. Please upload a document first.")
            
        # Embed the query
        query_embedding = get_embeddings([query])[0]
        query_np = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_np, top_k)
        
        # Retrieve the results from metadata
        results = [self.metadata[i] for i in indices[0]]
        
        # You could also filter by distance if needed
        # e.g., results = [self.metadata[i] for i, dist in zip(indices[0], distances[0]) if dist < threshold]
        
        return results

def get_vector_store(session_id: str) -> VectorStore:
    """Factory function to get a vector store from the cache or create a new one."""
    if session_id not in VECTOR_STORE_CACHE:
        VECTOR_STORE_CACHE[session_id] = VectorStore(session_id)
        print(f"Created new vector store for session: {session_id}")
    return VECTOR_STORE_CACHE[session_id]