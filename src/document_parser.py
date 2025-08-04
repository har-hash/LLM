# src/document_parser.py

import pdfplumber
from docx import Document
import os
from typing import List

def parse_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    with pdfplumber.open(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text

def parse_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    doc = Document(file_path)
    full_text = [p.text for p in doc.paragraphs]
    return "\n".join(full_text)

def parse_document(file_path: str) -> str:
    """
    Detects the file type and uses the appropriate parser.
    Returns the raw text content of the document.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".pdf":
        return parse_pdf(file_path)
    elif extension == ".docx":
        return parse_docx(file_path)
    elif extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {extension}")