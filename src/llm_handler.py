# src/llm_handler.py

import google.generativeai as genai
import json
from .config import settings # Assuming you add GOOGLE_API_KEY to your config
from .schemas import ParsedQuery, Answer
from typing import List, Dict, Any

# Configure the Google client
genai.configure(api_key=settings.GOOGLE_API_KEY) # Update your config.py to load this key

def get_embeddings(texts: List[str], model: str = "models/embedding-001") -> List[List[float]]:
    """Generates embeddings for a list of texts using Google's model."""
    response = genai.embed_content(model=model, content=texts, task_type="RETRIEVAL_DOCUMENT")
    return response['embedding']

def parse_query_with_llm(query: str) -> ParsedQuery:
    """Uses Gemini to parse a natural language query into a structured format."""
    model = genai.GenerativeModel('gemini-1.5-flash')

    system_prompt = """
    You are an intelligent assistant for an insurance company. Your task is to parse a user's query into a structured JSON object. Do not output anything other than the JSON object.

    Classify the user's intent and extract all relevant entities. The possible intents are:
    - "coverage_check": User wants to know if something is covered.
    - "condition_retrieval": User is asking about specific conditions, waiting periods, or rules.
    - "definition_lookup": User is asking for the definition of a term.
    - "decision_check": A shorthand query with key-value pairs that requires a decision.

    Extract entities such as: age, gender, location, procedure, policy_duration, disease, etc.

    Example 1:
    Query: "46M, knee surgery, Pune, 3-month policy"
    Output:
    {
        "intent": "decision_check",
        "details": {
            "age": 46,
            "gender": "male",
            "procedure": "knee surgery",
            "location": "Pune",
            "policy_duration": "3 months"
        }
    }
    """

    full_prompt = f"{system_prompt}\n\nQuery: \"{query}\""
    response = model.generate_content(full_prompt)

    # Clean up the response to get only the JSON part
    json_text = response.text.strip().replace('```json', '').replace('```', '').strip()
    return ParsedQuery.parse_raw(json_text)


# In src/llm_handler.py

def generate_final_answer(query: str, relevant_clauses: List[Dict[str, Any]]) -> Answer:
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    context = "\n\n---\n\n".join(
        [f"Document: {c['metadata']['document_name']}\nClause {c['metadata'].get('clause_number', 'N/A')}: {c['page_content']}" for c in relevant_clauses]
    )

    # --- NEW, MORE CONVERSATIONAL PROMPT ---
    system_prompt = f"""
    You are a helpful, friendly, and conversational insurance assistant. Your name is 'IntelliQueryAI'.
    Your task is to answer the user's question in a natural way, based ONLY on the provided context clauses.
    Then, fill out the structured JSON object.

    **--- High-Quality Example Start ---**
    **User Query Example:** "What is the waiting period for pre-existing diseases (PED) to be covered?"
    **Context Clause Example:** "Clause 4.1: Pre-existing Diseases: The Company shall not be liable for any claim arising from a PED until thirty-six (36) months of continuous coverage have elapsed since the inception of the first policy. The maximum liability per claim shall be 50% of the Sum Insured."
    **Good JSON Output Example:**
    {{
        "decision": "Covered with Conditions",
        "justification": "Yes, pre-existing diseases are covered, but there's a 36-month (3 year) waiting period after your policy starts. Also, please note that claims for pre-existing diseases are limited to 50% of your total Sum Insured.",
        "amount": null,
        "conditions": "36-month waiting period. Coverage is limited to 50% of the Sum Insured.",
        "referenced_clauses": [
            {{
                "clause_number": "4.1",
                "text": "The Company shall not be liable for any claim arising from a PED until thirty-six (36) months...",
                "document_name": "policy_document.pdf"
            }}
        ]
    }}
    **--- High-Quality Example End ---**

    **Instructions for the real task:**
    1.  **`decision`**: Set to "Covered", "Not Covered", "Covered with Conditions", or "Information Provided".
    2.  **`justification`**: This is the most important field. Write a friendly, conversational answer directly addressing the user's question. Use the information from the context. **If the context mentions specific monetary limits, amounts, percentages, or time periods (e.g., 'Rs. 50,000', '1% of sum insured', '45 days'), you MUST extract and include them in this conversational answer.**
    3.  **`conditions`**: Briefly list the key conditions or limits in a structured way.
    4.  **`referenced_clauses`**: Include all the clauses you used to form your answer.
    
    **Now, complete the following task based on the real context and query.**

    **Real Context from Policy Documents:**
    {context}
    
    **Real User Query:** {query}

    **Output must be only the JSON object below:**
    """
    # --- END OF NEW PROMPT ---

    response = model.generate_content(system_prompt)
    json_text = response.text.strip().replace('```json', '').replace('```', '').strip()
    return Answer.parse_raw(json_text)