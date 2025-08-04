# src/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Application settings loaded from environment variables.
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # Check if the API key is set
    if not GOOGLE_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the .env file.")

# Instantiate settings to be imported by other modules
settings = Settings()