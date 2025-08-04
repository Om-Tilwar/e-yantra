# config.py

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# --- API Keys and Auth Tokens ---
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # No longer needed for the LLM
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # This is now used for the LLM

# --- Pinecone Settings ---
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "hackrx-vector-db"

# --- Model Settings ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GROQ_LLM_MODEL = "llama3-8b-8192" 