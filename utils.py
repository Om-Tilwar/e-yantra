# utils.py

import requests
import fitz  # PyMuPDF
import uuid
import logging
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq # MODIFIED: Import ChatGroq

from config import (
    GROQ_API_KEY, # MODIFIED: Import Groq API key
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    GROQ_LLM_MODEL # MODIFIED: Import Groq model name
)

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def download_and_extract_text(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        logging.info(f"Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        logging.info("Extracting text from PDF.")
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDF: {e}")
        raise
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise

class QASystem:
    def __init__(self):
        """Initializes all components of the Q&A system."""
        logging.info("--- Initializing QASystem ---")
        
        # 1. Initialize the Groq LLM (MODIFIED)
        logging.info("Step 1: Initializing Groq LLM...")
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_LLM_MODEL,
            temperature=0,
        )
        logging.info("   ✅ Groq LLM initialized.")
        
        # 2. Initialize the sentence embedding model
        logging.info("Step 2: Initializing SentenceTransformer embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = 384
        logging.info(f"   ✅ Embedding model '{EMBEDDING_MODEL}' loaded.")

        # 3. Initialize Pinecone connection
        logging.info("Step 3: Initializing Pinecone connection...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        logging.info("   ✅ Pinecone connection successful.")
        
        # 4. Check for and create Pinecone index if needed
        logging.info(f"Step 4: Verifying Pinecone index '{self.index_name}'...")
        if self.index_name not in self.pc.list_indexes().names():
            logging.warning(f"   Index '{self.index_name}' not found. Creating new index...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT)
            )
            logging.info(f"   ✅ Index '{self.index_name}' created.")
        else:
            logging.info(f"   ✅ Index '{self.index_name}' already exists.")
            
        self.index = self.pc.Index(self.index_name)
        logging.info("--- ✅ QASystem initialization complete ---")

    def _chunk_text(self, text: str):
        """Splits a long text into smaller, manageable chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=150, length_function=len
        )
        return text_splitter.split_text(text)

    def setup_document_context(self, text: str) -> str:
        """Processes a document's text: chunks, embeds, and upserts to Pinecone."""
        namespace = str(uuid.uuid4())
        logging.info(f"Setting up new document context in namespace: {namespace}")
        chunks = self._chunk_text(text)
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        vectors_to_upsert = [
            {'id': f'vec-{i}', 'values': emb, 'metadata': {'text': chunk}}
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        
        # Upsert in batches to avoid overwhelming the connection
        for i in range(0, len(vectors_to_upsert), 100):
            batch = vectors_to_upsert[i:i + 100]
            self.index.upsert(vectors=batch, namespace=namespace)
        
        logging.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to namespace {namespace}.")
        return namespace

    def get_answer(self, question: str, namespace: str) -> str:
        """Answers a question by retrieving context from Pinecone and querying the LLM."""
        question_embedding = self.embedding_model.encode(question).tolist()
        
        query_result = self.index.query(
            namespace=namespace,
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        
        context = "\n\n".join(match['metadata']['text'] for match in query_result['matches'])
        
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question accurately. "
            "If the answer is not available in the context, state that clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        response = self.llm.invoke(prompt)
        return response.content.strip()