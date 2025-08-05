# utils.py

import requests
import fitz  # PyMuPDF
import uuid
import logging
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    GROQ_LLM_MODEL
)

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_and_extract_text(pdf_url: str, request_id: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        logging.info(f"[Request ID: {request_id}] Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        logging.info(f"[Request ID: {request_id}] Extracting text from PDF binary content.")
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except requests.exceptions.RequestException as e:
        logging.error(f"[Request ID: {request_id}] Error downloading PDF: {e}")
        raise
    except Exception as e:
        logging.error(f"[Request ID: {request_id}] Error extracting text from PDF: {e}")
        raise

# utils.py

class QASystem:
    # ... (__init__, _chunk_text, setup_document_context are the same) ...

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts in a single batch."""
        logging.info(f"Generating embeddings for a batch of {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
        logging.info("Batch embedding generation complete.")
        return embeddings

    def get_answer_from_embedding(self, question: str, question_embedding: List[float], namespace: str, request_id: str) -> str:
        """Finds an answer using a pre-computed question embedding."""
        logging.info(f"[Request ID: {request_id}] Querying Pinecone for: '{question}'")
        query_result = self.index.query(
            namespace=namespace,
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        
        context = "\n\n".join(match['metadata']['text'] for match in query_result['matches'])
        logging.info(f"[Request ID: {request_id}] Retrieved {len(query_result['matches'])} context snippets.")
        
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question accurately. "
            "If the answer is not available in the context, state that clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        logging.info(f"[Request ID: {request_id}] Sending prompt to Groq LLM.")
        response = self.llm.invoke(prompt)
        logging.info(f"[Request ID: {request_id}] Received response from LLM.")
        return response.content.strip()

    def _chunk_text(self, text: str, request_id: str):
        """Splits a long text into smaller, manageable chunks."""
        logging.info(f"[Request ID: {request_id}] Starting text chunking.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150, length_function=len)
        chunks = text_splitter.split_text(text)
        logging.info(f"[Request ID: {request_id}] Text chunking complete. Created {len(chunks)} chunks.")
        return chunks

    def setup_document_context(self, text: str, request_id: str) -> str:
        """Processes a document's text: chunks, embeds, and upserts to Pinecone."""
        namespace = str(uuid.uuid4())
        logging.info(f"[Request ID: {request_id}] Setting up new document context in namespace: {namespace}")
        
        chunks = self._chunk_text(text, request_id)
        
        logging.info(f"[Request ID: {request_id}] Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).tolist()
        logging.info(f"[Request ID: {request_id}] Embedding generation complete.")
        
        vectors_to_upsert = [{'id': f'vec-{i}', 'values': emb, 'metadata': {'text': chunk}} for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
        
        logging.info(f"[Request ID: {request_id}] Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
        
        logging.info(f"[Request ID: {request_id}] Successfully upserted all vectors to namespace {namespace}.")
        return namespace

    def get_answer(self, question: str, namespace: str, request_id: str) -> str:
        """Answers a question by retrieving context from Pinecone and querying the LLM."""
        logging.info(f"[Request ID: {request_id}] Generating embedding for the question.")
        question_embedding = self.embedding_model.encode(question).tolist()
        
        logging.info(f"[Request ID: {request_id}] Querying Pinecone for relevant context.")
        query_result = self.index.query(
            namespace=namespace,
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        
        context = "\n\n".join(match['metadata']['text'] for match in query_result['matches'])
        logging.info(f"[Request ID: {request_id}] Retrieved {len(query_result['matches'])} context snippets.")
        
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question accurately. "
            "If the answer is not available in the context, state that clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        logging.info(f"[Request ID: {request_id}] Sending prompt to Groq LLM.")
        response = self.llm.invoke(prompt)
        logging.info(f"[Request ID: {request_id}] Received response from LLM.")
        return response.content.strip()