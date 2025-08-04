#utils.py 
import requests
import fitz  # PyMuPDF
import uuid
import logging
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from groq import InternalServerError # Import the specific error for retrying

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

# download_and_extract_text and the QASystem _init_ remain the same...

def download_and_extract_text(pdf_url: str, request_id: str) -> str:
    # This function remains unchanged
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

class QASystem:
    def _init_(self):
        # This function remains unchanged
        logging.info("--- Initializing QASystem ---")
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_LLM_MODEL, temperature=0)
        logging.info("  ✅ Groq LLM initialized.")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logging.info(f"  ✅ Embedding model '{EMBEDDING_MODEL}' loaded (Dimension: {self.embedding_dim}).")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        logging.info("  ✅ Pinecone connection successful.")
        if self.index_name not in self.pc.list_indexes().names():
            logging.warning(f"Index '{self.index_name}' not found. Creating new index...")
            self.pc.create_index(name=self.index_name, dimension=self.embedding_dim, metric="cosine", spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT))
            logging.info(f"  ✅ Index '{self.index_name}' created.")
        else:
            logging.info(f"  ✅ Index '{self.index_name}' already exists.")
        self.index = self.pc.Index(self.index_name)
        logging.info("--- ✅ QASystem initialization complete ---")

    # The synchronous get_answer and other methods can remain for other uses if needed
    # ...

    # --- NEW ASYNC and RESILIENT get_answer method ---
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), # Wait 2s, then 4s, up to 10s
        stop=stop_after_attempt(3), # Retry a maximum of 3 times
        retry=retry_if_exception_type(InternalServerError), # Only retry on Groq's 503 error
        before_sleep=lambda retry_state: logging.warning(
            f"Groq over capacity, retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    async def get_answer_async(self, question: str, namespace: str, request_id: str) -> str:
        """Asynchronously answers a question with automatic retries on failure."""
        
        logging.info(f"[Request ID: {request_id}] ASYNC: Starting task for question: '{question}'")
        loop = asyncio.get_running_loop()

        # 1. Embed the question (CPU-bound task) in a separate thread
        question_embedding = await loop.run_in_executor(
            None, self.embedding_model.encode, question
        )

        # 2. Query Pinecone (I/O-bound, but pinecone-client is sync, so run in thread)
        query_result = await loop.run_in_executor(
            None,
            lambda: self.index.query(
                namespace=namespace,
                vector=question_embedding.tolist(),
                top_k=5,
                include_metadata=True
            )
        )
        logging.info(f"[Request ID: {request_id}] ASYNC: Retrieved {len(query_result['matches'])} context snippets for '{question}'.")
        
        # NOTE: Your log showed "Retrieved 0 context snippets."
        # This means your namespace lookup failed. Ensure the document was indexed
        # using the exact PDF URL as the namespace. This is a separate issue to fix.
        context = "\n\n".join(match['metadata']['text'] for match in query_result['matches'])
        
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question accurately. "
            "If the answer is not available in the context, state that clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        # 3. Call the LLM asynchronously (I/O-bound)
        # The @retry decorator will automatically handle failures here.
        logging.info(f"[Request ID: {request_id}] ASYNC: Sending prompt to Groq LLM for '{question}'.")
        response = await self.llm.ainvoke(prompt)
        
        logging.info(f"[Request ID: {request_id}] ASYNC: Received response from LLM for '{question}'.")
        return response.content.strip()