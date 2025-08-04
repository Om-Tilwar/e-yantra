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

# --- All functions before QASystem class remain the same ---
def download_and_extract_text(pdf_url: str, request_id: str) -> str:
    # ... no changes here
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
        # ... no changes here
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

    # The synchronous methods can remain if you need them
    # ...

    # --- REVISED ASYNC METHOD ---
    # The @retry decorator is REMOVED from here
    async def get_answer_async(self, question: str, namespace: str, request_id: str) -> str:
        """Asynchronously answers a question by retrieving context and querying the LLM."""
        
        logging.info(f"[Request ID: {request_id}] ASYNC: Starting task for question: '{question}'")
        loop = asyncio.get_running_loop()

        # 1. Embed the question in a separate thread (CPU-bound)
        question_embedding = await loop.run_in_executor(
            None, self.embedding_model.encode, question
        )

        # 2. Query Pinecone in a separate thread (sync I/O)
        query_result = await loop.run_in_executor(
            None,
            lambda: self.index.query(
                namespace=namespace,
                vector=question_embedding.tolist(),
                top_k=5,
                include_metadata=True
            )
        )
        context = "\n\n".join(match['metadata']['text'] for match in query_result['matches'])
        logging.info(f"[Request ID: {request_id}] ASYNC: Retrieved {len(query_result['matches'])} context snippets for '{question}'.")

        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question accurately. "
            "If the answer is not available in the context, state that clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        
        # 3. Call the LLM using a helper function that has the retry logic
        # This is the new, more robust pattern
        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=10),
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(InternalServerError),
            before_sleep=lambda retry_state: logging.warning(
                f"[Request ID: {request_id}] Groq over capacity for question '{question}'. "
                f"Retrying in {retry_state.next_action.sleep:.2f} seconds..."
            )
        )
        async def invoke_llm_with_retry():
            logging.info(f"[Request ID: {request_id}] ASYNC: Sending prompt to Groq LLM for '{question}'.")
            return await self.llm.ainvoke(prompt)

        # Execute the retry-enabled function
        response = await invoke_llm_with_retry()
        
        logging.info(f"[Request ID: {request_id}] ASYNC: Received response from LLM for '{question}'.")
        return response.content.strip()