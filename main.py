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

# ## TRACER ## Added request_id to function signature for correlated logging
def download_and_extract_text(pdf_url: str, request_id: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        logging.info(f"[Request ID: {request_id}] Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url, timeout=30) # Added timeout
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
        # Initialization logging is already good, no changes needed here.
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

    def _chunk_text(self, text: str, request_id: str):
        """Splits a long text into smaller, manageable chunks."""
        # ## TRACER ##
        logging.info(f"[Request ID: {request_id}] Starting text chunking.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150, length_function=len)
        chunks = text_splitter.split_text(text)
        logging.info(f"[Request ID: {request_id}] Text chunking complete. Created {len(chunks)} chunks.")
        return chunks

    # ## TRACER ## Added request_id
    def setup_document_context(self, text: str, request_id: str) -> str:
        """Processes a document's text: chunks, embeds, and upserts to Pinecone."""
        namespace = str(uuid.uuid4())
        logging.info(f"[Request ID: {request_id}] Setting up new document context in namespace: {namespace}")

        chunks = self._chunk_text(text, request_id)

        # ## TRACER ## This is a critical, resource-intensive step
        logging.info(f"[Request ID: {request_id}] Starting to generate embeddings for {len(chunks)} chunks. This may take time and memory...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).tolist() # show_progress_bar=False for non-interactive logs
        logging.info(f"[Request ID: {request_id}] Embedding generation complete.")

        vectors_to_upsert = [{'id': f'vec-{i}', 'values': emb, 'metadata': {'text': chunk}} for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]

        # ## TRACER ##
        logging.info(f"[Request ID: {request_id}] Starting to upsert {len(vectors_to_upsert)} vectors to Pinecone.")
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            logging.info(f"[Request ID: {request_id}] Upserting batch {i//batch_size + 1} to namespace {namespace}...")
            self.index.upsert(vectors=batch, namespace=namespace)
        
        logging.info(f"[Request ID: {request_id}] Successfully upserted all vectors to namespace {namespace}.")
        return namespace

    # ## TRACER ## Added request_id
    def get_answer(self, question: str, namespace: str, request_id: str) -> str:
        """Answers a question by retrieving context from Pinecone and querying the LLM."""
        logging.info(f"[Request ID: {request_id}] Generating embedding for the question.")
        question_embedding = self.embedding_model.encode(question).tolist()
        
        logging.info(f"[Request ID: {request_id}] Querying Pinecone for relevant context.")
        query_result = self.index.query(namespace=namespace, vector=question_embedding, top_k=5, include_metadata=True)
        
        context = "\n\n".join(match['metadata']['text'] for match in query_result['matches'])
        logging.info(f"[Request ID: {request_id}] Retrieved {len(query_result['matches'])} context snippets from Pinecone.")
        
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