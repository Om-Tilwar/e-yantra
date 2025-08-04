# main.py

import logging
from typing import List
import time
import requests
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel, HttpUrl

from utils import download_and_extract_text, QASystem
from config import AUTH_TOKEN

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO)


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Document Q&A API",
    description="An API to answer questions from a PDF document using a RAG pipeline.",
    version="1.0.0"
)


# --- Initialize Q&A System on Startup ---
# This object will be shared across all requests.
qa_system = None
try:
    qa_system = QASystem()
    logging.info("QASystem initialized successfully.")
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize QASystem. Error: {e}")


# --- Pydantic Models for Request and Response ---
# These models provide automatic data validation and API documentation.
class QARequest(BaseModel):
    documents: HttpUrl  # Validates that 'documents' is a valid URL string.
    questions: List[str]


class QAResponse(BaseModel):
    answers: List[str]


# --- Dependency for Authentication ---
async def verify_token(authorization: str = Header(..., description="Bearer token for authorization.")):
    """Checks the Authorization header for a valid Bearer token."""
    if not authorization.startswith("Bearer ") or authorization.split(" ")[1] != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
        )


# --- API Endpoint ---
@app.post("/hackrx/run",
          response_model=QAResponse,
          summary="Process a PDF and answer questions",
          tags=["Q&A System"])
async def run_hackrx(request_data: QARequest, _=Depends(verify_token)):
    """
    This endpoint performs the following steps:
    1. Authenticates the request using a Bearer token.
    2. Receives a PDF URL and a list of questions.
    3. Downloads and extracts text from the PDF.
    4. Indexes the text into a vector database (Pinecone).
    5. Answers each question based on the indexed document context.
    """
    if not qa_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Q&A system is not available. Please check server logs."
        )

    try:
        # --- CORE LOGIC (unchanged) ---
        pdf_url = str(request_data.documents)  # Convert Pydantic's HttpUrl to string
        questions = request_data.questions

        logging.info(f"Processing request for PDF: {pdf_url}")

        # 1. Download and extract text
        document_text = download_and_extract_text(pdf_url)
        if not document_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract text from the document."
            )

        # 2. Index the document content
        namespace = qa_system.setup_document_context(document_text)
        logging.info(f"Document indexed under namespace: {namespace}")

        # 3. Answer all questions with a delay between each call
        answers = []
        for q in questions:
            answers.append(qa_system.get_answer(q, namespace))
            print(f"Answered question. Waiting for 5 seconds before the next one...")
            time.sleep(5) # Add a 5-second delay

        return QAResponse(answers=answers)

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download PDF from URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not download the document from the URL: {pdf_url}"
        )
    except Exception as e:
        # Generic catch-all for any other errors.
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred."
        )

# To run the app, use the command: uvicorn main:app --reload