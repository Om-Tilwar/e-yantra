# main.py

import logging
from typing import List
import uuid  # <-- FIX: Import uuid to generate request IDs
from fastapi import FastAPI, Depends, HTTPException, status, Header, APIRouter
from pydantic import BaseModel, HttpUrl

# FIX: download_and_extract_text is no longer needed in this file
from utils import QASystem
from config import AUTH_TOKEN, DOCUMENT_NAMESPACE  # <-- FIX: Import the shared namespace

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Document Q&A API",
    description="An API to answer questions from a pre-indexed PDF document.",
    version="1.1.0"
)

# --- Create an APIRouter with a prefix ---
router = APIRouter(prefix="/v1/api")

# --- Initialize Q&A System on Startup ---
qa_system = None
try:
    qa_system = QASystem()
    logging.info("QASystem initialized successfully on startup.")
except Exception as e:
    logging.critical(f"FATAL: Failed to initialize QASystem on startup. Error: {e}", exc_info=True)

# --- Pydantic Models ---
# FIX: The request now only needs questions, not the document URL.
class QARequest(BaseModel):
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# --- Dependency for Authentication ---
async def verify_token(authorization: str = Header(..., description="Bearer token for authorization.")):
    if not authorization.startswith("Bearer ") or authorization.split(" ")[1] != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
        )

# --- API Endpoint (Fast Querying) ---
@router.post("/hackrx/run",
             response_model=QAResponse,
             summary="Answer questions from the pre-indexed document",
             tags=["Q&A System"])
async def run_hackrx(request_data: QARequest, _=Depends(verify_token)):
    # FIX: Define the request_id at the start of the request.
    request_id = str(uuid.uuid4())
    logging.info(f"[Request ID: {request_id}] Received new query request.")

    if not qa_system:
        raise HTTPException(status_code=503, detail="Q&A system is not available.")

    try:
        questions = request_data.questions
        # FIX: Use the pre-defined, shared namespace from your config file.
        namespace = DOCUMENT_NAMESPACE
        logging.info(f"[Request ID: {request_id}] Processing {len(questions)} questions for namespace: '{namespace}'")

        # STEP 1: Embed all questions in one parallel batch call.
        question_embeddings = qa_system.get_embeddings(questions)

        # STEP 2: Loop through the results to get answers. This part is fast.
        answers = []
        for i, (q, q_embedding) in enumerate(zip(questions, question_embeddings)):
            logging.info(f"[Request ID: {request_id}] Answering question {i+1}/{len(questions)}: '{q}'")
            answer = qa_system.get_answer_from_embedding(q, q_embedding, namespace, request_id)
            answers.append(answer)
            logging.info(f"[Request ID: {request_id}] Got answer for question {i+1}.")

        logging.info(f"[Request ID: {request_id}] All questions answered. Sending response.")
        return QAResponse(answers=answers)

    except Exception as e:
        # FIX: Simplified the exception handling as we are no longer downloading files.
        logging.error(f"[Request ID: {request_id}] An unexpected error occurred during query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during the query process."
        )

# Include the router in the main app
app.include_router(router)