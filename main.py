# # main.py

# import logging
# from typing import List
# import requests
# from fastapi import FastAPI, Depends, HTTPException, status, Header, APIRouter
# from pydantic import BaseModel, HttpUrl
# import uuid

# from utils import download_and_extract_text, QASystem
# from config import AUTH_TOKEN

# # --- Setup basic logging ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- FastAPI App Initialization ---
# app = FastAPI(
#     title="Document Q&A API",
#     description="An API to answer questions from a PDF document using a RAG pipeline.",
#     version="1.0.0"
# )

# # --- Create an APIRouter with a prefix ---
# router = APIRouter(prefix="/v1/api")

# # --- Initialize Q&A System on Startup ---
# qa_system = None
# try:
#     qa_system = QASystem()
#     logging.info("QASystem initialized successfully on startup.")
# except Exception as e:
#     logging.critical(f"FATAL: Failed to initialize QASystem on startup. Error: {e}", exc_info=True)

# # --- Pydantic Models for Request and Response ---
# class QARequest(BaseModel):
#     documents: HttpUrl
#     questions: List[str]

# class QAResponse(BaseModel):
#     answers: List[str]

# # --- Dependency for Authentication ---
# async def verify_token(authorization: str = Header(..., description="Bearer token for authorization.")):
#     if not authorization.startswith("Bearer ") or authorization.split(" ")[1] != AUTH_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid or missing bearer token.",
#         )

# # --- API Endpoint attached to the router ---
# @router.post("/hackrx/run",
#              response_model=QAResponse,
#              summary="Process a PDF and answer questions",
#              tags=["Q&A System"])
# async def run_hackrx(request_data: QARequest, _=Depends(verify_token)):
#     request_id = str(uuid.uuid4())
#     logging.info(f"[Request ID: {request_id}] Received new request.")

#     if not qa_system:
#         logging.error(f"[Request ID: {request_id}] Halting request because QASystem is not available.")
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Q&A system is not available. Please check server logs."
#         )

#     try:
#         pdf_url = str(request_data.documents)
#         questions = request_data.questions
#         logging.info(f"[Request ID: {request_id}] Processing PDF: {pdf_url} with {len(questions)} questions.")

#         # Step 1: Download and extract text
#         logging.info(f"[Request ID: {request_id}] STEP 1: Starting download and text extraction.")
#         document_text = download_and_extract_text(pdf_url, request_id)
#         if not document_text:
#             logging.error(f"[Request ID: {request_id}] Failed to extract any text from the document.")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Failed to extract text from the document."
#             )
#         logging.info(f"[Request ID: {request_id}] STEP 1: Completed. Extracted {len(document_text)} characters.")

#         # Step 2: Index the document content
#         logging.info(f"[Request ID: {request_id}] STEP 2: Starting document setup and indexing.")
#         namespace = qa_system.setup_document_context(document_text, request_id)
#         logging.info(f"[Request ID: {request_id}] STEP 2: Completed. Document indexed under namespace: {namespace}")

#         # Step 3: Answer questions
#         logging.info(f"[Request ID: {request_id}] STEP 3: Starting to answer questions.")
#         answers = []
#         for i, q in enumerate(questions):
#             logging.info(f"[Request ID: {request_id}] Answering question {i+1}/{len(questions)}: '{q}'")
#             answer = qa_system.get_answer(q, namespace, request_id)
#             answers.append(answer)
#             logging.info(f"[Request ID: {request_id}] Got answer for question {i+1}.")
#             # The time.sleep() was removed to prevent server timeouts.

#         logging.info(f"[Request ID: {request_id}] All questions answered. Sending response.")
#         return QAResponse(answers=answers)

#     except requests.exceptions.RequestException as e:
#         logging.error(f"[Request ID: {request_id}] Failed to download PDF from URL: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Could not download the document from the URL: {pdf_url}"
#         )
#     except Exception as e:
#         logging.error(f"[Request ID: {request_id}] An unexpected error occurred: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An internal server error occurred."
#         )

# # Include the router in the main app
# app.include_router(router)


import logging

from typing import List

import requests

from fastapi import FastAPI, Depends, HTTPException, status, Header, APIRouter

from pydantic import BaseModel, HttpUrl

import uuid

from utils import download_and_extract_text, QASystem
from config import AUTH_TOKEN

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Document Q&A API",
    description="An API to answer questions from a PDF document using a RAG pipeline.",
    version="1.0.0"
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

# --- Pydantic Models for Request and Response ---
class QARequest(BaseModel):
    documents: HttpUrl
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

# --- API Endpoint attached to the router ---
@router.post("/hackrx/run",
             response_model=QAResponse,
             summary="Process a PDF and answer questions",
             tags=["Q&A System"])
async def run_hackrx(request_data: QARequest, _=Depends(verify_token)):
    request_id = str(uuid.uuid4())
    logging.info(f"[Request ID: {request_id}] Received new request.")

    if not qa_system:
        logging.error(f"[Request ID: {request_id}] Halting request because QASystem is not available.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Q&A system is not available. Please check server logs."
        )

    try:
        pdf_url = str(request_data.documents)
        questions = request_data.questions
        logging.info(f"[Request ID: {request_id}] Processing PDF: {pdf_url} with {len(questions)} questions.")

        # Step 1: Download and extract text
        logging.info(f"[Request ID: {request_id}] STEP 1: Starting download and text extraction.")
        document_text = download_and_extract_text(pdf_url, request_id)
        if not document_text:
            logging.error(f"[Request ID: {request_id}] Failed to extract any text from the document.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract text from the document."
            )
        logging.info(f"[Request ID: {request_id}] STEP 1: Completed. Extracted {len(document_text)} characters.")

        # Step 2: Offline indexing skipped (embeddings already upserted)
        # We assume the document embeddings were created in a pre-processing step
        # and stored under a namespace equal to the PDF URL.
        namespace = pdf_url
        logging.info(f"[Request ID: {request_id}] STEP 2: Skipped offline indexing. Using namespace: {namespace}")

        # Step 3: Answer questions
        logging.info(f"[Request ID: {request_id}] STEP 3: Starting to answer questions.")
        answers = []
        import time
        for i, q in enumerate(questions):
            logging.info(f"[Request ID: {request_id}] Answering question {i+1}/{len(questions)}: '{q}'")
            answer = qa_system.get_answer(q, namespace, request_id)
            answers.append(answer)
            logging.info(f"[Request ID: {request_id}] Got answer for question {i+1}.")
            time.sleep(1)

        logging.info(f"[Request ID: {request_id}] All questions answered. Sending response.")
        return QAResponse(answers=answers)

    except requests.exceptions.RequestException as e:
        logging.error(f"[Request ID: {request_id}] Failed to download PDF from URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not download the document from the URL: {pdf_url}"
        )
    except Exception as e:
        logging.error(f"[Request ID: {request_id}] An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred."
        )

# Include the router in the main app
app.include_router(router)
