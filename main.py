import os
import json
import logging
import uuid
import shutil
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.genai.errors import APIError

# Load .env before importing services
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai

from services.document_loader import load_document
from services.chunker import chunk_documents
from services.vector_store import store_chunks, search_chunks
from services.conversation_store import (
    add_message,
    create_conversation,
    get_conversation,
    get_messages,
)

logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

ai = genai.Client(api_key=API_KEY)

app = FastAPI(title="Gemini RAG Document Assistant")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"


class QuestionRequest(BaseModel):
    document_id: str
    question: str
    conversation_id: str | None = None


def build_prompt(question, matches, history=None):
    context_parts = []
    for i, match in enumerate(matches, start=1):
        page = match["metadata"].get("page", 0)
        page_label = f"Page {page}" if page else "Text file"

        context_parts.append(
            f"[Source {i} | {page_label}]\n{match['text']}"
        )

    context = "\n\n".join(context_parts)
    history_lines = [
        f"{message['role'].capitalize()}: {message['content']}"
        for message in history or []
    ]
    conversation_history = "\n".join(history_lines) or "No previous messages."
    return f"""
You are a document question-answering assistant.

Use the conversation history only to understand follow-up questions. The SOURCES
below are the only evidence you may use for factual claims.
Answer the question using only the SOURCES below.
Do not use outside knowledge.
If the sources do not contain enough evidence, say:
"I could not find that information in the document."

CITATION RULES:
- Cite claims using the source labels, like [Source 1].
- Only cite sources that support the claim.
- Do not invent page numbers or source labels.
- If you cannot support the answer, do not guess.

CONVERSATION HISTORY:
{conversation_history}

SOURCES:
{context}

QUESTION:
{question}
"""


def build_sources(matches):
    return [
        {
            "source": f"Source {i}",
            "page": match["metadata"].get("page", 0) or None,
            "text": match["text"]
        }
        for i, match in enumerate(matches, start=1)
    ]


def sse_event(name, data):
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def explain_error(error, operation):
    if isinstance(error, APIError):
        code = error.code
        provider_message = " ".join(str(error.message or "").split())[:500]
        reason = f" Gemini's reason: {provider_message}" if provider_message else ""

        if code == 429:
            return 429, (
                f"Gemini's request or usage quota has been reached.{reason} "
                "Wait briefly before retrying, or check the Gemini API quota and billing settings."
            )
        if code == 503:
            return 503, (
                f"Gemini is temporarily overloaded or unavailable.{reason} "
                "Please retry in a little while."
            )
        if code in (500, 502, 504):
            return 503, (
                f"Gemini had a temporary server-side failure (HTTP {code}).{reason} "
                "Please retry in a little while."
            )
        if code in (401, 403):
            return 502, (
                f"Gemini rejected this request because of an API credential or permission issue "
                f"(HTTP {code}).{reason} Check the API key and model access configuration."
            )
        return 502, (
            f"Gemini could not complete the request (HTTP {code}).{reason} "
            "Check the request and try again."
        )

    if isinstance(error, httpx.TimeoutException):
        return 504, (
            f"The request timed out while {operation}. The AI service may be busy; "
            "please retry."
        )
    if isinstance(error, httpx.NetworkError):
        return 503, (
            f"The AI service could not be reached while {operation}. "
            "Check the connection and retry."
        )
    if operation == "processing your document" and isinstance(error, ValueError):
        return 422, str(error)

    reference = uuid.uuid4().hex[:10]
    logger.exception("Unexpected failure while %s (reference %s)", operation, reference)
    return 500, (
        f"An unexpected error occurred while {operation}. Please retry. "
        f"Reference: {reference}."
    )


def resolve_conversation(document_id, conversation_id):
    if not conversation_id:
        return create_conversation(document_id)

    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if conversation["document_id"] != document_id:
        raise HTTPException(
            status_code=409,
            detail="Conversation belongs to a different document."
        )
    return conversation_id


@app.get("/conversations/{conversation_id}")
def get_conversation_history(conversation_id: str):
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {
        "conversation_id": conversation_id,
        "document_id": conversation["document_id"],
        "messages": get_messages(conversation_id)
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in [".txt", ".pdf"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported."
        )

    document_id = str(uuid.uuid4())
    safe_path = UPLOAD_DIR / f"{document_id}{extension}"

    try:
        with safe_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if safe_path.stat().st_size > 10 * 1024 * 1024:
            safe_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413,
                detail="File must be smaller than 10 MB."
            )

        pages = load_document(str(safe_path))
        chunks = chunk_documents(pages)

        count = store_chunks(document_id, chunks)

        return {
            "message": "Document processed successfully",
            "document_id": document_id,
            "filename": Path(filename).name,
            "chunks": count,
            "pages_extracted": len(pages)
        }

    except HTTPException:
        raise
    except Exception as error:
        safe_path.unlink(missing_ok=True)
        status_code, detail = explain_error(error, "processing your document")
        raise HTTPException(
            status_code=status_code,
            detail=detail
        )


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    conversation_id = resolve_conversation(
        request.document_id,
        request.conversation_id
    )
    history = get_messages(conversation_id, limit=12)
    add_message(conversation_id, "user", question)

    try:
        matches = search_chunks(
            document_id=request.document_id,
            question=question,
            top_k=5
        )

        if not matches:
            answer = "I could not find that information in the document."
            add_message(conversation_id, "assistant", answer)
            return {
                "answer": answer,
                "sources": [],
                "conversation_id": conversation_id
            }

        response = ai.models.generate_content(
            model="gemini-3.8-flash",
            contents=build_prompt(question, matches, history)
        )
        answer = response.text or "I could not generate an answer."
        sources = build_sources(matches)
        add_message(conversation_id, "assistant", answer, sources)

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id
        }

    except Exception as error:
        status_code, detail = explain_error(error, "searching your document or generating an answer")
        raise HTTPException(
            status_code=status_code,
            detail=detail
        )


@app.post("/ask/stream")
def ask_question_stream(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    conversation_id = resolve_conversation(
        request.document_id,
        request.conversation_id
    )
    history = get_messages(conversation_id, limit=12)
    add_message(conversation_id, "user", question)

    try:
        matches = search_chunks(
            document_id=request.document_id,
            question=question,
            top_k=5
        )
    except Exception as error:
        status_code, detail = explain_error(error, "searching your document")
        raise HTTPException(
            status_code=status_code,
            detail=detail
        )

    sources = build_sources(matches)

    def stream_events():
        yield sse_event("conversation", {"conversation_id": conversation_id})
        yield sse_event("sources", {"sources": sources})

        if not matches:
            answer = "I could not find that information in the document."
            add_message(conversation_id, "assistant", answer, sources)
            yield sse_event(
                "token",
                {"text": answer}
            )
            yield sse_event("done", {})
            return

        try:
            response_stream = ai.models.generate_content_stream(
                model="gemini-3.8-flash",
                contents=build_prompt(question, matches, history)
            )

            answer_parts = []
            for chunk in response_stream:
                if chunk.text:
                    answer_parts.append(chunk.text)
                    yield sse_event("token", {"text": chunk.text})

            answer = "".join(answer_parts) or "I could not generate an answer."
            if not answer_parts:
                yield sse_event("token", {"text": answer})
            add_message(conversation_id, "assistant", answer, sources)
            yield sse_event("done", {})
        except Exception as error:
            _, detail = explain_error(error, "generating an answer")
            yield sse_event(
                "error",
                {"detail": detail}
            )

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def frontend_not_built():
        raise HTTPException(
            status_code=503,
            detail="Frontend build not found. Run npm --prefix frontend run build."
        )