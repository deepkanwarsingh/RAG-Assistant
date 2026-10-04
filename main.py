import os
import uuid
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing services
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

from services.document_loader import load_document
from services.chunker import chunk_documents
from services.vector_store import store_chunks, search_chunks

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

ai = genai.Client(api_key=API_KEY)

app = FastAPI(title="Gemini RAG Document Assistant")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


class QuestionRequest(BaseModel):
    document_id: str
    question: str


@app.get("/")
def home():
    return FileResponse("static/index.html")


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
        print("Upload error:", error)
        raise HTTPException(
            status_code=500,
            detail="Could not process this document."
        )


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        matches = search_chunks(
            document_id=request.document_id,
            question=question,
            top_k=5
        )

        if not matches:
            return {
                "answer": "I could not find that information in the document.",
                "sources": []
            }

        context_parts = []
        for i, match in enumerate(matches, start=1):
            page = match["metadata"].get("page", 0)
            page_label = f"Page {page}" if page else "Text file"

            context_parts.append(
                f"[Source {i} | {page_label}]\n{match['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a document question-answering assistant.

Answer the question using only the SOURCES below.
Do not use outside knowledge.
If the sources do not contain enough evidence, say:
"I could not find that information in the document."

CITATION RULES:
- Cite claims using the source labels, like [Source 1].
- Only cite sources that support the claim.
- Do not invent page numbers or source labels.
- If you cannot support the answer, do not guess.

SOURCES:
{context}

QUESTION:
{question}
"""

        response = ai.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        sources = []
        for i, match in enumerate(matches, start=1):
            page = match["metadata"].get("page", 0)
            sources.append({
                "source": f"Source {i}",
                "page": page or None,
                "text": match["text"]
            })

        return {
            "answer": response.text or
                "I could not generate an answer.",
            "sources": sources
        }

    except Exception as error:
        print("Question error:", error)
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve or generate an answer."
        )