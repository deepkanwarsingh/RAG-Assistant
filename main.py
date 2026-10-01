import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

ai = genai.Client(api_key=API_KEY)

app = FastAPI(title="Gemini Document Assistant")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def frontend():
    return FileResponse("static/index.html")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

current_document = None
current_filename = None


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Gemini Document Assistant API is running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global current_document, current_filename

    if Path(file.filename or "").suffix.lower() != ".txt":
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported."
        )

    content = await file.read()

    if len(content) > 2 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File must be smaller than 2 MB."
        )

    try:
        document = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded text."
        )

    if not document.strip():
        raise HTTPException(
            status_code=400,
            detail="The document is empty."
        )

    current_document = document
    current_filename = file.filename

    safe_name = Path(file.filename).name
    (UPLOAD_DIR / safe_name).write_text(document, encoding="utf-8")

    return {
        "message": "Document uploaded successfully",
        "filename": current_filename,
        "characters": len(document)
    }


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not current_document:
        raise HTTPException(
            status_code=400,
            detail="Upload a document before asking a question."
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the supplied document.
Keep your answer simple and clear.
Do not invent information.
If the answer is not present, say:
"I could not find that information in the document."

DOCUMENT:
{current_document}

QUESTION:
{question}
"""

    try:
        response = ai.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        return {
            "filename": current_filename,
            "question": question,
            "answer": response.text or ""
        }

    except Exception as error:
        print("Gemini error:", error)
        raise HTTPException(
            status_code=502,
            detail="Gemini could not answer the question."
        )