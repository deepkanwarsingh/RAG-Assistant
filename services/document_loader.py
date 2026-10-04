
from pathlib import Path
import pymupdf


def load_document(file_path: str):
    """
    Extract text from PDF or TXT.
    Returns a list of pages/sections with text and metadata.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    documents = []

    if extension == ".pdf":
        pdf = pymupdf.open(file_path)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append({
                    "text": text,
                    "page": page_number
                })

        pdf.close()

    elif extension == ".txt":
        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "text": text,
                "page": None
            })

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not documents:
        raise ValueError(
            "No readable text found. The PDF may be scanned."
        )

    return documents