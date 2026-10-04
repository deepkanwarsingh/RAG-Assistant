
def chunk_documents(
    pages,
    chunk_size=1000,
    overlap=150
):
    """
    Split extracted text into overlapping chunks.
    Preserve the source page number.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for page_data in pages:
        text = page_data["text"]
        page_number = page_data["page"]

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks