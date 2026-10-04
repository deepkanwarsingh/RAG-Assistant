
import chromadb
from services.embeddings import generate_embeddings

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="document_chunks",
    metadata={"hnsw:space": "cosine"}
)


def store_chunks(document_id, chunks):
    """
    Embed and store chunks for one document.
    """

    if not chunks:
        return 0

    ids = []
    texts = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        ids.append(f"{document_id}_{index}")
        texts.append(chunk["text"])

        metadatas.append({
            "document_id": document_id,
            "chunk_index": index,
            "page": chunk["page"] or 0
        })

    embeddings = generate_embeddings(texts)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def search_chunks(document_id, question, top_k=5):
    """
    Retrieve the most relevant chunks for one document.
    """

    query_embedding = generate_embeddings([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    if not results["ids"]:
        return matches

    for i, chunk_id in enumerate(results["ids"][0]):
        matches.append({
            "id": chunk_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return matches