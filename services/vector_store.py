
import os

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from services.embeddings import generate_embeddings

MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.35"))
if not -1.0 <= MIN_SIMILARITY <= 1.0:
    raise ValueError("RAG_MIN_SIMILARITY must be between -1 and 1.")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="document_chunks",
    metadata={"hnsw:space": "cosine"}
)

RRF_K = 60


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


def search_chunks(document_id, question, top_k=5, min_similarity=MIN_SIMILARITY):
    """
    Combine vector similarity and TF-IDF keyword ranks for one document.
    """

    document_data = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"]
    )
    document_ids = document_data["ids"]
    documents = document_data["documents"] or []
    metadatas = document_data["metadatas"] or []
    if not document_ids or top_k <= 0:
        return []

    candidate_count = min(max(top_k * 4, 20), len(document_ids))
    query_embedding = generate_embeddings([question])[0]

    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"]
    )

    try:
        lexical_vectors = TfidfVectorizer().fit_transform([*documents, question])
        keyword_scores = cosine_similarity(
            lexical_vectors[-1],
            lexical_vectors[:-1]
        ).ravel()
    except ValueError:
        keyword_scores = [0.0] * len(document_ids)

    matches_by_id = {}

    if semantic_results["ids"]:
        for rank, chunk_id in enumerate(semantic_results["ids"][0], start=1):
            distance = semantic_results["distances"][0][rank - 1]
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue

            matches_by_id[chunk_id] = {
                "id": chunk_id,
                "text": semantic_results["documents"][0][rank - 1],
                "metadata": semantic_results["metadatas"][0][rank - 1],
                "distance": distance,
                "similarity": similarity,
                "keyword_score": 0.0,
                "hybrid_score": 1.0 / (RRF_K + rank)
            }

    lexical_ranks = sorted(
        range(len(document_ids)),
        key=lambda index: keyword_scores[index],
        reverse=True
    )
    for rank, index in enumerate(lexical_ranks[:candidate_count], start=1):
        keyword_score = float(keyword_scores[index])
        if keyword_score <= 0.0:
            break

        chunk_id = document_ids[index]
        match = matches_by_id.get(chunk_id)
        if match is None:
            match = {
                "id": chunk_id,
                "text": documents[index],
                "metadata": metadatas[index],
                "distance": None,
                "similarity": None,
                "keyword_score": keyword_score,
                "hybrid_score": 0.0
            }
            matches_by_id[chunk_id] = match
        else:
            match["keyword_score"] = keyword_score

        match["hybrid_score"] += 1.0 / (RRF_K + rank)

    matches = sorted(
        matches_by_id.values(),
        key=lambda match: match["hybrid_score"],
        reverse=True
    )
    return matches[:top_k]