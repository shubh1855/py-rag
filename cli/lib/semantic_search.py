import json
import os
import re
from typing import Any

import numpy as np
from lib.search_utils import format_search_result
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict[str, Any]] | None = None
        self.document_map: dict[int, dict[str, Any]] = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace only.")

        embedding = self.model.encode([text])

        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]

        self.embeddings = self.model.encode(
            movie_strings,
            show_progress_bar=True,
        )

        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_path = "cache/movie_embeddings.npy"

        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            self.embeddings = embeddings

            if len(embeddings) == len(documents):
                return embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        results = []

        for embedding, document in zip(self.embeddings, self.documents):
            score = cosine_similarity(query_embedding, embedding)
            results.append((score, document))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in results[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[dict[str, int]] | None = None

    def build_chunk_embeddings(
        self,
        documents: list[dict],
    ) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(self.documents):
            description = doc["description"]

            if not description.strip():
                continue

            chunks = semantic_chunk(
                description,
                max_chunk_size=4,
                overlap=1,
            )

            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
        )

        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)

        np.save(
            "cache/chunk_embeddings.npy",
            self.chunk_embeddings,
        )

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self,
        documents: list[dict],
    ) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            embeddings = np.load(embeddings_path)
            self.chunk_embeddings = embeddings

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.chunk_metadata = metadata["chunks"]

            return embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        if self.documents is None:
            raise ValueError("No documents loaded.")

        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata[chunk_idx]

            score = cosine_similarity(
                query_embedding,
                chunk_embedding,
            )

            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": score,
                }
            )

        movie_scores = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]

            if (
                movie_idx not in movie_scores
                or score > movie_scores[movie_idx]["score"]
            ):
                movie_scores[movie_idx] = chunk_score

        sorted_movies = sorted(
            movie_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        top_movies = sorted_movies[:limit]

        results = []

        for movie_score in top_movies:
            movie_idx = movie_score["movie_idx"]
            score = movie_score["score"]

            document = self.documents[movie_idx]

            results.append(
                format_search_result(
                    id=document["id"],
                    title=document["title"],
                    document=document["description"][:100],
                    score=score,
                    metadata={},
                )
            )

        return results


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunk(
    text: str,
    max_chunk_size: int,
    overlap: int,
) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("Max chunk size must be greater than 0.")

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= max_chunk_size:
        raise ValueError("Overlap must be less than max chunk size.")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [sentence for sentence in sentences if sentence]

    chunks = []
    start = 0
    step = max_chunk_size - overlap

    while start < len(sentences):
        end = start + max_chunk_size
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)

        if end >= len(sentences):
            break

        start += step

    return chunks


def verify_model() -> None:
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text):
    search = SemanticSearch()

    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    search = SemanticSearch()

    embedding = search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    search = SemanticSearch()

    with open("data/movies.json", "r") as file:
        data = json.load(file)

    documents = data["movies"]

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} "
        f"vectors in {embeddings.shape[1]} dimensions"
    )
