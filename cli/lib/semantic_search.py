import json
import os
import re
from typing import TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import Movie, SearchResult, format_search_result


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class SemanticSearch:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace only.")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(
        self,
        documents: list[Movie],
    ) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]

        embeddings = self.model.encode(
            movie_strings,
            show_progress_bar=True,
        )

        self.embeddings = embeddings

        os.makedirs("cache", exist_ok=True)

        np.save(
            "cache/movie_embeddings.npy",
            embeddings,
        )

        return embeddings

    def load_or_create_embeddings(
        self,
        documents: list[Movie],
    ) -> np.ndarray:
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

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        results: list[SearchResult] = []

        for embedding, document in zip(
            self.embeddings,
            self.documents,
        ):
            score = cosine_similarity(
                query_embedding,
                embedding,
            )

            results.append(
                format_search_result(
                    doc_id=document["id"],
                    title=document["title"],
                    document=document["description"],
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results[:limit]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(model_name)

        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def build_chunk_embeddings(
        self,
        documents: list[Movie],
    ) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []

        for movie_idx, doc in enumerate(documents):
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

        embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
        )

        self.chunk_embeddings = embeddings
        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)

        np.save(
            "cache/chunk_embeddings.npy",
            embeddings,
        )

        with open(
            "cache/chunk_metadata.json",
            "w",
        ) as file:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                file,
                indent=2,
            )

        return embeddings

    def load_or_create_chunk_embeddings(
        self,
        documents: list[Movie],
    ) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            embeddings = np.load(embeddings_path)
            self.chunk_embeddings = embeddings

            with open(metadata_path, "r") as file:
                metadata = json.load(file)

            self.chunk_metadata = metadata["chunks"]

            return embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. "
                "Call `load_or_create_chunk_embeddings` first."
            )

        if self.documents is None:
            raise ValueError("No documents loaded.")

        query_embedding = self.generate_embedding(query)

        movie_scores: dict[int, float] = {}

        for chunk_index, chunk_embedding in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata[chunk_index]

            score = cosine_similarity(
                query_embedding,
                chunk_embedding,
            )

            movie_idx = metadata["movie_idx"]

            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        sorted_movies = sorted(
            movie_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[SearchResult] = []

        for movie_idx, score in sorted_movies[:limit]:
            document = self.documents[movie_idx]

            results.append(
                format_search_result(
                    doc_id=document["id"],
                    title=document["title"],
                    document=document["description"],
                    score=score,
                )
            )

        return results


def cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray,
) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


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

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [text]

    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    chunks: list[str] = []

    start = 0
    step = max_chunk_size - overlap

    while start < len(sentences):
        chunk = " ".join(sentences[start : start + max_chunk_size]).strip()

        if chunk:
            chunks.append(chunk)

        end = start + max_chunk_size

        if end >= len(sentences):
            break

        start += step

    return chunks


def verify_model() -> None:
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str) -> None:
    search = SemanticSearch()

    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    search = SemanticSearch()

    embedding = search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings() -> None:
    search = SemanticSearch()

    with open("data/movies.json", "r") as file:
        data = json.load(file)

    documents: list[Movie] = data["movies"]

    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")

    print(
        f"Embeddings shape: "
        f"{embeddings.shape[0]} vectors "
        f"in {embeddings.shape[1]} dimensions"
    )
