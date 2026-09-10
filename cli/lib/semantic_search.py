import json
import os
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


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
