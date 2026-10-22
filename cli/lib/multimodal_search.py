import numpy as np
from inverted_index import load_movies
from lib.search_utils import Movie
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ImageSearchResult(dict[str, int | str | float]):
    """Represents a movie returned by image search."""


class MultimodalSearch:
    def __init__(
        self,
        documents: list[Movie],
        model_name: str = "clip-ViT-B-32",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents

        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]

        self.text_embeddings: NDArray[np.float32] = np.asarray(
            self.model.encode(
                self.texts,
                show_progress_bar=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def embed_image(self, image_path: str) -> NDArray[np.float32]:
        """Generate an embedding for an image."""
        with Image.open(image_path) as image:
            embedding = self.model.encode(
                [image],
                convert_to_numpy=True,
            )[0]

        return np.asarray(embedding, dtype=np.float32)

    def search_with_image(
        self,
        image_path: str,
    ) -> list[dict[str, int | str | float]]:
        """Find the five movies most similar to an image."""
        if not self.documents:
            return []

        image_embedding = self.embed_image(image_path)

        similarities = cosine_similarity(
            image_embedding.reshape(1, -1),
            self.text_embeddings,
        )[0]

        results: list[dict[str, int | str | float]] = []

        for index, similarity in enumerate(similarities):
            document = self.documents[index]

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "description": document["description"],
                    "similarity": float(similarity),
                }
            )

        results.sort(
            key=lambda result: float(result["similarity"]),
            reverse=True,
        )

        return results[:5]


def verify_image_embedding(image_path: str) -> None:
    """Generate an image embedding and print its dimensions."""
    model = SentenceTransformer("clip-ViT-B-32")

    with Image.open(image_path) as image:
        embedding = model.encode(
            [image],
            convert_to_numpy=True,
        )[0]

    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(
    image_path: str,
) -> list[dict[str, int | str | float]]:
    """Load movies and search using an image."""
    movies = load_movies()
    searcher = MultimodalSearch(movies)

    return searcher.search_with_image(image_path)
