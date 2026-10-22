from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        """Generate an embedding for an image."""
        with Image.open(image_path) as image:
            embedding = self.model.encode([image])[0]

        return embedding


def verify_image_embedding(image_path: str) -> None:
    """Generate an image embedding and print its dimensions."""
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")
