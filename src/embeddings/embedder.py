from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Loads the embedding model and converts text into vectors.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        # Load the pretrained BGE model once and reuse it
        # for all chunks instead of loading it repeatedly.
        self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts: list[str]):
        """
        Convert a list of texts into normalized embedding vectors.
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )