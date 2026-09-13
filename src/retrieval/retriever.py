import json
from pathlib import Path

import faiss

from src.embeddings.embedder import EmbeddingModel


INDEX_FILE = Path("data/faiss.index")
METADATA_FILE = Path("data/faiss_metadata.json")


class Retriever:
    """
    Finds the most relevant document chunks for a question
    using the BGE embedding model and FAISS.
    """

    def __init__(self):
        # Load the same embedding model used when indexing chunks.
        self.embedding_model = EmbeddingModel()

        # Load the saved FAISS vector index.
        self.index = faiss.read_index(str(INDEX_FILE))

        # Load the chunk information corresponding to each
        # position inside the FAISS index.
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Convert the question into an embedding and return
        the top-k most similar chunks.
        """
        # Convert the question into the same 384-dimensional
        # embedding space used for the document chunks.
        query_vector = self.embedding_model.encode([question])

        # Search FAISS for the closest vectors.
        scores, indices = self.index.search(query_vector, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            # FAISS uses -1 when a result is unavailable.
            if index == -1:
                continue

            chunk = self.metadata[index].copy()

            # Keep the similarity score so we can inspect
            # how strongly the chunk matched the question.
            chunk["score"] = float(score)

            results.append(chunk)

        return results