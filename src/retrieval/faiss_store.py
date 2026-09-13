import json
from pathlib import Path

import faiss
import numpy as np


EMBEDDED_CHUNKS_FILE = Path("data/embedded_chunks.json")
INDEX_FILE = Path("data/faiss.index")
METADATA_FILE = Path("data/faiss_metadata.json")


class FAISSStore:
    """
    Builds and saves a FAISS similarity-search index for our
    embedded DOCUMIND chunks.
    """

    def build(self):
        # Load the chunks together with their embedding vectors.
        with open(EMBEDDED_CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            raise ValueError("No embedded chunks found.")

        # Convert the list of embeddings into a NumPy array.
        embeddings = np.array(
            [chunk["embedding"] for chunk in chunks],
            dtype="float32",
        )

        # Use inner product because our embeddings were normalized.
        index = faiss.IndexFlatIP(embeddings.shape[1])

        # Add every chunk vector to the FAISS index.
        index.add(embeddings)

        # Keep the original chunk records so a FAISS result
        # can be mapped back to its document/page/text.
        metadata = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "page_num": chunk["page_num"],
                "chunk_type": chunk["chunk_type"],
                "section_heading": chunk["section_heading"],
                "text": chunk["text"],
            }
            for chunk in chunks
        ]

        # Save the FAISS index.
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_FILE))

        # Save the chunk-to-FAISS-position mapping.
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Indexed chunks: {index.ntotal}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"FAISS index: {INDEX_FILE}")
        print(f"Metadata: {METADATA_FILE}")