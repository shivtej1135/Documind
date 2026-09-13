import json
from pathlib import Path
import sys

# Add the project root so we can import our embedding module.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.embeddings.embedder import EmbeddingModel


INPUT_FILE = Path("data/chunks.json")
OUTPUT_FILE = Path("data/embedded_chunks.json")


def main():
    # Load all chunks generated during Day 2 Step 1.
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Extract only the text because that is what the embedding
    # model converts into numerical vectors.
    texts = [chunk["text"] for chunk in chunks]

    # Load the BGE model once and embed all chunk texts.
    model = EmbeddingModel()
    embeddings = model.encode(texts)

    # Make sure no chunk was silently dropped.
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match "
            f"embedding count ({len(embeddings)})"
        )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        # Keep the original chunk metadata and attach its vector.
        embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding.tolist(),
            }
        )

    # Save the embedded chunks for the FAISS step.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            embedded_chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved {len(embedded_chunks)} embedded chunks")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()