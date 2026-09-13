import sys
from pathlib import Path

# Add the project root so Python can import our retrieval module.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.retrieval.faiss_store import FAISSStore


def main():
    # Build the FAISS index from our embedded chunks.
    store = FAISSStore()
    store.build()


if __name__ == "__main__":
    main()