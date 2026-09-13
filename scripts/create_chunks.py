import json
from pathlib import Path
import sys

# Add the project root so Python can import from src/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.chunking.chunker import DocumentChunker


INPUT_DIR = Path("data/output")
OUTPUT_FILE = Path("data/chunks.json")


def main():
    chunker = DocumentChunker()
    all_chunks = []

    # Read every structured JSON generated on Day 1.
    json_files = list(INPUT_DIR.glob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        print(f"Processing: {json_file.name}")

        with open(json_file, "r", encoding="utf-8") as f:
            document = json.load(f)

        chunks = chunker.chunk_document(document)

        all_chunks.extend(chunks)

        print(f"Created {len(chunks)} chunks")

    # Save all chunks in one file for the embedding step.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()