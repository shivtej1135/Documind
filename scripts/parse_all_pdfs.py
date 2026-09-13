from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.parsing.docling_parser import DoclingParser
from src.parsing.exporter import save_json


INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")


def main():
    parser = DoclingParser()

    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        print(f"\nParsing: {pdf_file.name}")

        try:
            parsed_doc = parser.parse(pdf_file)

            output_file = OUTPUT_DIR / f"{pdf_file.stem}.json"

            save_json(parsed_doc, output_file)

            print(f"Saved: {output_file.name}")

        except Exception as e:
            print(f"Failed: {pdf_file.name}")
            print(e)


if __name__ == "__main__":
    main()