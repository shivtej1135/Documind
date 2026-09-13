from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.parsing.docling_parser import DoclingParser


def main():
    parser = DoclingParser()

    pdf_path = Path("data/input/Simplebaseline.pdf")

    document = parser.converter.convert(str(pdf_path)).document
    print("TEXT ITEMS:", len(document.texts))
    print("TABLES:", len(document.tables))
    print("PICTURES:", len(document.pictures))

    if document.texts:
        print("\nFIRST TEXT ITEM:")
        print(document.texts[0])

    if document.tables:
        print("\nFIRST TABLE:")
        print(document.tables[0])

    if document.pictures:
        print("\nFIRST PICTURE:")
        print(document.pictures[0])

    # Print all available attributes/methods on the document
    print(dir(document))


if __name__ == "__main__":
    main()