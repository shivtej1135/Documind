import json
from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


INPUT_DIR = Path("data/input")
IMAGE_DIR = Path("data/images")

OUTPUT_FILE = Path("data/visual_elements.json")


def main():

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    visual_elements = []

    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:

        print(f"Processing {pdf_file.name}")

        document = converter.convert(str(pdf_file)).document

        for figure_index, picture in enumerate(
            document.pictures,
            start=1,
        ):

            page_num = None

            if picture.prov:
                page_num = picture.prov[0].page_no

            image_file = (
                IMAGE_DIR
                / f"{pdf_file.stem}_p{page_num}_fig{figure_index}.png"
            )

            visual_elements.append(
                {
                    "doc_id": pdf_file.stem,
                    "page_num": page_num,
                    "type": "figure",
                    "image_path": str(image_file),
                    "caption": picture.caption_text(document),
                }
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            visual_elements,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved {len(visual_elements)} visual elements"
    )
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()