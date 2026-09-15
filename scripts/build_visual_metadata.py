
import json
from pathlib import Path

import fitz

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

    # Tell Docling to generate images for detected figures.
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

        document = converter.convert(
            str(pdf_file)
        ).document

        # Open the original PDF with PyMuPDF.
        # We use it to crop table regions using
        # the bounding boxes provided by Docling.
        pdf = fitz.open(pdf_file)

        # -------------------------
        # FIGURES
        # -------------------------

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

        # -------------------------
        # TABLES
        # -------------------------

        for table_index, table in enumerate(
            document.tables,
            start=1,
        ):

            if not table.prov:
                print(
                    f"  Table {table_index}: "
                    "no provenance information"
                )
                continue

            provenance = table.prov[0]

            page_num = provenance.page_no
            bbox = provenance.bbox

            print(
                f"  Table {table_index}: "
                f"page={page_num}"
            )

            # Docling page numbers start at 1,
            # while PyMuPDF page indexes start at 0.
            page = pdf[page_num - 1]

            # Docling uses bottom-left coordinates.
            # PyMuPDF uses top-left coordinates.
            page_height = page.rect.height

            x0 = bbox.l
            x1 = bbox.r

            y0 = page_height - bbox.t
            y1 = page_height - bbox.b

            crop_rect = fitz.Rect(
                x0,
                y0,
                x1,
                y1,
            )

            # Render the cropped table at 2x resolution
            # so the vision model gets a clear image.
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                clip=crop_rect,
                alpha=False,
            )

            image_file = (
                IMAGE_DIR
                / f"{pdf_file.stem}_p{page_num}_table{table_index}.png"
            )

            IMAGE_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            pixmap.save(image_file)

            visual_elements.append(
                {
                    "doc_id": pdf_file.stem,
                    "page_num": page_num,
                    "type": "table",
                    "image_path": str(image_file),
                    "caption": table.caption_text(document),
                }
            )

        pdf.close()

    # Save both figures and tables into the same
    # visual metadata file used by ImageRouter.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

