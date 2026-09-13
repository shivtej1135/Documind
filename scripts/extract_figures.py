from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


INPUT_DIR = Path("data/input")
IMAGE_DIR = Path("data/images")


def main():
    # Enable image generation in Docling.
    pipeline_options = PdfPipelineOptions()

    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.generate_page_images = True

    # Create converter with image generation enabled.
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # Create the folder where all figure images will be saved.
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Find all PDF files in our input folder.
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        # Convert the PDF into Docling's structured document.
        document = converter.convert(str(pdf_file)).document

        print(f"Figures found: {len(document.pictures)}")

        for figure_index, picture in enumerate(
            document.pictures,
            start=1,
        ):
            # Get the page number from Docling's provenance.
            page_num = None

            if picture.prov:
                page_num = picture.prov[0].page_no

            # Get the figure caption linked by Docling.
            caption = picture.caption_text(document)

            # Get the actual cropped figure image.
            image = picture.get_image(document)

            if image is None:
                print(
                    f"  Figure {figure_index}: "
                    "image not available"
                )
                continue

            # Create the output filename.
            image_file = (
                IMAGE_DIR
                / f"{pdf_file.stem}_p{page_num}_fig{figure_index}.png"
            )

            # Save the figure image.
            image.save(image_file)

            print(
                f"  Figure {figure_index}: "
                f"page={page_num}, "
                f"saved={image_file.name}"
            )

            print(f"    Caption: {caption}")


if __name__ == "__main__":
    main()