
import json
from pathlib import Path

import pymupdf


def extract_pdf(document: dict) -> dict:
    """
    Extract text from each page of a PDF.
    Preserve document metadata and physical page numbers.
    """

    pdf_path = Path(document["local_path"])

    pages = []

    with pymupdf.open(pdf_path) as pdf:

        for page_index in range(len(pdf)):
            page = pdf[page_index]

            text = page.get_text(
                "text",
                # Coordinate sorting interleaves SOFI's two newspaper columns,
                # making otherwise contiguous quotations impossible to recover.
                sort=document["domain"] != "corporate_responsibility",
            )

            page_data = {
                "page": page_index + 1,
                "printed_page": None,
                "text": text.strip(),
            }

            pages.append(page_data)

    result = {
        "document_id": document["document_id"],
        "title": document["title"],
        "domain": document["domain"],
        "company": document["company"],
        "report_year": document["report_year"],
        "filing_type": document["filing_type"],
        "source_uri": (
            document["original_url"]
            if document["original_url"]
            else document["local_path"]
        ),
        "pages": pages,
    }

    return result


def extract_all_pdfs(
    manifest_path: str = "data/manifest.json",
    output_dir: str = "data/processed",
) -> None:
    """
    Extract all PDF documents listed in the manifest.
    Save each document as a JSON file.
    """

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        documents = json.load(file)

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for document in documents:

        try:
            extracted = extract_pdf(document)

            file_name = (
                document["document_id"] + ".json"
            )

            save_path = output_path / file_name

            with open(
                save_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    extracted,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"Extracted: {document['title']} "
                f"({len(extracted['pages'])} pages)"
            )

        except Exception as error:
            print(
                f"Failed: {document['title']} - {error}"
            )
