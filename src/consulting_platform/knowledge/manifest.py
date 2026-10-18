
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from consulting_platform.knowledge.metadata import (
    identify_company,
    identify_filing_type,
    identify_report_year,
)

DOMAIN_NAMES = [
    "technology",
    "clinical",
    "sec",
    "corporate_responsibility",
]


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def inspect_pdf(file_path: Path) -> dict:
    """Read basic PDF information."""

    with pymupdf.open(file_path) as pdf:
        title = pdf.metadata.get("title") or file_path.stem
        page_count = len(pdf)

    return {
        "title": title.strip() or file_path.stem,
        "page_count": page_count,
    }


def create_manifest(
    data_dir: str = "data",
) -> list[dict]:
    """Create a manifest for locally imported PDF files."""

    data_path = Path(data_dir)
    raw_path = data_path / "raw"

    registry_path = data_path / "source_registry.json"
    source_registry = {}

    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8-sig") as file:
            source_registry = json.load(file)

    documents = []

    for domain in DOMAIN_NAMES:
        domain_path = raw_path / domain

        if not domain_path.exists():
            continue

        for pdf_path in sorted(domain_path.rglob("*.pdf")):
            relative_path = pdf_path.relative_to(raw_path).as_posix()
            source_info = source_registry.get(relative_path, {})
            file_hash = calculate_sha256(pdf_path)

            try:
                pdf_info = inspect_pdf(pdf_path)

                document = {
                    "document_id": file_hash[:16],
                    "title": pdf_info["title"],
                    "domain": domain,
                    "company": (
                        identify_company(pdf_path)
                    ),
                    "report_year": (
                        identify_report_year(pdf_path)
                        if domain in {"sec", "corporate_responsibility"}
                        else None
                    ),
                    "filing_type": (
                        identify_filing_type(pdf_path)
                        if domain == "sec"
                        else None
                    ),
                    "local_path": pdf_path.as_posix(),
                    "original_url": source_info.get("original_url"),
                    "import_date": datetime.now(
                        timezone.utc
                    ).date().isoformat(),
                    "sha256": file_hash,
                    "content_type": "application/pdf",
                    "coverage": "Metadata pending review",
                    "limitations": [
                        "Company and report year require verification."
                    ],
                    "ingestion_status": "imported",
                    "page_count": pdf_info["page_count"],
                }

                documents.append(document)

            except Exception as error:
                print(
                    f"Failed to inspect {pdf_path.name}: {error}"
                )

    manifest_path = data_path / "manifest.json"
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return documents
