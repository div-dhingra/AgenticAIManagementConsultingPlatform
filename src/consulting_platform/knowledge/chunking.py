
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_TOKENS = 220
OVERLAP_TOKENS = 30


def create_chunk_id(
    document_id: str,
    page_number: int,
    chunk_number: int,
) -> str:
    """Create a stable chunk ID."""

    value = (
        f"{document_id}-"
        f"{page_number}-"
        f"{chunk_number}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:24]



def split_text_into_chunks(
    text: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split text and verify the final token length."""

    if not text.strip():
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Invalid overlap_tokens.")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        verbose=False,
    )

    special_tokens = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    available_tokens = max_tokens - special_tokens

    if available_tokens <= overlap_tokens:
        raise ValueError("Token limit is too small.")

    chunks = []
    start = 0

    while start < len(token_ids):

        end = min(
            start + available_tokens,
            len(token_ids),
        )

        # Decode and check the actual token count.
        while end > start:

            current_tokens = token_ids[start:end]

            chunk_text = tokenizer.decode(
                current_tokens,
                skip_special_tokens=True,
            ).strip()

            actual_token_count = len(
                tokenizer.encode(
                    chunk_text,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )
            )

            if actual_token_count <= max_tokens:
                break

            # Reduce the chunk until it fits.
            end -= 1

        if end <= start:
            raise ValueError(
                "Unable to create a valid chunk."
            )

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(token_ids):
            break

        # Keep overlap between neighboring chunks.
        start = max(start + 1, end - overlap_tokens)

    return chunks



def chunk_document(
    document: dict,
    tokenizer,
) -> list[dict]:
    """Create chunks from one extracted PDF."""

    all_chunks = []

    for page in document["pages"]:
        page_number = page["page"]
        page_text = page["text"]

        text_chunks = split_text_into_chunks(
            page_text,
            tokenizer,
        )

        for chunk_number, chunk_text in enumerate(
            text_chunks,
            start=1,
        ):
            chunk = {
                "chunk_id": create_chunk_id(
                    document["document_id"],
                    page_number,
                    chunk_number,
                ),
                "document_id": document["document_id"],
                "title": document["title"],
                "domain": document["domain"],
                "company": document["company"],
                "report_year": document["report_year"],
                "source_uri": document["source_uri"],
                "page": page_number,
                "printed_page": page["printed_page"],
                "text": chunk_text,
            }

            all_chunks.append(chunk)

    return all_chunks


def chunk_all_documents(
    processed_dir: str = "data/processed",
    output_file: str = "data/chunks.jsonl",
) -> None:
    """Chunk all extracted documents into one JSONL file."""

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    processed_path = Path(processed_dir)
    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_chunks = 0

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output:

        for file_path in sorted(
            processed_path.glob("*.json")
        ):
            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as file:
                document = json.load(file)

            chunks = chunk_document(
                document,
                tokenizer,
            )

            for chunk in chunks:
                output.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False,
                    ) + "\n"
                )

            total_chunks += len(chunks)

            print(
                f"{document['title']}: "
                f"{len(chunks)} chunks"
            )

    print(f"Total chunks: {total_chunks}")
    print(f"Saved to: {output_path}")
