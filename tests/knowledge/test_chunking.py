
import json
from pathlib import Path

from transformers import AutoTokenizer

from consulting_platform.knowledge.chunking import (
    MAX_TOKENS,
    MODEL_NAME,
    create_chunk_id,
    split_text_into_chunks,
)


def test_stable_chunk_id():
    """Same input should produce the same chunk ID."""

    first_id = create_chunk_id(
        "document123", 1, 1
    )

    second_id = create_chunk_id(
        "document123", 1, 1
    )

    assert first_id == second_id


def test_different_chunk_ids():
    """Different chunk positions need different IDs."""

    first_id = create_chunk_id(
        "document123", 1, 1
    )

    second_id = create_chunk_id(
        "document123", 1, 2
    )

    assert first_id != second_id


def test_empty_text():
    """Empty text should not generate chunks."""

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    result = split_text_into_chunks(
        "",
        tokenizer,
    )

    assert result == []


def test_token_limit():
    """Every chunk must respect the token limit."""

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    text = "Cisco router technical specifications. " * 100

    chunks = split_text_into_chunks(
        text,
        tokenizer,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        token_count = len(
            tokenizer.encode(
                chunk,
                add_special_tokens=True,
                truncation=False,
            )
        )

        assert token_count <= MAX_TOKENS


def test_generated_chunks():
    """Check the actual chunks.jsonl file."""

    file_path = Path("data/chunks.jsonl")

    assert file_path.exists()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    chunk_ids = set()
    chunk_count = 0

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            chunk = json.loads(line)

            assert chunk["chunk_id"] not in chunk_ids

            chunk_ids.add(chunk["chunk_id"])

            assert chunk["document_id"]
            assert chunk["text"].strip()
            assert chunk["page"] >= 1

            token_count = len(
                tokenizer.encode(
                    chunk["text"],
                    add_special_tokens=True,
                    truncation=False,
                )
            )

            assert token_count <= MAX_TOKENS, (
                f"Chunk {chunk['chunk_id']} "
                f"has {token_count} tokens"
            )

            chunk_count += 1

    assert chunk_count > 0
