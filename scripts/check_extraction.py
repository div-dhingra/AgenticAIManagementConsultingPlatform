
import json
from pathlib import Path


def main():
    processed_dir = Path("data/processed")

    json_files = sorted(
        processed_dir.glob("*.json")
    )

    print("PDF Extraction Results")
    print("-" * 60)

    for file_path in json_files:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:
            document = json.load(file)

        pages = document["pages"]

        total_pages = len(pages)

        empty_pages = 0
        total_characters = 0

        for page in pages:
            text = page["text"]

            total_characters += len(text)

            if not text.strip():
                empty_pages += 1

        print("Document:", document["title"])
        print("Domain:", document["domain"])
        print("Pages:", total_pages)
        print("Empty pages:", empty_pages)
        print("Characters:", total_characters)
        print("-" * 60)


if __name__ == "__main__":
    main()
