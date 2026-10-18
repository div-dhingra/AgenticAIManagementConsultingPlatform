
import json
from pathlib import Path


def main():
    manifest_path = Path("data/manifest.json")

    with open(manifest_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print("\nSEC Document Metadata")
    print("-" * 65)

    for document in documents:
        if document["domain"] == "sec":
            print("Company:", document["company"])
            print("Year:", document["report_year"])
            print("Filing:", document["filing_type"])
            print("Pages:", document["page_count"])
            print("File:", Path(document["local_path"]).name)
            print("-" * 65)


if __name__ == "__main__":
    main()
