
from consulting_platform.knowledge.manifest import create_manifest


def main():
    documents = create_manifest()

    print("Manifest created successfully!")
    print(f"Total documents: {len(documents)}")

    for document in documents:
        print(
            f"{document['domain']}: "
            f"{document['title']} "
            f"({document['page_count']} pages)"
        )


if __name__ == "__main__":
    main()
