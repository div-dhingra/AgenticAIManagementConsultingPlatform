
from consulting_platform.knowledge.chunking import (
    chunk_all_documents,
)


def main():
    print("Starting document chunking...")

    chunk_all_documents()

    print("Chunking completed.")


if __name__ == "__main__":
    main()
