
from consulting_platform.knowledge.extraction import (
    extract_all_pdfs,
)


def main():
    print("Starting PDF extraction...")

    extract_all_pdfs()

    print("PDF extraction finished.")


if __name__ == "__main__":
    main()
