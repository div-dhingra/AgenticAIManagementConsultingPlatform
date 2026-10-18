from pathlib import Path

from consulting_platform.knowledge.index import build_index


def main():
    project_root = Path(__file__).resolve().parents[1]

    chunks_path = project_root / "data" / "chunks.jsonl"
    output_dir = project_root / "data" / "index"

    build_index(chunks_path, output_dir)


if __name__ == "__main__":
    main()