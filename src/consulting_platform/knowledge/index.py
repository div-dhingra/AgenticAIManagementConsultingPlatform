import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(chunks_path):
    """
    Read chunk records from a JSONL file.
    """
    chunks = []

    with open(chunks_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunk = json.loads(line)
                chunks.append(chunk)

    return chunks


def build_index(chunks_path, output_dir):
    """
    Build semantic and lexical retrieval indexes.
    """

    chunks_path = Path(chunks_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading chunks...")
    chunks = load_chunks(chunks_path)

    if not chunks:
        raise ValueError("No chunks found.")

    print(f"Loaded {len(chunks)} chunks.")

    texts = []

    for chunk in chunks:
        texts.append(chunk["text"])

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    print("Saving index files...")

    np.save(output_dir / "embeddings.npy", embeddings)

    joblib.dump(
        vectorizer,
        output_dir / "tfidf_vectorizer.joblib",
    )

    joblib.dump(
        tfidf_matrix,
        output_dir / "tfidf_matrix.joblib",
    )

    with open(output_dir / "chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False)

    print("Index build completed successfully.")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    print(f"Saved to: {output_dir}")