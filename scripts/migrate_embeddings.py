"""Idempotent migration of chunk-level JSON embeddings to per-document float32 .npy files."""

import argparse
import json
import logging
import os
import re
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def migrate_embeddings(embeddings_dir: str = "data/embeddings") -> int:
    """Convert chunk-level JSON embedding files to per-document .npy files.

    Args:
        embeddings_dir: Path to directory containing embedding files.

    Returns:
        Number of documents migrated.
    """
    if not os.path.exists(embeddings_dir):
        logger.info("Embeddings directory '%s' does not exist; nothing to migrate.", embeddings_dir)
        return 0

    pattern = re.compile(r"^doc_(\d+)_chunk_(\d+)\.json$")
    doc_chunks: dict[int, list[tuple[int, str]]] = defaultdict(list)

    for fn in os.listdir(embeddings_dir):
        match = pattern.match(fn)
        if match:
            doc_id = int(match.group(1))
            chunk_idx = int(match.group(2))
            doc_chunks[doc_id].append((chunk_idx, os.path.join(embeddings_dir, fn)))

    migrated_count = 0
    for doc_id, chunk_files in doc_chunks.items():
        npy_path = os.path.join(embeddings_dir, f"{doc_id}.npy")
        chunk_files.sort(key=lambda x: x[0])

        if os.path.exists(npy_path):
            try:
                existing = np.load(npy_path)
                if len(existing) == len(chunk_files):
                    logger.debug("Document %d already migrated; skipping.", doc_id)
                    continue
            except Exception:
                pass

        embeddings = []
        for _, path in chunk_files:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                embeddings.append(data["embedding"])

        emb_arr = np.array(embeddings, dtype=np.float32)
        np.save(npy_path, emb_arr)
        migrated_count += 1
        logger.info("Migrated document %d (%d chunks) to %s", doc_id, len(embeddings), npy_path)

    return migrated_count


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON embeddings to .npy format.")
    parser.add_argument("--path", default="data/embeddings", help="Path to embeddings directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    count = migrate_embeddings(args.path)
    print(f"Migration completed: {count} document(s) migrated.")


if __name__ == "__main__":
    main()
