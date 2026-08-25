"""Generate and persist chunk embeddings in the knowledge SQLite database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, update

from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.db_models import KnowledgeChunkRow


def _encode_texts(model: Any, tokenizer: Any, texts: list[str], *, batch_size: int = 8) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    batches = []
    for offset in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[offset : offset + batch_size],
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model(**encoded)
            mask = encoded["attention_mask"]
            if mask[:, -1].sum() == mask.shape[0]:
                vectors = output.last_hidden_state[:, -1]
            else:
                positions = mask.sum(dim=1) - 1
                vectors = output.last_hidden_state[
                    torch.arange(output.last_hidden_state.shape[0], device=model.device),
                    positions,
                ]
            vectors = functional.normalize(vectors, p=2, dim=1)
        batches.append(vectors.float().cpu().numpy())
    return np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)


def store_chunk_embeddings(
    chunks_path: str | Path,
    model_path: str | Path,
    database_url: str | None = None,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Encode chunk index text and atomically update matching SQLite rows."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    chunks = json.loads(Path(chunks_path).read_text(encoding="utf8"))
    if not isinstance(chunks, list) or not all(isinstance(item, dict) for item in chunks):
        raise ValueError("chunks JSON must be an array of objects")
    if len({str(item.get("chunk_id", "")) for item in chunks}) != len(chunks):
        raise ValueError("chunks JSON contains duplicate chunk IDs")

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, dtype=torch.float16, local_files_only=True
    ).cuda().eval()
    texts = [str(item.get("index_text") or item.get("text") or "") for item in chunks]
    vectors = _encode_texts(model, tokenizer, texts)
    dimension = int(vectors.shape[1])
    stored_model = model_name or str(model_path)

    factory = create_session_factory(database_url)
    with factory() as session:
        ids = set(session.scalars(select(KnowledgeChunkRow.id)))
        chunk_ids = {str(item["chunk_id"]) for item in chunks}
        missing = chunk_ids - ids
        if missing:
            raise ValueError(f"SQLite is missing {len(missing)} chunk rows")
        for item, vector in zip(chunks, vectors, strict=True):
            session.execute(
                update(KnowledgeChunkRow)
                .where(KnowledgeChunkRow.id == str(item["chunk_id"]))
                .values(
                    embedding=np.asarray(vector, dtype="<f4").tobytes(),
                    embedding_model=stored_model,
                    embedding_dimension=dimension,
                )
            )
        session.commit()

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "chunks": len(chunks),
        "dimension": dimension,
        "model": stored_model,
        "dtype": "float32",
        "storage": "knowledge_chunks.embedding BLOB",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Store chunk embeddings in SQLite")
    parser.add_argument("chunks", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--model-name")
    args = parser.parse_args()
    print(json.dumps(store_chunk_embeddings(
        args.chunks, args.model, args.database_url, model_name=args.model_name
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
