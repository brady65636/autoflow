"""Offline dense + BM25 + RRF + Qwen reranker evaluation for one corpus.

Heavy ML imports are intentionally local to this CLI so the regular backend and
unit tests do not require a CUDA environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any

from autoflow_scheduling.observability import get_tracer
from autoflow_scheduling.observability.monitoring_context import file_version

from .retrieval_profile import (
    ContentCompatibility,
    DocumentContentType,
    QuestionType,
    SearchProfile,
    build_query_profile_text,
    content_type_compatibility,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_EMBED_INSTRUCTION = (
    "Given an automotive technical query, retrieve passages that directly contain "
    "the requested knowledge. Pay attention to the question type, required knowledge, "
    "section title, and subheading."
)
_RERANK_INSTRUCTION = (
    "Judge whether the automotive Document directly contains the knowledge requested "
    "by the Query. Prefer an exact section and subheading match over broad topic overlap."
)
_LLM_RERANK_SYSTEM = (
    "Judge whether the document directly answers the query. "
    "Output exactly one relevance label: DIRECT, RELATED, or IRRELEVANT."
)
_LABEL_RE = re.compile(r"\b(DIRECT|RELATED|IRRELEVANT)\b", re.IGNORECASE)
_LLM_LABEL_SCORES = {"DIRECT": 1.0, "RELATED": 0.5, "IRRELEVANT": 0.0}


class RerankerMode(str, Enum):
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _language(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        sequence_lengths,
    ]


def _encode_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as functional

    batches = []
    for offset in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[offset : offset + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model(**encoded)
            vectors = _last_token_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors = functional.normalize(vectors, p=2, dim=1)
        batches.append(vectors.float().cpu().numpy())
    return np.concatenate(batches, axis=0)


def _rrf(
    dense_order: list[int],
    bm25_order: list[int],
    limit: int,
    k: int = 60,
    *,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[int]:
    if dense_weight < 0 or bm25_weight < 0 or dense_weight == bm25_weight == 0:
        raise ValueError("RRF weights must be non-negative and not both zero")
    scores: defaultdict[int, float] = defaultdict(float)
    for weight, order in (
        (dense_weight, dense_order[:limit]),
        (bm25_weight, bm25_order[:limit]),
    ):
        for rank, index in enumerate(order, 1):
            scores[index] += weight / (k + rank)
    return [index for index, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _dense_order(document_embeddings: Any, query_vector: Any) -> list[int]:
    scores = document_embeddings @ query_vector
    return sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)


def _bm25_order(bm25: Any, query: str) -> list[int]:
    scores = bm25.get_scores(_tokens(query))
    return sorted(
        (index for index, score in enumerate(scores) if float(score) > 0),
        key=lambda index: float(scores[index]),
        reverse=True,
    )


def _candidate_summary(
    order: list[int],
    chunks: list[dict[str, Any]],
    scores: dict[int, float] | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return bounded IDs/ranks only; never include query or document text."""
    return [
        {
            "chunk_id": str(chunks[index].get("chunk_id", index)),
            "rank": rank,
            **({"score": round(scores[index], 6)} if scores and index in scores else {}),
        }
        for rank, index in enumerate(order[:limit], 1)
    ]


def _rank_deltas(
    before: list[int], after: list[int], chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    before_ranks = {
        chunks[index].get("chunk_id", index): rank
        for rank, index in enumerate(before, 1)
    }
    after_ranks = {
        chunks[index].get("chunk_id", index): rank
        for rank, index in enumerate(after, 1)
    }
    return [
        {
            "chunk_id": str(chunk_id),
            "rank_delta": after_ranks.get(chunk_id, 21) - before_ranks.get(chunk_id, 21),
        }
        for chunk_id in list(dict.fromkeys([*before_ranks, *after_ranks]))[:20]
    ]


def _timed_call(function: Any, *args: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    return function(*args), time.perf_counter() - started


def _business_rule_score(case: dict[str, Any], chunk: dict[str, Any]) -> tuple[float, list[str]]:
    """Return a bounded soft business-rule score for a rerank candidate."""
    reasons: list[str] = []
    try:
        question_type = QuestionType(str(case["question_type"]))
        content_type = DocumentContentType(str(chunk.get("document_content_type", "other")))
    except (KeyError, ValueError):
        return 0.0, ["unknown_profile"]

    compatibility = content_type_compatibility(question_type, content_type)
    compatibility_score = {
        ContentCompatibility.PRIMARY: 1.0,
        ContentCompatibility.SUPPORTING: 0.5,
        ContentCompatibility.NONE: 0.0,
    }[compatibility]
    reasons.append(f"compatibility={compatibility.value}")

    confidence = max(0.0, min(1.0, float(chunk.get("metadata_confidence", 0.0))))
    quality = (chunk.get("quality") or {}).get("rag_text_status", "unknown")
    quality_score = {"pass": 1.0, "warning": 0.75, "quarantine": 0.0}.get(quality, 0.5)
    reasons.append(f"quality={quality}")

    score = 0.6 * compatibility_score + 0.2 * confidence + 0.2 * quality_score
    return round(score, 6), reasons


def _parallel_hybrid_orders(
    document_embeddings: Any,
    query_vector: Any,
    bm25: Any,
    query: str,
    *,
    dense_limit: int,
    bm25_limit: int,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> dict[str, Any]:
    """Run independent dense and lexical retrieval concurrently, then fuse with RRF."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as executor:
        dense_future = executor.submit(
            _timed_call, _dense_order, document_embeddings, query_vector
        )
        bm25_future = executor.submit(_timed_call, _bm25_order, bm25, query)
        dense_order, dense_seconds = dense_future.result()
        bm25_order, bm25_seconds = bm25_future.result()
    rrf_started = time.perf_counter()
    rrf_order = _rrf(
        dense_order,
        bm25_order,
        max(dense_limit, bm25_limit),
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    return {
        "dense": dense_order,
        "bm25": bm25_order,
        "rrf": rrf_order,
        "dense_seconds": dense_seconds,
        "bm25_seconds": bm25_seconds,
        "rrf_seconds": time.perf_counter() - rrf_started,
        "parallel_seconds": time.perf_counter() - started,
    }


def _format_reranker_input(query: str, document: str) -> str:
    return f"<Instruct>: {_RERANK_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"


def _rerank_scores(
    model: Any,
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    *,
    batch_size: int,
    max_length: int,
) -> list[float]:
    import torch

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements based on '
        'the Query and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    scores: list[float] = []

    for offset in range(0, len(pairs), batch_size):
        texts = [_format_reranker_input(*pair) for pair in pairs[offset : offset + batch_size]]
        encoded = tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
            return_attention_mask=False,
        )
        encoded["input_ids"] = [
            prefix_tokens + token_ids + suffix_tokens for token_ids in encoded["input_ids"]
        ]
        batch = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :]
            binary_logits = torch.stack(
                [logits[:, token_false_id], logits[:, token_true_id]], dim=1
            )
            probabilities = torch.softmax(binary_logits, dim=1)[:, 1]
        scores.extend(float(value) for value in probabilities.cpu())
    return scores


def _llm_rerank_scores(
    model: Any,
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    *,
    batch_size: int,
    max_length: int,
) -> list[float]:
    """Ask an instruction-tuned LLM for a bounded relevance score per pair."""
    import torch

    scores: list[float] = []
    for offset in range(0, len(pairs), batch_size):
        prompts = []
        for query, document in pairs[offset : offset + batch_size]:
            messages = [
                {"role": "system", "content": _LLM_RERANK_SYSTEM},
                {
                    "role": "user",
                    "content": f"Query:\n{query}\n\nDocument:\n{document}",
                },
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        outputs = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        for output in outputs:
            match = _LABEL_RE.search(output)
            if match is None:
                raise ValueError(f"LLM reranker returned an invalid label: {output!r}")
            scores.append(_LLM_LABEL_SCORES[match.group(1).upper()])
    return scores


def _section_key(item: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(item.get("section_title", item.get("title", ""))), tuple(
        item.get("section_path", item.get("path", []))
    )


def _expected(item: dict[str, Any], case: dict[str, Any]) -> bool:
    title, path = _section_key(item)
    expected_sections = case.get("expected_sections") or [
        {
            "title": case["expected_title"],
            "path_contains": case.get("expected_path_contains", []),
        }
    ]
    return any(
        title == expected["title"]
        and all(part in path for part in expected.get("path_contains", []))
        for expected in expected_sections
    )


def _section_order(chunk_order: list[int], chunks: list[dict[str, Any]]) -> list[int]:
    seen: set[str] = set()
    result = []
    for index in chunk_order:
        section_id = str(chunks[index]["section_id"])
        if section_id in seen:
            continue
        seen.add(section_id)
        result.append(index)
    return result


def _rank(order: list[int], chunks: list[dict[str, Any]], case: dict[str, Any]) -> int | None:
    for rank, index in enumerate(_section_order(order, chunks), 1):
        if _expected(chunks[index], case):
            return rank
    return None


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    total = max(1, len(ranks))
    return {
        "hit@1": round(sum(rank == 1 for rank in ranks) / total, 4),
        "hit@3": round(sum(rank is not None and rank <= 3 for rank in ranks) / total, 4),
        "hit@5": round(sum(rank is not None and rank <= 5 for rank in ranks) / total, 4),
        "hit@10": round(sum(rank is not None and rank <= 10 for rank in ranks) / total, 4),
        "mrr": round(sum(1 / rank for rank in ranks if rank) / total, 4),
    }


def _evaluate(
    chunks_path: Path,
    sections_path: Path,
    cases_path: Path,
    output_path: Path,
    embedding_model_path: Path,
    reranker_model_path: Path | None = None,
    *,
    reranker_mode: RerankerMode | str = RerankerMode.CROSS_ENCODER,
    llm_model_path: Path | None = None,
    dense_limit: int = 50,
    bm25_limit: int = 50,
    rerank_candidates: int = 30,
    final_chunk_limit: int = 15,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    business_rule_weight: float = 0.2,
) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from rank_bm25 import BM25Okapi
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    mode = RerankerMode(reranker_mode)
    if final_chunk_limit <= 0:
        raise ValueError("final_chunk_limit must be positive")
    if dense_weight < 0 or bm25_weight < 0 or dense_weight == bm25_weight == 0:
        raise ValueError("RRF weights must be non-negative and not both zero")
    if not 0 <= business_rule_weight <= 1:
        raise ValueError("business_rule_weight must be between 0 and 1")
    if mode is RerankerMode.CROSS_ENCODER and reranker_model_path is None:
        raise ValueError("reranker_model_path is required for cross_encoder mode")
    if mode is RerankerMode.LLM and llm_model_path is None:
        raise ValueError("llm_model_path is required for llm mode")

    chunks = json.loads(chunks_path.read_text(encoding="utf8"))
    sections = json.loads(sections_path.read_text(encoding="utf8"))
    cases = json.loads(cases_path.read_text(encoding="utf8"))
    section_lookup = {str(section["section_id"]): section for section in sections}
    document_texts = [str(chunk.get("index_text") or chunk["text"]) for chunk in chunks]
    query_profiles = [
        SearchProfile(
            query=case["query"],
            question_type=QuestionType(case["question_type"]),
            required_knowledge=case["required_knowledge"],
        )
        for case in cases
    ]
    query_texts = [build_query_profile_text(profile) for profile in query_profiles]
    instructed_queries = [
        f"Instruct: {_EMBED_INSTRUCTION}\nQuery: {query_text}" for query_text in query_texts
    ]

    tracer = get_tracer()
    timings: dict[str, float] = {}
    started = time.perf_counter()
    with tracer.stage("embedding", metadata={"cases": len(cases), "chunks": len(chunks)}) as stage:
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_path, padding_side="left", local_files_only=True
        )
        embedding_model = AutoModel.from_pretrained(
            embedding_model_path, dtype=torch.float16, local_files_only=True
        ).cuda().eval()
        timings["embedding_model_load_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        document_embeddings = _encode_texts(
            embedding_model, embedding_tokenizer, document_texts,
            batch_size=8, max_length=2048,
        )
        query_embeddings = _encode_texts(
            embedding_model,
            embedding_tokenizer,
            instructed_queries,
            batch_size=8,
            max_length=512,
        )
        timings["embedding_seconds"] = time.perf_counter() - started
        tracer.update(stage, status="complete", dimensions=int(document_embeddings.shape[1]))

    del embedding_model, embedding_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    bm25 = BM25Okapi([_tokens(text) for text in document_texts])
    stage_orders: list[dict[str, Any]] = []
    started = time.perf_counter()
    with tracer.stage("parallel_retrieval", metadata={"cases": len(cases)}) as parent:
        for case_index, (case, query_vector) in enumerate(
            zip(cases, query_embeddings, strict=True)
        ):
            case_id = str(case.get("case_id", case.get("id", case_index)))
            with tracer.stage(
                "parallel_case",
                metadata={
                    "case_id": case_id,
                    "question_type": case["question_type"],
                    "language": case.get("language", _language(case["query"])),
                },
            ) as case_stage:
                orders = _parallel_hybrid_orders(
                    document_embeddings, query_vector, bm25, case["query"],
                    dense_limit=dense_limit, bm25_limit=bm25_limit,
                    dense_weight=dense_weight, bm25_weight=bm25_weight,
                )
                stage_orders.append(orders)
                for route in ("dense", "bm25", "rrf"):
                    with tracer.stage(
                        route,
                        metadata={"case_id": case_id, "candidate_count": len(orders[route])},
                    ) as route_stage:
                        tracer.update(
                            route_stage,
                            status="complete",
                            seconds=orders[f"{route}_seconds"],
                            candidates=_candidate_summary(orders[route], chunks),
                        )
                tracer.update(
                    case_stage,
                    status="complete",
                    candidate_count=len(orders["rrf"]),
                    seconds=orders["parallel_seconds"],
                )
        timings["hybrid_retrieval_seconds"] = time.perf_counter() - started
        tracer.update(parent, status="complete", cases=len(stage_orders))

    reranker_tokenizer = None
    reranker_model = None
    reranker_load_error = None
    if mode is not RerankerMode.NONE:
        model_path = (
            reranker_model_path if mode is RerankerMode.CROSS_ENCODER else llm_model_path
        )
        with tracer.stage(
            "reranker_model_load", metadata={"mode": mode.value}
        ) as load_stage:
            started = time.perf_counter()
            load_error_type = None
            try:
                reranker_tokenizer = AutoTokenizer.from_pretrained(
                    model_path, padding_side="left", local_files_only=True
                )
                reranker_model = AutoModelForCausalLM.from_pretrained(
                    model_path, dtype=torch.float16, local_files_only=True
                ).cuda().eval()
            except Exception as error:
                load_error_type = type(error).__name__
                reranker_load_error = f"{load_error_type}: {str(error)[:240]}"
                reranker_tokenizer = None
                reranker_model = None
            finally:
                timings["reranker_model_load_seconds"] = time.perf_counter() - started
                tracer.update(
                    load_stage,
                    status="error" if reranker_load_error else "complete",
                    seconds=timings["reranker_model_load_seconds"],
                    error_type=load_error_type if reranker_load_error else None,
                    error=reranker_load_error,
                )

    case_results = []
    rerank_total = 0.0
    fallback_count = 0
    for case, query_text, orders in zip(cases, query_texts, stage_orders, strict=True):
        # RRF ranks chunks first. Only this bounded chunk set is allowed to
        # determine the final section results; we do not aggregate all corpus
        # sections before applying the chunk cutoff.
        candidates = orders["rrf"][:final_chunk_limit]
        if mode is not RerankerMode.NONE:
            candidates = candidates[: min(rerank_candidates, final_chunk_limit)]
        case_id = str(case.get("case_id", case.get("id", len(case_results))))
        reranker_error = None
        scores: list[float] = []
        reranked = candidates
        if mode is not RerankerMode.NONE and candidates:
            started = time.perf_counter()
            with tracer.stage(
                "reranker",
                metadata={
                    "case_id": case_id,
                    "question_type": case["question_type"],
                    "language": case.get("language", _language(case["query"])),
                    "mode": mode.value,
                    "candidate_count": len(candidates),
                },
            ) as rerank_stage:
                try:
                    if reranker_load_error is not None:
                        raise RuntimeError(
                            f"reranker model load failed: {reranker_load_error}"
                        )
                    pairs = [(query_text, document_texts[index]) for index in candidates]
                    if mode is RerankerMode.CROSS_ENCODER:
                        scores = _rerank_scores(
                            reranker_model, reranker_tokenizer, pairs,
                            batch_size=4, max_length=2048,
                        )
                    else:
                        scores = _llm_rerank_scores(
                            reranker_model, reranker_tokenizer, pairs,
                            batch_size=1, max_length=2048,
                        )
                    business_scores = {}
                    business_reasons = {}
                    final_scores = {}
                    for index, model_score in zip(candidates, scores, strict=True):
                        rule_score, reasons = _business_rule_score(case, chunks[index])
                        business_scores[index] = rule_score
                        business_reasons[index] = reasons
                        final_scores[index] = (
                            (1 - business_rule_weight) * model_score
                            + business_rule_weight * rule_score
                        )
                    reranked = [
                        index for index, _ in sorted(
                            final_scores.items(), key=lambda item: item[1], reverse=True,
                        )
                    ]
                    score_by_index = final_scores
                    tracer.update(
                        rerank_stage,
                        status="complete",
                        candidates_before=_candidate_summary(candidates, chunks),
                        candidates_after=_candidate_summary(
                            reranked, chunks, score_by_index
                        ),
                        rank_delta=_rank_deltas(candidates, reranked, chunks),
                    )
                except Exception as error:  # Degraded retrieval is better than no result.
                    fallback_count += 1
                    reranker_error = f"{type(error).__name__}: {str(error)[:240]}"
                    reranked = candidates
                    scores = []
                    tracer.update(
                        rerank_stage,
                        status="fallback",
                        fallback="rrf",
                        error_type=type(error).__name__,
                        error=reranker_error,
                        rank_delta=_rank_deltas(candidates, reranked, chunks),
                    )
                finally:
                    rerank_total += time.perf_counter() - started
        if not scores:
            score_by_index = {}
            business_scores = {}
            business_reasons = {}
            final_scores = {}
        top_sections = []
        for index in _section_order(reranked, chunks)[:3]:
            chunk = chunks[index]
            section = section_lookup[str(chunk["section_id"])]
            top_sections.append(
                {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "path": section["path"],
                    "score": round(score_by_index[index], 6)
                    if index in score_by_index
                    else None,
                    "model_score": round(scores[candidates.index(index)], 6)
                    if index in candidates and scores
                    else None,
                    "business_rule_score": business_scores.get(index),
                    "business_rule_reasons": business_reasons.get(index, []),
                    "matched_chunk_id": chunk["chunk_id"],
                    "matched_subheading": chunk.get("subheading"),
                    "text": section["text"],
                }
            )
        final_stage = "rrf" if mode is RerankerMode.NONE or reranker_error else "reranker"
        rank_orders = {
            "dense": orders["dense"],
            "bm25": orders["bm25"],
            "rrf": orders["rrf"][:final_chunk_limit],
            "reranker": reranked,
        }
        ranks = {
            name: _rank(order, chunks, case) for name, order in rank_orders.items()
        }
        case_results.append(
            {
                **case,
                "case_id": case_id,
                "ranks": ranks,
                "candidate_summary": _candidate_summary(candidates, chunks, score_by_index),
                "model_scores": {str(index): round(score, 6) for index, score in zip(candidates, scores, strict=True)},
                "business_rule_scores": {str(index): business_scores.get(index, 0.0) for index in candidates},
                "business_rule_reasons": {str(index): business_reasons.get(index, []) for index in candidates},
                "final_scores": {str(index): round(score, 6) for index, score in final_scores.items()},
                "rank_delta": _rank_deltas(candidates, reranked, chunks),
                "final_stage": final_stage,
                "reranker_error": reranker_error,
                "top_sections": top_sections,
            }
        )
    timings["rerank_seconds"] = rerank_total

    metric_stages = ["dense", "bm25", "rrf"]
    if mode is not RerankerMode.NONE:
        metric_stages.append("reranker")
    metrics = {
        stage: _metrics([result["ranks"][stage] for result in case_results])
        for stage in metric_stages
    }
    report = {
        "corpus": {
            "chunks": len(chunks),
            "sections": len(sections),
            "evaluation_cases": len(cases),
        },
        "models": {
            "embedding": str(embedding_model_path),
            "reranker_mode": mode.value,
            "reranker": str(reranker_model_path)
            if mode is RerankerMode.CROSS_ENCODER
            else str(llm_model_path)
            if mode is RerankerMode.LLM
            else None,
            "embedding_dimension": int(document_embeddings.shape[1]),
            "device": torch.cuda.get_device_name(0),
        },
        "parameters": {
            "dense_limit": dense_limit,
            "bm25_limit": bm25_limit,
            "rerank_candidates": rerank_candidates,
            "final_chunk_limit": final_chunk_limit,
            "rrf_k": 60,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "business_rule_weight": business_rule_weight,
            "parallel_retrieval_workers": 2,
        },
        "fallback": {
            "strategy": "rrf",
            "count": fallback_count,
        },
        "timings": {key: round(value, 4) for key, value in timings.items()},
        "metrics": metrics,
        "cases": case_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    return report


def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    tracer = get_tracer()
    mode = RerankerMode(kwargs.get("reranker_mode", RerankerMode.CROSS_ENCODER))
    chunks_path = Path(args[0]) if args else Path(kwargs["chunks_path"])
    embedding_path = Path(args[4]) if len(args) > 4 else Path(kwargs["embedding_model_path"])
    reranker_path = (
        Path(args[5]) if len(args) > 5 and args[5] else kwargs.get("reranker_model_path")
    )
    metadata = {
        "corpus_version": file_version(chunks_path),
        "embedding_model": embedding_path.name,
        "reranker_mode": mode.value,
        "reranker_model": Path(reranker_path).name if reranker_path else None,
        "question_type": "mixed",
        "language": "mixed",
    }
    try:
        with tracer.root("query_evaluation", metadata=metadata) as root:
            report = _evaluate(*args, **kwargs)
            empty_count = sum(not case["candidate_summary"] for case in report["cases"])
            tracer.update(
                root,
                status="complete",
                success=True,
                metrics=report["metrics"],
                timings=report["timings"],
                fallback_count=report["fallback"]["count"],
                empty_retrieval_count=empty_count,
                cases=len(report["cases"]),
            )
            return report
    except Exception as error:
        tracer.update(
            root if "root" in locals() else None,
            status="error",
            error_type=type(error).__name__,
        )
        raise
    finally:
        tracer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval and Qwen reranking")
    parser.add_argument("chunks", type=Path)
    parser.add_argument("sections", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("embedding_model", type=Path)
    parser.add_argument("reranker_model", type=Path, nargs="?")
    parser.add_argument(
        "--reranker-mode",
        choices=[mode.value for mode in RerankerMode],
        default=RerankerMode.CROSS_ENCODER.value,
    )
    parser.add_argument("--llm-model", type=Path)
    parser.add_argument("--rerank-candidates", type=int, default=30)
    parser.add_argument("--dense-limit", type=int, default=50)
    parser.add_argument("--bm25-limit", type=int, default=50)
    parser.add_argument("--final-chunk-limit", type=int, default=15)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--business-rule-weight", type=float, default=0.2)
    args = parser.parse_args()

    report = evaluate(
        args.chunks,
        args.sections,
        args.cases,
        args.output,
        args.embedding_model,
        args.reranker_model,
        reranker_mode=args.reranker_mode,
        llm_model_path=args.llm_model,
        rerank_candidates=args.rerank_candidates,
        dense_limit=args.dense_limit,
        bm25_limit=args.bm25_limit,
        final_chunk_limit=args.final_chunk_limit,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        business_rule_weight=args.business_rule_weight,
    )
    print(json.dumps({"metrics": report["metrics"], "timings": report["timings"]}))


if __name__ == "__main__":
    main()
