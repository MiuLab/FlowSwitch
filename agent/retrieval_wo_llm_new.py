"""Workflow retrieval helpers with a simple function-based API.

Usage example::

    from retrieval_wo_llm import retrieve_workflows

    dialogue = [
        {"role": "user", "content": "Please find a dinner restaurant for me."},
        {"role": "assistant", "content": "Sure, what cuisine are you interested in?"},
        {"role": "user", "content": "Italian for two people."},
    ]

    workflows = retrieve_workflows(dialogue, topk=5, query_variant="last3")

Callers can import this module to run retrieval directly without relying on the
CLI batch runner.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import util

from utils import load_pool, load_reranker

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:
    raise ImportError("Please install rank_bm25: pip install rank-bm25") from exc


POOL_NAME_MAP = {
    "summary": "workflow_summary",
    "text": "workflow_text",
    "code": "workflow_code",
    "flowchart": "workflow_flowchart",
}

METHODS = ["naive", "hier_domain", "hier_role", "hier_domain_role"]
CANDIDATE_MULTIPLIER = 5

DEFAULT_METHOD = "naive"
DEFAULT_POOL_TYPE = "flowchart"
DEFAULT_QUERY_VARIANT = "last3"

VALID_QUERY_VARIANTS = {"full", "last1", "last2", "last3"}
QUERY_VARIANT_ALIASES = {
    "last1~3": "last3",
    "last1-3": "last3",
    "last1to3": "last3",
}

_RERANKER_ENABLED = False


def set_reranker_enabled(enabled: bool) -> None:
    """Globally toggle reranker usage without modifying caller code."""
    global _RERANKER_ENABLED
    _RERANKER_ENABLED = bool(enabled)


def is_reranker_enabled() -> bool:
    return _RERANKER_ENABLED


def normalize_variant(name: str) -> str:
    lowered = name.strip().lower()
    normalized = QUERY_VARIANT_ALIASES.get(lowered, lowered)
    if normalized not in VALID_QUERY_VARIANTS:
        raise ValueError(
            f"Unsupported query variant '{name}'. Supported options: {sorted(VALID_QUERY_VARIANTS)}"
        )
    return normalized


def simple_tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"\W+", text.lower()) if token]


class BM25Index:
    """Wrapper that keeps ids associated with BM25 tokens."""

    def __init__(self, items: Sequence[Tuple[str, str]]):
        self.ids: List[str] = []
        tokenized: List[List[str]] = []
        for item_id, text in items:
            self.ids.append(item_id)
            cleaned = text.replace("\n", " ")
            tokenized.append(simple_tokenize(cleaned))
        self.model = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        allowed_ids: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, float]]:
        tokens = simple_tokenize(query)
        if not tokens:
            return []
        scores = self.model.get_scores(tokens)
        scored = list(zip(self.ids, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if allowed_ids is not None:
            allowed_set = set(allowed_ids)
            scored = [pair for pair in scored if pair[0] in allowed_set]
        if top_k is not None:
            scored = scored[:top_k]
        return scored


def clean_content(content: str) -> str:
    text = content.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def build_last_history(messages: Sequence[Dict], max_turns: int = 3) -> str:
    if max_turns <= 0:
        return ""

    collected: List[Dict] = []
    user_count = 0
    for msg in reversed(messages):
        if msg.get("role") == "user":
            if user_count >= max_turns:
                break
            user_count += 1
        collected.append(msg)

    collected.reverse()
    snippets = [
        f"{msg['role']}: {clean_content(msg.get('content', ''))}" for msg in collected
    ]
    return " ".join(snippets)


def build_full_history(messages: Sequence[Dict]) -> str:
    snippets = [
        f"{msg['role']}: {clean_content(msg.get('content', ''))}" for msg in messages
    ]
    return " ".join(snippets)


def compose_query(
    messages: Sequence[Dict], requested_variants: Sequence[str], max_turns: int = 3
) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    if "full" in requested_variants:
        queries["full"] = build_full_history(messages)
    for variant in requested_variants:
        if variant == "full" or variant in queries:
            continue
        if variant.startswith("last"):
            try:
                turn_count = int(variant[4:])
            except ValueError as exc:
                raise ValueError(f"Invalid last-turn variant: {variant}") from exc
            effective_turns = max(1, min(turn_count, max_turns))
            queries[variant] = build_last_history(messages, max_turns=effective_turns)
            continue
        raise ValueError(f"Unknown query variant: {variant}")
    return queries


def build_metadata(
    base_pool: Dict[str, Dict],
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[str]]],
]:
    domain_to_ids: Dict[str, List[str]] = {}
    role_to_ids: Dict[str, List[str]] = {}
    domain_role_to_ids: Dict[str, Dict[str, List[str]]] = {}
    for uuid, info in base_pool.items():
        domain = info.get("domain", "")
        role = info.get("role", "")
        domain_to_ids.setdefault(domain, []).append(uuid)
        role_to_ids.setdefault(role, []).append(uuid)
        domain_role_to_ids.setdefault(domain, {}).setdefault(role, []).append(uuid)
    return domain_to_ids, role_to_ids, domain_role_to_ids


def rerank_candidates(
    query_text: str,
    candidates: Sequence[Tuple[str, float]],
    reranker_model,
    scenario_embeddings: Dict[str, torch.Tensor],
    top_k: int,
    query_embedding: Optional[torch.Tensor] = None,
) -> List[Tuple[str, float]]:
    if not candidates:
        return []
    if reranker_model is None or not _RERANKER_ENABLED:
        # When reranker is disabled, return top-k BM25 candidates directly.
        return [(doc_id, score) for doc_id, score in candidates[:top_k]]

    doc_ids = [doc_id for doc_id, _ in candidates if doc_id in scenario_embeddings]
    if not doc_ids:
        return []
    if query_embedding is None:
        query_embedding = reranker_model.encode(
            query_text, convert_to_tensor=True, normalize_embeddings=True
        )
    doc_embeddings = torch.stack([scenario_embeddings[doc_id] for doc_id in doc_ids])
    query_embedding = query_embedding.to(doc_embeddings.device)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    reranked = [
        (doc_id, float(score)) for doc_id, score in zip(doc_ids, scores.detach().cpu())
    ]
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked[:top_k]


def build_scenario_index(
    pool_name: str, reranker_model
) -> Tuple[BM25Index, Dict[str, str], Dict[str, torch.Tensor]]:
    pool = load_pool(pool_name)
    documents: List[Tuple[str, str]] = []
    metadata: Dict[str, str] = {}
    for uuid, info in pool.items():
        pieces = [
            info.get("scenario_name", ""),
            info.get("domain", ""),
            info.get("role", ""),
            info.get("workflow", ""),
        ]
        doc_text = " ".join(piece for piece in pieces if piece)
        documents.append((uuid, doc_text))
        metadata[uuid] = doc_text
    index = BM25Index(documents)

    embeddings: Dict[str, torch.Tensor] = {}
    if reranker_model is not None:
        texts = [text for _, text in documents]
        if texts:
            doc_embs = reranker_model.encode(
                texts, convert_to_tensor=True, normalize_embeddings=True
            )
            for (doc_id, _), emb in zip(documents, doc_embs):
                embeddings[doc_id] = emb.detach().cpu()
    return index, metadata, embeddings


def run_naive(
    query: str,
    scenario_index: BM25Index,
    top_k: int,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[Tuple[str, float]]:
    limit = max(top_k * CANDIDATE_MULTIPLIER, top_k)
    return scenario_index.search(query, top_k=limit, allowed_ids=allowed_ids)


def run_hier_domain(
    query: str,
    scenario_index: BM25Index,
    domain_index: BM25Index,
    domain_to_ids: Dict[str, List[str]],
    top_k: int,
    domain_top_n: int,
) -> List[Tuple[str, float]]:
    domain_scores = domain_index.search(query, top_k=domain_top_n)
    candidate_ids: List[str] = []
    for domain, _ in domain_scores:
        candidate_ids.extend(domain_to_ids.get(domain, []))
    if not candidate_ids:
        return []
    return scenario_index.search(
        query,
        top_k=max(top_k * CANDIDATE_MULTIPLIER, top_k),
        allowed_ids=candidate_ids,
    )


def run_hier_role(
    query: str,
    scenario_index: BM25Index,
    role_index: BM25Index,
    role_to_ids: Dict[str, List[str]],
    top_k: int,
    role_top_n: int,
) -> List[Tuple[str, float]]:
    role_scores = role_index.search(query, top_k=role_top_n)
    candidate_ids: List[str] = []
    for role, _ in role_scores:
        candidate_ids.extend(role_to_ids.get(role, []))
    if not candidate_ids:
        return []
    return scenario_index.search(
        query,
        top_k=max(top_k * CANDIDATE_MULTIPLIER, top_k),
        allowed_ids=candidate_ids,
    )


def run_hier_domain_role(
    query: str,
    scenario_index: BM25Index,
    domain_index: BM25Index,
    role_index: BM25Index,
    domain_role_to_ids: Dict[str, Dict[str, List[str]]],
    top_k: int,
    domain_top_n: int,
    role_top_n: int,
) -> List[Tuple[str, float]]:
    domain_scores = domain_index.search(query, top_k=domain_top_n)
    candidate_ids: List[str] = []
    for domain, _ in domain_scores:
        role_map = domain_role_to_ids.get(domain, {})
        if not role_map:
            continue
        role_candidates = role_index.search(
            query, top_k=role_top_n, allowed_ids=list(role_map.keys())
        )
        for role, _ in role_candidates:
            candidate_ids.extend(role_map.get(role, []))
    if not candidate_ids:
        return []
    return scenario_index.search(
        query,
        top_k=max(top_k * CANDIDATE_MULTIPLIER, top_k),
        allowed_ids=candidate_ids,
    )


_RESOURCE_CACHE: Dict[str, object] = {}
_THRESHOLD_SKIP_LOGGED = False


def _ensure_resources() -> Dict[str, object]:
    if _RESOURCE_CACHE:
        return _RESOURCE_CACHE

    reranker_model = load_reranker()

    base_pool = load_pool(POOL_NAME_MAP["text"])
    domain_to_ids, role_to_ids, domain_role_to_ids = build_metadata(base_pool)

    domain_desc = load_pool("domain_desc")
    role_desc = load_pool("role_desc")
    domain_index = BM25Index(
        [(domain, f"{domain} {desc}") for domain, desc in domain_desc.items()]
    )
    role_index = BM25Index(
        [(role, f"{role} {desc}") for role, desc in role_desc.items()]
    )

    scenario_indexes: Dict[str, BM25Index] = {}
    scenario_embeddings_map: Dict[str, Dict[str, torch.Tensor]] = {}
    for pool_key, pool_name in POOL_NAME_MAP.items():
        index, _, embeddings = build_scenario_index(pool_name, reranker_model)
        scenario_indexes[pool_key] = index
        scenario_embeddings_map[pool_key] = embeddings

    # Use the text pool as the canonical embedding source for scenario-level retrieval so that
    # the same UUID does not get overwritten by other formats (code/summary/flowchart).
    scenario_embeddings_all: Dict[str, torch.Tensor] = {}
    text_embeddings = scenario_embeddings_map.get("text", {})
    scenario_embeddings_all.update(text_embeddings)

    domain_embeddings: Dict[str, torch.Tensor] = {}
    role_embeddings: Dict[str, torch.Tensor] = {}
    if reranker_model is not None:
        domain_texts = [
            (domain, f"{domain} {desc}") for domain, desc in domain_desc.items()
        ]
        role_texts = [(role, f"{role} {desc}") for role, desc in role_desc.items()]
        if domain_texts:
            encoded = reranker_model.encode(
                [text for _, text in domain_texts],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            for (domain, _), emb in zip(domain_texts, encoded):
                domain_embeddings[domain] = emb.detach().cpu()
        if role_texts:
            encoded = reranker_model.encode(
                [text for _, text in role_texts],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            for (role, _), emb in zip(role_texts, encoded):
                role_embeddings[role] = emb.detach().cpu()

    _RESOURCE_CACHE.update(
        {
            "reranker_model": reranker_model,
            "domain_index": domain_index,
            "role_index": role_index,
            "domain_to_ids": domain_to_ids,
            "role_to_ids": role_to_ids,
            "domain_role_to_ids": domain_role_to_ids,
            "scenario_indexes": scenario_indexes,
            "scenario_embeddings_map": scenario_embeddings_map,
            "scenario_embeddings_all": scenario_embeddings_all,
            "domain_embeddings": domain_embeddings,
            "role_embeddings": role_embeddings,
        }
    )

    return _RESOURCE_CACHE


def _validate_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"Unsupported method '{method}'. Valid options: {METHODS}")
    return method


def _validate_pool(pool_type: str) -> str:
    if pool_type not in POOL_NAME_MAP:
        raise ValueError(
            f"Unsupported pool type '{pool_type}'. Valid options: {list(POOL_NAME_MAP.keys())}"
        )
    return pool_type


def _prepare_query(
    dialogue: Sequence[Dict[str, str]],
    query_variant: str,
    last_turns: int,
) -> Tuple[str, str]:
    normalized = normalize_variant(query_variant)
    queries = compose_query(dialogue, [normalized], max_turns=last_turns)
    return normalized, queries.get(normalized, "")


def _fetch_candidates(
    method: str,
    query: str,
    topk: int,
    *,
    scenario_index: BM25Index,
    domain_index: BM25Index,
    role_index: BM25Index,
    domain_to_ids: Dict[str, List[str]],
    role_to_ids: Dict[str, List[str]],
    domain_role_to_ids: Dict[str, Dict[str, List[str]]],
    domain_top_n: int,
    role_top_n: int,
) -> List[Tuple[str, float]]:
    if method == "naive":
        return run_naive(query, scenario_index, topk)
    if method == "hier_domain":
        return run_hier_domain(
            query,
            scenario_index,
            domain_index,
            domain_to_ids,
            topk,
            domain_top_n,
        )
    if method == "hier_role":
        return run_hier_role(
            query,
            scenario_index,
            role_index,
            role_to_ids,
            topk,
            role_top_n,
        )
    if method == "hier_domain_role":
        return run_hier_domain_role(
            query,
            scenario_index,
            domain_index,
            role_index,
            domain_role_to_ids,
            topk,
            domain_top_n,
            role_top_n,
        )
    raise ValueError(f"Unknown method '{method}'")


def retrieve_workflows(
    dialogue: Sequence[Dict[str, str]],
    topk: int,
    *,
    query_variant: str = DEFAULT_QUERY_VARIANT,
    method: str = DEFAULT_METHOD,
    pool_type: str = DEFAULT_POOL_TYPE,
    domain_top_n: int = 3,
    role_top_n: int = 3,
    last_turns: int = 3,
    score_threshold: Optional[float] = None,
) -> List[str]:
    """Return top-k workflow UUIDs for the provided dialogue.

    Args:
        dialogue: Conversation history (list of message dicts with ``role`` and
            ``content`` keys).
        topk: Number of predictions to return.
        query_variant: Which dialogue window to use (``full``, ``last1``, ``last2``,
            ``last3``)。
        method: Retrieval strategy (``naive``, ``hier_domain``, ``hier_role``,
            ``hier_domain_role``).
        pool_type: Scenario pool to search (``summary``, ``text``, ``code``,
            ``flowchart``).
        domain_top_n: Number of domains to keep for hierarchical retrieval.
        role_top_n: Number of roles to keep for hierarchical retrieval.
        last_turns: Maximum number of user turns considered when ``query_variant``
            是 ``lastN``。
        score_threshold: Minimum similarity score required for a workflow to remain
            in the returned predictions. Only applied when a reranker model is loaded.

    Returns:
        List of workflow UUID strings ordered by relevance.
    """

    if not dialogue:
        return []

    method = _validate_method(method)
    pool_type = _validate_pool(pool_type)

    resources = _ensure_resources()

    _, query_text = _prepare_query(dialogue, query_variant, last_turns)
    if not query_text:
        return []

    scenario_index: BM25Index = resources["scenario_indexes"][pool_type]
    scenario_embeddings: Dict[str, torch.Tensor] = resources["scenario_embeddings_map"][
        pool_type
    ]
    domain_index: BM25Index = resources["domain_index"]
    role_index: BM25Index = resources["role_index"]
    domain_to_ids: Dict[str, List[str]] = resources["domain_to_ids"]
    role_to_ids: Dict[str, List[str]] = resources["role_to_ids"]
    domain_role_to_ids: Dict[str, Dict[str, List[str]]] = resources[
        "domain_role_to_ids"
    ]
    reranker_model = resources["reranker_model"]

    candidate_pairs = _fetch_candidates(
        method,
        query_text,
        topk,
        scenario_index=scenario_index,
        domain_index=domain_index,
        role_index=role_index,
        domain_to_ids=domain_to_ids,
        role_to_ids=role_to_ids,
        domain_role_to_ids=domain_role_to_ids,
        domain_top_n=domain_top_n,
        role_top_n=role_top_n,
    )

    if method == "naive":
        candidate_pairs = candidate_pairs[: max(topk * CANDIDATE_MULTIPLIER, topk)]

    if not candidate_pairs:
        return []

    query_embedding = None
    if reranker_model is not None:
        query_embedding = reranker_model.encode(
            query_text, convert_to_tensor=True, normalize_embeddings=True
        )

    reranked = rerank_candidates(
        query_text,
        candidate_pairs,
        reranker_model,
        scenario_embeddings,
        topk,
        query_embedding=query_embedding,
    )

    threshold_applicable = reranker_model is not None

    if score_threshold is not None:
        if threshold_applicable:
            before = len(reranked)
            reranked = [pair for pair in reranked if pair[1] >= score_threshold]
            logger.debug(
                "Applied score threshold %.3f: kept %d/%d (method=%s pool=%s variant=%s)",
                score_threshold,
                len(reranked),
                before,
                method,
                pool_type,
                query_variant,
            )
        else:
            global _THRESHOLD_SKIP_LOGGED
            if not _THRESHOLD_SKIP_LOGGED:
                logger.warning(
                    "Score threshold %.3f ignored because reranker resources were unavailable.",
                    score_threshold,
                )
                _THRESHOLD_SKIP_LOGGED = True

    reranked = reranked[:topk]

    return [doc_id for doc_id, _ in reranked]


def _encode_query_embedding(model, query_text: str) -> torch.Tensor:
    return (
        model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        .detach()
        .cpu()
    )


def _rank_embeddings(
    query_embedding: torch.Tensor,
    embeddings: Dict[str, torch.Tensor],
    top_k: int,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[Tuple[str, float]]:
    if allowed_ids is None:
        items = embeddings.items()
    else:
        allowed_set = set(allowed_ids)
        items = ((key, embeddings[key]) for key in allowed_set if key in embeddings)

    scores: List[Tuple[str, float]] = []
    for key, emb in items:
        emb = emb.to(query_embedding.device)
        score = float(torch.dot(query_embedding, emb))
        scores.append((key, score))
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores[:top_k]


def _require_reranker(resources: Dict[str, object]) -> None:
    if resources.get("reranker_model") is None:
        raise RuntimeError(
            "E5 retrieval requires the reranker model (intfloat/e5-base-v2) to be available."
        )


def run_naive_e5(
    query: str,
    scenario_index: BM25Index,
    top_k: int,
    allowed_ids: Optional[Iterable[str]] = None,
) -> List[Tuple[str, float]]:
    resources = _ensure_resources()
    _require_reranker(resources)
    model = resources["reranker_model"]
    scenario_embeddings_all: Dict[str, torch.Tensor] = resources["scenario_embeddings_all"]
    query_embedding = _encode_query_embedding(model, query)
    limit = max(top_k * CANDIDATE_MULTIPLIER, top_k)
    return _rank_embeddings(query_embedding, scenario_embeddings_all, limit, allowed_ids)


def run_hier_domain_e5(
    query: str,
    scenario_index: BM25Index,
    domain_index: BM25Index,
    domain_to_ids: Dict[str, List[str]],
    top_k: int,
    domain_top_n: int,
) -> List[Tuple[str, float]]:
    resources = _ensure_resources()
    _require_reranker(resources)
    model = resources["reranker_model"]
    domain_embeddings: Dict[str, torch.Tensor] = resources["domain_embeddings"]
    scenario_embeddings_all: Dict[str, torch.Tensor] = resources["scenario_embeddings_all"]
    query_embedding = _encode_query_embedding(model, query)

    domain_scores = _rank_embeddings(query_embedding, domain_embeddings, domain_top_n)
    candidate_ids: List[str] = []
    for domain, _ in domain_scores:
        candidate_ids.extend(domain_to_ids.get(domain, []))
    if not candidate_ids:
        return []
    limit = max(top_k * CANDIDATE_MULTIPLIER, top_k)
    return _rank_embeddings(
        query_embedding, scenario_embeddings_all, limit, allowed_ids=candidate_ids
    )


def run_hier_role_e5(
    query: str,
    scenario_index: BM25Index,
    role_index: BM25Index,
    role_to_ids: Dict[str, List[str]],
    top_k: int,
    role_top_n: int,
) -> List[Tuple[str, float]]:
    resources = _ensure_resources()
    _require_reranker(resources)
    model = resources["reranker_model"]
    role_embeddings: Dict[str, torch.Tensor] = resources["role_embeddings"]
    scenario_embeddings_all: Dict[str, torch.Tensor] = resources["scenario_embeddings_all"]
    query_embedding = _encode_query_embedding(model, query)

    role_scores = _rank_embeddings(query_embedding, role_embeddings, role_top_n)
    candidate_ids: List[str] = []
    for role, _ in role_scores:
        candidate_ids.extend(role_to_ids.get(role, []))
    if not candidate_ids:
        return []
    limit = max(top_k * CANDIDATE_MULTIPLIER, top_k)
    return _rank_embeddings(
        query_embedding, scenario_embeddings_all, limit, allowed_ids=candidate_ids
    )


def run_hier_domain_role_e5(
    query: str,
    scenario_index: BM25Index,
    domain_index: BM25Index,
    role_index: BM25Index,
    domain_role_to_ids: Dict[str, Dict[str, List[str]]],
    top_k: int,
    domain_top_n: int,
    role_top_n: int,
) -> List[Tuple[str, float]]:
    resources = _ensure_resources()
    _require_reranker(resources)
    model = resources["reranker_model"]
    domain_embeddings: Dict[str, torch.Tensor] = resources["domain_embeddings"]
    role_embeddings: Dict[str, torch.Tensor] = resources["role_embeddings"]
    scenario_embeddings_all: Dict[str, torch.Tensor] = resources["scenario_embeddings_all"]
    query_embedding = _encode_query_embedding(model, query)

    domain_scores = _rank_embeddings(query_embedding, domain_embeddings, domain_top_n)
    candidate_ids: List[str] = []
    for domain, _ in domain_scores:
        role_map = domain_role_to_ids.get(domain, {})
        if not role_map:
            continue
        role_scores = _rank_embeddings(
            query_embedding, role_embeddings, role_top_n, allowed_ids=role_map.keys()
        )
        for role, _ in role_scores:
            candidate_ids.extend(role_map.get(role, []))
    if not candidate_ids:
        return []
    limit = max(top_k * CANDIDATE_MULTIPLIER, top_k)
    return _rank_embeddings(
        query_embedding, scenario_embeddings_all, limit, allowed_ids=candidate_ids
    )


__all__ = [
    "retrieve_workflows",
    "set_reranker_enabled",
    "is_reranker_enabled",
    "run_naive_e5",
    "run_hier_domain_e5",
    "run_hier_role_e5",
    "run_hier_domain_role_e5",
]

