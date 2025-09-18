"""Run workflow retrieval experiments without LLM generated descriptions.

Usage:
    python retrieval_wo_llm.py --input data/turn_level_data_final.jsonl \
        --output-dir outputs_wo_llm --topk 5 --score-threshold 12

Arguments:
    --input            Path to the turn-level JSONL file.
    --output-dir       Directory to store the enriched JSONL outputs.
    --methods          Retrieval methods to run (naive, hier_domain, hier_role, hier_domain_role).
    --pool-types       Which scenario pools to use (summary, text, code, flowchart).
    --topk             Number of predictions to output per turn.
    --score-threshold  Minimum BM25 score; lower scores produce empty predictions.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer, util

from utils import load_jsonl, load_pool, load_reranker, save_jsonl

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


def simple_tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"\W+", text.lower()) if token]


class BM25Index:
    """Wrapper that keeps ids associated with BM25 tokens."""

    def __init__(self, items: Sequence[Tuple[str, str]]):
        self.ids = []
        self.texts = []
        tokenized = []
        for item_id, text in items:
            self.ids.append(item_id)
            cleaned = text.replace("\n", " ")
            self.texts.append(cleaned)
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
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_last_user_messages(messages: Sequence[Dict], limit: int) -> List[str]:
    snippets: List[str] = []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = clean_content(msg.get("content", ""))
        if text:
            snippets.append(text)
        if len(snippets) >= limit:
            break
    snippets.reverse()
    return snippets


def extract_action_blocks(messages: Sequence[Dict]) -> Tuple[str, str]:
    action_name = ""
    action_params = ""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not action_name:
            match = re.search(r"Action\s*:\s*([\w.-]+)", content)
            if match:
                action_name = match.group(1)
        if not action_params:
            match = re.search(r"Action Input\s*:\s*(\{.*\})", content, re.DOTALL)
            if match:
                raw = match.group(1).strip()
                parsed = None
                try:
                    parsed = json.loads(raw)
                except Exception:
                    pass
                if isinstance(parsed, dict):
                    flat_parts: List[str] = []

                    def flatten(prefix: str, value):
                        if isinstance(value, dict):
                            for k, v in value.items():
                                flatten(f"{prefix}{k} ", v)
                        else:
                            flat_parts.append(f"{prefix}{value}")

                    flatten("", parsed)
                    action_params = " ".join(flat_parts)
                else:
                    action_params = re.sub(r"[^A-Za-z0-9 ]+", " ", raw)
        if action_name and action_params:
            break
    return action_name, action_params


def extract_intent(messages: Sequence[Dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        match = re.search(r"intent\s*=\s*([^,\n\.]+)", content)
        if match:
            return clean_content(match.group(1))
    return ""


def build_last_history(messages: Sequence[Dict], max_turns: int = 3) -> str:
    window = max_turns * 2
    selected = messages[-window:]
    snippets = [f"{msg['role']}: {clean_content(msg.get('content', ''))}" for msg in selected]
    return " ".join(snippets)


def build_full_history(messages: Sequence[Dict]) -> str:
    snippets = [f"{msg['role']}: {clean_content(msg.get('content', ''))}" for msg in messages]
    return " ".join(snippets)


def compose_query(messages: Sequence[Dict], max_turns: int = 3) -> Dict[str, str]:
    user_texts = extract_last_user_messages(messages, limit=max_turns)
    action_name, action_params = extract_action_blocks(messages)
    intent = extract_intent(messages)

    structured_parts = [" ".join(user_texts), action_name, action_params, intent]
    primary = " ".join(part for part in structured_parts if part).strip()

    last_history = build_last_history(messages, max_turns=max_turns)
    full_history = build_full_history(messages)

    if not primary:
        primary = last_history or full_history

    return {
        "primary": primary,
        "last": last_history,
        "full": full_history,
    }


def merge_with_priority(
    primary: Sequence[Tuple[str, float]],
    secondary: Sequence[Tuple[str, float]],
    boost: float,
    limit: int,
) -> List[Tuple[str, float]]:
    scored: Dict[str, float] = {}
    for doc_id, score in primary:
        scored[doc_id] = max(scored.get(doc_id, float("-inf")), score + boost)
    for doc_id, score in secondary:
        scored[doc_id] = max(scored.get(doc_id, float("-inf")), score)
    ordered = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    return [(doc_id, score) for doc_id, score in ordered[:limit]]


def build_metadata(base_pool: Dict[str, Dict]) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, Dict[str, List[str]]],
]:
    domain_to_ids: Dict[str, List[str]] = defaultdict(list)
    role_to_ids: Dict[str, List[str]] = defaultdict(list)
    domain_role_to_ids: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for uuid, info in base_pool.items():
        domain = info.get("domain", "")
        role = info.get("role", "")
        domain_to_ids[domain].append(uuid)
        role_to_ids[role].append(uuid)
        domain_role_to_ids[domain][role].append(uuid)
    return domain_to_ids, role_to_ids, domain_role_to_ids


def rerank_candidates(
    query_text: str,
    candidates: Sequence[Tuple[str, float]],
    reranker_model,
    scenario_embeddings: Dict[str, torch.Tensor],
    top_k: int,
) -> List[Tuple[str, float]]:
    if not candidates:
        return []
    doc_ids = [doc_id for doc_id, _ in candidates if doc_id in scenario_embeddings]
    if not doc_ids:
        return []
    query_embedding = reranker_model.encode(
        query_text, convert_to_tensor=True, normalize_embeddings=True
    )
    doc_embeddings = torch.stack([scenario_embeddings[doc_id] for doc_id in doc_ids])
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
                embeddings[doc_id] = emb
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
    restricted = []
    if candidate_ids:
        restricted = scenario_index.search(
            query,
            top_k=max(top_k * CANDIDATE_MULTIPLIER, top_k),
            allowed_ids=candidate_ids,
        )
    global_results = scenario_index.search(query, top_k=top_k * CANDIDATE_MULTIPLIER)
    return merge_with_priority(
        restricted,
        global_results,
        boost=0.5,
        limit=max(top_k * CANDIDATE_MULTIPLIER, top_k),
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
    restricted = []
    if candidate_ids:
        restricted = scenario_index.search(
            query,
            top_k=max(top_k * CANDIDATE_MULTIPLIER, top_k),
            allowed_ids=candidate_ids,
        )
    global_results = scenario_index.search(query, top_k=top_k * CANDIDATE_MULTIPLIER)
    return merge_with_priority(
        restricted,
        global_results,
        boost=0.5,
        limit=max(top_k * CANDIDATE_MULTIPLIER, top_k),
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
    restricted = []
    if candidate_ids:
        restricted = scenario_index.search(
            query,
            top_k=max(top_k * CANDIDATE_MULTIPLIER, top_k),
            allowed_ids=candidate_ids,
        )
    global_results = scenario_index.search(query, top_k=top_k * CANDIDATE_MULTIPLIER)
    return merge_with_priority(
        restricted,
        global_results,
        boost=1.0,
        limit=max(top_k * CANDIDATE_MULTIPLIER, top_k),
    )


def ensure_answer_type_key(entry: Dict) -> None:
    if "answer type" not in entry and "answer_type" in entry:
        entry["answer type"] = entry["answer_type"]


def main():
    parser = argparse.ArgumentParser(description="Workflow retrieval without LLM support")
    parser.add_argument("--input", type=str, required=True, help="Turn-level jsonl input")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to store retrieval outputs"
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--domain-top-n", type=int, default=3)
    parser.add_argument("--role-top-n", type=int, default=3)
    parser.add_argument("--last-turns", type=int, default=3)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=METHODS, help="Retrieval methods"
    )
    parser.add_argument(
        "--pool-types",
        nargs="+",
        choices=list(POOL_NAME_MAP.keys()),
        default=list(POOL_NAME_MAP.keys()),
        help="Which pools to evaluate",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=12.0,
        help="Minimum BM25 score required for a non-empty prediction",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(str(input_path))
    queries = [compose_query(item["messages"], max_turns=args.last_turns) for item in data]

    reranker_model = load_reranker()

    base_pool = load_pool(POOL_NAME_MAP["text"])
    domain_to_ids, role_to_ids, domain_role_to_ids = build_metadata(base_pool)

    domain_desc = load_pool("domain_desc")
    role_desc = load_pool("role_desc")
    domain_index = BM25Index([(domain, desc) for domain, desc in domain_desc.items()])
    role_index = BM25Index([(role, desc) for role, desc in role_desc.items()])

    scenario_indexes: Dict[str, BM25Index] = {}
    scenario_embeddings_map: Dict[str, Dict[str, torch.Tensor]] = {}
    for pool_key in args.pool_types:
        pool_name = POOL_NAME_MAP[pool_key]
        index, _, embeddings = build_scenario_index(pool_name, reranker_model)
        scenario_indexes[pool_key] = index
        scenario_embeddings_map[pool_key] = embeddings

    for method in args.methods:
        for pool_key in args.pool_types:
            scenario_index = scenario_indexes[pool_key]
            scenario_embeddings = scenario_embeddings_map[pool_key]
            outputs = []
            for item, query_dict in zip(data, queries):
                query_text = query_dict["primary"] or query_dict["full"]
                if not query_text:
                    query_text = query_dict["full"]

                def fetch(q: str) -> List[Tuple[str, float]]:
                    if method == "naive":
                        return run_naive(q, scenario_index, args.topk)
                    if method == "hier_domain":
                        return run_hier_domain(
                            q,
                            scenario_index,
                            domain_index,
                            domain_to_ids,
                            args.topk,
                            args.domain_top_n,
                        )
                    if method == "hier_role":
                        return run_hier_role(
                            q,
                            scenario_index,
                            role_index,
                            role_to_ids,
                            args.topk,
                            args.role_top_n,
                        )
                    if method == "hier_domain_role":
                        return run_hier_domain_role(
                            q,
                            scenario_index,
                            domain_index,
                            role_index,
                            domain_role_to_ids,
                            args.topk,
                            args.domain_top_n,
                            args.role_top_n,
                        )
                    raise ValueError(f"Unknown method: {method}")

                candidate_pairs = fetch(query_text)
                if (
                    not candidate_pairs
                    and query_dict["full"]
                    and query_dict["full"] != query_text
                ):
                    candidate_pairs = fetch(query_dict["full"])

                if method == "naive":
                    candidate_pairs = candidate_pairs[
                        : max(args.topk * CANDIDATE_MULTIPLIER, args.topk)
                    ]

                bm25_top_score = (
                    candidate_pairs[0][1] if candidate_pairs else float("-inf")
                )

                pred_pairs = rerank_candidates(
                    query_text,
                    candidate_pairs,
                    reranker_model,
                    scenario_embeddings,
                    args.topk,
                )

                preds = [doc_id for doc_id, _ in pred_pairs]

                if bm25_top_score < args.score_threshold:
                    preds = []

                enriched = dict(item)
                ensure_answer_type_key(enriched)
                enriched["prediction"] = preds
                for i in range(1, args.topk + 1):
                    enriched[f"prediction_top{i}"] = preds[:i]
                outputs.append(enriched)

            output_path = (
                output_dir
                / f"{method}_{pool_key}_top{args.topk}_domain{args.domain_top_n}_role{args.role_top_n}.jsonl"
            )
            save_jsonl(outputs, str(output_path))


if __name__ == "__main__":
    main()
