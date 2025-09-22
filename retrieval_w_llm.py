"""
Run workflow retrieval experiments with LLM-based hierarchical retrieval.
Usage:
    python retrieval_w_llm.py --input data/turn_level_data_final.jsonl \
        --output-dir outputs_w_llm --topk 5 --methods hier_domain hier_role \
        --pool-types text code --last-n-turns 0
Arguments:
    --input            Path to the turn-level JSONL file.
    --output-dir       Directory to store the enriched JSONL outputs.
    --methods          Retrieval methods to run (hier_domain, hier_role, hier_domain_role).
    --pool-types       Which scenario pools to use (summary, text, code, flowchart).
    --topk             Number of predictions to output per turn.
    --last-n-turns     Use only last n turns (0=all, 1=last 1 turn, 2=last 2 turns, etc).
"""

import json
import jsonlines
import os
from typing import List, Dict, Tuple, Any
from utils import load_pool, load_bm25_retriever, load_reranker, MAPPING, evaluate
from request_qwen import request_qwen_chat
import numpy as np
import logging
import argparse
import pandas as pd
import glob
from tqdm import tqdm
from datetime import datetime
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HierarchicalRetriever:
    def __init__(self, pool_name: str = "workflow_text", use_reranker: bool = True):
        self.pool_name = pool_name
        self.use_reranker = use_reranker

        self.workflow_pool = load_pool(pool_name)
        self.domain_desc = self._load_json("pools/domain_desc.json")
        self.role_desc = self._load_json("pools/role_desc.json")

        self.bm25_retriever = load_bm25_retriever(pool_name)
        if use_reranker:
            self.reranker = load_reranker()

        self.scenario_to_uuid = self._build_scenario_mapping()
        
    def _load_json(self, filepath: str) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_scenario_mapping(self) -> Dict[str, List[str]]:
        scenario_mapping = {}
        for uuid, workflow_data in self.workflow_pool.items():
            scenario = workflow_data['scenario_name']
            if scenario not in scenario_mapping:
                scenario_mapping[scenario] = []
            scenario_mapping[scenario].append(uuid)
        return scenario_mapping
    
    def _get_scenarios_by_domain(self, domain: str) -> List[str]:
        scenarios = []
        if domain in MAPPING:
            for role in MAPPING[domain]:
                scenarios.extend(MAPPING[domain][role])
        return scenarios
    
    def _get_scenarios_by_role(self, role: str) -> List[str]:
        for domain in MAPPING:
            if role in MAPPING[domain]:
                return MAPPING[domain][role]
        return []
    
    def _llm_select_from_candidates(self, query: str, candidates: List[str], 
                                   candidate_descriptions: Dict[str, str], 
                                   selection_type: str, top_k: int = 3) -> List[str]:

        candidate_info = []
        for candidate in candidates:
            desc = candidate_descriptions.get(candidate, "No description available")
            candidate_info.append(f"- {candidate}: {desc}")
        
        candidate_text = "\n".join(candidate_info)
        
        prompt = f"""Given the user query/dialogue history, please select the most relevant {selection_type}s from the candidates below.

User Query/Dialogue History:
{query}

Available {selection_type.capitalize()}s:
{candidate_text}

Please analyze the user's intent and select the top {top_k} most relevant {selection_type}s that best match the user's needs.
Return only the names of the selected {selection_type}s, one per line, in order of relevance (most relevant first).

Selected {selection_type}s:"""

        try:
            response = request_qwen_chat(
                messages=[{"role": "user", "content": prompt}],
                model="qwen3-8b",
                temperature=0.1
            )
            
            selected_text = response.get("response", "").strip()
            selected_lines = [line.strip().lstrip('- ') for line in selected_text.split('\n') if line.strip()]
            
            # Filter to only include valid candidates
            selected = []
            for line in selected_lines:
                if line in candidates:
                    selected.append(line)
                    if len(selected) >= top_k:
                        break
            
            # If no valid selections, fallback to first few candidates
            if not selected:
                selected = candidates[:top_k]
                
            return selected
            
        except Exception as e:
            logger.info(f"LLM selection failed for {selection_type}: {e}")
            # Fallback to first few candidates
            return candidates[:top_k]
    
    def _retrieve_and_rerank(self, query: str, candidate_uuids: List[str], top_k: int = 5) -> List[str]:
        if not candidate_uuids:
            return []
        
        candidate_texts = []
        uuid_to_text = {}
        
        for uuid in candidate_uuids:
            if uuid in self.workflow_pool:
                workflow_text = self.workflow_pool[uuid]['workflow']
                candidate_texts.append(workflow_text)
                uuid_to_text[uuid] = workflow_text
        
        if not candidate_texts:
            return []
        
        query_tokens = query.split()
        tokenized_candidates = [text.split() for text in candidate_texts]
        
        bm25_candidates = BM25Okapi(tokenized_candidates)
        bm25_scores = bm25_candidates.get_scores(query_tokens)
        
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        
        # Apply reranking if enabled
        if self.use_reranker and hasattr(self, 'reranker'):
            try:
                rerank_candidates = ranked_indices[:min(20, len(ranked_indices))]
                rerank_texts = [candidate_texts[i] for i in rerank_candidates]
                
                query_embedding = self.reranker.encode([f"query: {query}"])
                doc_embeddings = self.reranker.encode([f"passage: {text}" for text in rerank_texts])
                
                similarities = (query_embedding @ doc_embeddings.T)[0]
                
                rerank_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
                final_indices = [rerank_candidates[i] for i in rerank_indices]
                
            except Exception as e:
                logger.info(f"Reranking failed: {e}")
                final_indices = ranked_indices
        else:
            final_indices = ranked_indices
        
        result_uuids = []
        for i in final_indices[:top_k]:
            if i < len(candidate_uuids):
                result_uuids.append(candidate_uuids[i])
        
        return result_uuids
    
    def retrieve_2layer_domain_scenario(self, query: str, top_k: int = 5) -> List[str]:

        logger.info("Starting 2-layer Domain→Scenario retrieval...")
        
        all_domains = list(self.domain_desc.keys())
        selected_domains = self._llm_select_from_candidates(
            query=query,
            candidates=all_domains,
            candidate_descriptions=self.domain_desc,
            selection_type="domain",
            top_k=3  # Select top 3 domains
        )
        
        logger.info(f"Selected domains: {selected_domains}")
        
        candidate_scenarios = []
        for domain in selected_domains:
            domain_scenarios = self._get_scenarios_by_domain(domain)
            candidate_scenarios.extend(domain_scenarios)
        
        candidate_scenarios = list(dict.fromkeys(candidate_scenarios))
        logger.info(f"Candidate scenarios: {candidate_scenarios}")
        
        candidate_uuids = []
        for scenario in candidate_scenarios:
            if scenario in self.scenario_to_uuid:
                candidate_uuids.extend(self.scenario_to_uuid[scenario])
        
        if not candidate_uuids:
            logger.info("No candidate UUIDs found")
            return []
        
        final_results = self._retrieve_and_rerank(query, candidate_uuids, top_k)
        logger.info(f"Final results: {final_results}")
        
        return final_results
    
    def retrieve_2layer_role_scenario(self, query: str, top_k: int = 5) -> List[str]:

        logger.info("Starting 2-layer Role→Scenario retrieval...")
        
        all_roles = list(self.role_desc.keys())
        selected_roles = self._llm_select_from_candidates(
            query=query,
            candidates=all_roles,
            candidate_descriptions=self.role_desc,
            selection_type="role",
            top_k=3  # Select top 3 roles
        )
        
        logger.info(f"Selected roles: {selected_roles}")

        candidate_scenarios = []
        for role in selected_roles:
            role_scenarios = self._get_scenarios_by_role(role)
            candidate_scenarios.extend(role_scenarios)

        candidate_scenarios = list(dict.fromkeys(candidate_scenarios))
        logger.info(f"Candidate scenarios: {candidate_scenarios}")

        candidate_uuids = []
        for scenario in candidate_scenarios:
            if scenario in self.scenario_to_uuid:
                candidate_uuids.extend(self.scenario_to_uuid[scenario])

        if not candidate_uuids:
            logger.info("No candidate UUIDs found")
            return []
        
        final_results = self._retrieve_and_rerank(query, candidate_uuids, top_k)
        logger.info(f"Final results: {final_results}")
        
        return final_results
    
    def retrieve_3layer_domain_role_scenario(self, query: str, top_k: int = 5) -> List[str]:
        logger.info("Starting 3-layer Domain→Role→Scenario retrieval...")

        all_domains = list(self.domain_desc.keys())
        selected_domains = self._llm_select_from_candidates(
            query=query,
            candidates=all_domains,
            candidate_descriptions=self.domain_desc,
            selection_type="domain",
            top_k=3  # Select top 3 domains
        )
        
        logger.info(f"Selected domains: {selected_domains}")

        candidate_roles = []
        for domain in selected_domains:
            if domain in MAPPING:
                candidate_roles.extend(list(MAPPING[domain].keys()))

        candidate_roles = list(dict.fromkeys(candidate_roles))
        
        if not candidate_roles:
            logger.info("No candidate roles found")
            return []

        candidate_role_desc = {role: self.role_desc[role] for role in candidate_roles if role in self.role_desc}
        
        selected_roles = self._llm_select_from_candidates(
            query=query,
            candidates=candidate_roles,
            candidate_descriptions=candidate_role_desc,
            selection_type="role",
            top_k=3  # Select top 3 roles
        )
        
        logger.info(f"Selected roles: {selected_roles}")

        candidate_scenarios = []
        for role in selected_roles:
            role_scenarios = self._get_scenarios_by_role(role)
            candidate_scenarios.extend(role_scenarios)

        candidate_scenarios = list(dict.fromkeys(candidate_scenarios))
        logger.info(f"Candidate scenarios: {candidate_scenarios}")

        candidate_uuids = []
        for scenario in candidate_scenarios:
            if scenario in self.scenario_to_uuid:
                candidate_uuids.extend(self.scenario_to_uuid[scenario])

        if not candidate_uuids:
            logger.info("No candidate UUIDs found")
            return []
        
        final_results = self._retrieve_and_rerank(query, candidate_uuids, top_k)
        logger.info(f"Final results: {final_results}")
        
        return final_results
    
    def format_messages_to_query(self, messages, last_n_turns=0):
        if last_n_turns > 0:
            messages = messages[-last_n_turns * 2:] if len(messages) > last_n_turns * 2 else messages
        
        query_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role and content:
                query_parts.append(f"{role}: {content}")
        return " | ".join(query_parts)

def run_evaluation_and_summary(output_dir, methods, pool_types):
    summary_data = []
    
    for pool_type in pool_types:
        for method_name, _ in methods:
            pattern = f"{output_dir}/results_{pool_type}_{method_name}*.jsonl"
            result_files = glob.glob(pattern)
            
            for result_file in result_files:
                if os.path.exists(result_file):
                    try:
                        eval_output = result_file.replace('.jsonl', '_eval.json')
                        evaluate(result_file, eval_output, topk=5)

                        with open(eval_output, 'r', encoding='utf-8') as f:
                            eval_data = json.load(f)

                        filename = os.path.basename(result_file)

                        for topk in range(1, 6):
                            topk_key = f"top{topk}"
                            if topk_key in eval_data:
                                row = {
                                    'filename': filename,
                                    'topk': topk,
                                    'overall': 0,
                                    'SINGLE': 0,
                                    'AND': 0,
                                    'OR': 0,
                                    'UNK': 0
                                }

                                total_samples = 0
                                correct_samples = 0
                                
                                for result in eval_data[topk_key]:
                                    answer_type = result['answer_type']
                                    accuracy = result['accuracy']
                                    count = result.get('count', 0)
                                    
                                    row[answer_type] = accuracy
                                    total_samples += count
                                    correct_samples += accuracy * count
                                
                                if total_samples > 0:
                                    row['overall'] = correct_samples / total_samples
                                
                                summary_data.append(row)
                    
                    except Exception as e:
                        logger.error(f"Error evaluating {result_file}: {e}")

    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_file = f"{output_dir}/evaluation_summary.csv"
        df.to_csv(summary_file, index=False)
        logger.info(f"Evaluation summary saved to: {summary_file}")

        for topk in range(1, 6):
            topk_data = df[df['topk'] == topk]
            if not topk_data.empty:
                print(f"\n=== Top-{topk} Results ===")
                print(topk_data.to_string(index=False))
    
    return summary_data

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_test_data(filepath):
    data = []
    with jsonlines.open(filepath, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data

def run_hierarchical_experiments(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{args.output_dir}/hierarchical_retrieval_experiment_{timestamp}.log"
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = setup_logging(log_file)
    
    logger.info(f"Starting hierarchical retrieval experiments")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Arguments: {args}")

    test_data = load_test_data(args.input)
    logger.info(f"Loaded {len(test_data)} test samples")

    retrievers = {}
    for pool_type in args.pool_types:
        retrievers[pool_type] = HierarchicalRetriever(pool_name=f"workflow_{pool_type}", use_reranker=True)

    method_mapping = {
        "hier_domain": ("2layer_domain_scenario", "retrieve_2layer_domain_scenario"),
        "hier_role": ("2layer_role_scenario", "retrieve_2layer_role_scenario"), 
        "hier_domain_role": ("3layer_domain_role_scenario", "retrieve_3layer_domain_role_scenario")
    }
    
    methods = [(method_mapping[m][0], method_mapping[m][1]) for m in args.methods if m in method_mapping]

    for pool_name, retriever in retrievers.items():
        logger.info(f"{'='*80}")
        logger.info(f"Running experiments with {pool_name} pool")
        logger.info(f"{'='*80}")
        
        for method_name, method_func in methods:
            logger.info(f"Testing {method_name}...")

            if args.last_n_turns > 0:
                output_file = f"{args.output_dir}/results_{pool_name}_{method_name}_last{args.last_n_turns}turns.jsonl"
            else:
                output_file = f"{args.output_dir}/results_{pool_name}_{method_name}.jsonl"

            if os.path.exists(output_file):
                try:
                    with jsonlines.open(output_file, 'r') as reader:
                        processed_count = sum(1 for _ in reader)
                    
                    if processed_count >= len(test_data):
                        logger.info(f"File {output_file} already complete with {processed_count} samples, skipping...")
                        continue
                    else:
                        logger.info(f"File {output_file} incomplete ({processed_count}/{len(test_data)}), continuing...")
                except Exception as e:
                    logger.warning(f"Error reading {output_file}: {e}")

            processed_turn_ids = set()
            
            if os.path.exists(output_file):
                try:
                    with jsonlines.open(output_file, 'r') as reader:
                        for obj in reader:
                            processed_turn_ids.add(obj.get('turn_id'))
                    logger.info(f"Already processed {len(processed_turn_ids)} samples, continuing...")
                except Exception as e:
                    logger.warning(f"Error reading existing results: {e}")
                    processed_turn_ids = set()
            
            processed_count = len(processed_turn_ids)
            
            for i, item in enumerate(tqdm(test_data, desc=f"{pool_name}_{method_name}")):
                try:
                    turn_id = item.get('turn_id', f'turn_{i}')

                    if turn_id in processed_turn_ids:
                        continue
                        
                    messages = item.get('messages', [])
                    ground_truth = item.get('answer', [])
                    answer_type = item.get('answer_type', 'SINGLE')

                    query = retriever.format_messages_to_query(messages, args.last_n_turns)
                    
                    if not query.strip():
                        logger.warning(f"Empty query for turn {turn_id}")
                        continue

                    method = getattr(retriever, method_func)
                    predictions = method(query, top_k=args.topk)

                    result_item = {
                        'turn_id': turn_id,
                        'messages': messages,
                        'answer': ground_truth,
                        'answer_type': answer_type,
                        'prediction_top1': predictions[:1] if predictions else [],
                        'prediction_top2': predictions[:2] if predictions else [],
                        'prediction_top3': predictions[:3] if predictions else [],
                        'prediction_top4': predictions[:4] if predictions else [],
                        'prediction_top5': predictions[:5] if predictions else [],
                        'query_used': query
                    }

                    temp_file = output_file + '.tmp'
                    try:
                        with jsonlines.open(temp_file, 'w') as writer:
                            writer.write(result_item)

                        with open(temp_file, 'r', encoding='utf-8') as temp_f:
                            temp_content = temp_f.read()
                        with open(output_file, 'a', encoding='utf-8') as main_f:
                            main_f.write(temp_content)

                        os.remove(temp_file)
                        
                    except Exception as e:
                        logger.error(f"Error saving result for turn {turn_id}: {e}")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    
                    processed_count += 1

                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count}/{len(test_data)} samples")
                        
                except Exception as e:
                    logger.error(f"Error processing turn {item.get('turn_id', i)}: {e}")
                    continue
            
            logger.info(f"Completed {method_name} for {pool_name}")
            logger.info(f"Total processed samples: {processed_count}")

    logger.info("Running evaluation and generating summary...")
    run_evaluation_and_summary(args.output_dir, methods, args.pool_types)

def main():
    parser = argparse.ArgumentParser(description="Run hierarchical workflow retrieval experiments")
    parser.add_argument("--input", type=str, required=True, help="Path to the turn-level JSONL file")
    parser.add_argument("--output-dir", type=str, default="outputs_w_llm", help="Directory to store outputs")
    parser.add_argument("--topk", type=int, default=5, help="Number of predictions to output per turn")
    parser.add_argument("--methods", nargs='+', default=["hier_domain", "hier_role", "hier_domain_role"], 
                       choices=["hier_domain", "hier_role", "hier_domain_role"],
                       help="Retrieval methods to run")
    parser.add_argument("--pool-types", nargs='+', default=["text", "code", "flowchart", "summary"],
                       choices=["text", "code", "flowchart", "summary"],
                       help="Which scenario pools to use")
    parser.add_argument("--last-n-turns", type=int, default=0, 
                       help="Use only last n turns (0=all, 1=last 1 turn, 2=last 2 turns, etc)")
    
    args = parser.parse_args()
    run_hierarchical_experiments(args)

if __name__ == "__main__":
    main()
