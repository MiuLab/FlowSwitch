import json
import torch
from request_llm import request_llm_vllm_chat, request_llm_vllm_completion
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# from retrieval_w_llm import HierarchicalRetriever
from retrieval_wo_llm import (
    POOL_NAME_MAP,
    CANDIDATE_MULTIPLIER,
    build_metadata,
    build_scenario_index,
    BM25Index,
    rerank_candidates,
    run_naive,
    run_hier_domain,
    run_hier_role,
    run_hier_domain_role,
)
from utils import load_pool, load_reranker


class WorkflowAgent:
    def __init__(self):
        self.agent_name = "Workflow Agent"
        self.system_message = """### Insturctions:
        You are an agent designed to interact with a user in a conversational manner.
        Your goal is to help the user complete their task according to different workflow SOPs.
        In order to accomplish this, you will need to understand the user's intention and determine the appropriate workflow SOPs to follow.
        Specifically, given the current dialogue context, current workflwo SOP and the user's intention, you will need to decide:
        1. Whether the user's intention is aligned with the current workflow SOPs.
            a. If the answer is no, based on your own knowledge, does current user's intention has to do with any potential tasks that will be described as a SOP?
                i. If yes, you will need generate a suitable search query to find the appropriate workflow SOP.
                ii. If no, you will need to respond to the user directly, if and only if the user's intention is not related to any possible workflow SOP.
                iii. If the answer is partially yes, you will need to search for any other workflow SOP that may be relevant to the user's intention.
            b. If the answer is yes, you will need to respond to the user directly without any further action.
        2. Note that All you have to do is to decide which action to take, you do not need to worry about any availability of the actions/functions.
        3. It is possible that the current workflow SOPs is empty, then you will need to determine whether to search for suitable workflow SOPs or respond to the user directly.
        ### Current Workflow SOPs:
        {current_workflow_sop}
        
        ### Output Format:
        Follow the below format in every response under any circumstances:
        ```json
        {{
            "action": "<search, stay>",
            "search_query": "<search query>",(empty if action is not search)
            "user_intention": "<user intention>"
        }}
        ```
        ### Reponse:
        """
        self.system_message_wo_thinking = """### Insturctions:
        You are an agent designed to interact with a user in a conversational manner.
        Your goal is to help the user complete their task according to different workflow SOPs.
        In order to accomplish this, you will need to understand the user's intention and determine the appropriate workflow SOPs to follow.
        Specifically, given the current dialogue context, current workflwo SOPs and the user's intention, you will need to decide:
        1. Whether the user's intention is aligned with the current workflow SOPs.
            a. If the answer is no, based on your own knowledge, does current user's intention has to do with any potential tasks that will be described as a SOP?
                i. If yes, you will need generate a suitable search query to find the appropriate workflow SOP.
                ii. If no, you will need to respond to the user directly, if and only if the user's intention is not related to any possible workflow SOP.
                iii. If the answer is partially yes, you will need to search for any other workflow SOP that may be relevant to the user's intention.
            b. If the answer is yes, you will need to respond to the user directly without any further action.
        2. Note that All you have to do is to decide which action to take, you do not need to worry about any availability of the actions/functions.
        3. It is possible that the current workflow SOP is empty, then you will need to determine whether to search for a suitable workflow SOP or respond to the user directly.
        ### Current Workflow SOPs:
        {current_workflow_sop}
        
        ### Output Format:
        Follow the below format in every response under any circumstances:
        ```json
        {{
            "reason": "<provide_your_reasoning_here>",
            "action": "<search, stay>",
            "search_query": "<search query>",(empty if action is not search)
            "user_intention": "<user intention>"
        }}
        ```
        ### Reponse:
        """
        self.agent_kwargs_w_thinking = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "max_tokens": 9000,
        }
        self.agent_kwargs_wo_thinking = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
            "max_tokens": 9000,
        }
        # ===========setting for retrieval without LLM==============
        self.reranker_model = load_reranker()

        self.domain_desc = load_pool("domain_desc")
        self.role_desc = load_pool("role_desc")
        self.domain_index = BM25Index(
            [(domain, desc) for domain, desc in self.domain_desc.items()]
        )
        self.role_index = BM25Index(
            [(role, desc) for role, desc in self.role_desc.items()]
        )
        base_pool = load_pool(POOL_NAME_MAP["text"])
        self.domain_to_ids, self.role_to_ids, self.domain_role_to_ids = build_metadata(
            base_pool
        )

        self.scenario_indexes: Dict[str, BM25Index] = {}
        self.scenario_embeddings_map: Dict[str, Dict[str, torch.Tensor]] = {}
        for pool_key in POOL_NAME_MAP.keys():
            pool_name = POOL_NAME_MAP[pool_key]
            index, _, embeddings = build_scenario_index(pool_name, self.reranker_model)
            self.scenario_indexes[pool_key] = index
            self.scenario_embeddings_map[pool_key] = embeddings
        # ===========setting for retrieval without LLM==============

    def get_response(
        self,
        context,
        current_workflow_sop,
        enable_thinking=False,
        retry=10,
        with_reason_output=True,
    ):
        config = (
            self.agent_kwargs_w_thinking
            if enable_thinking
            else self.agent_kwargs_wo_thinking
        )
        if enable_thinking:
            system_message = self.system_message.format(
                current_workflow_sop=current_workflow_sop
            )
        else:
            system_message = self.system_message_wo_thinking.format(
                current_workflow_sop=current_workflow_sop
            )
        messages = [{"role": "system", "content": system_message}] + context

        response = (
            request_llm_vllm_chat(messages, enable_thinking=enable_thinking, **config)
            .message.content.split("assistant:")[-1]
            .strip()
        )
        reasoning = response.split("</think>")[0].strip().replace("<think>", "").strip()
        response = (
            response.split("</think>")[-1]
            .strip()
            .replace("```json", "")
            .replace("```", "")
        )
        try:
            response = json.loads(response)
            if with_reason_output:
                response["reasoning"] = reasoning
            return response
        except:
            if retry > 0:
                return self.get_response(
                    context,
                    current_workflow_sop,
                    enable_thinking=enable_thinking,
                    retry=retry - 1,
                )
            else:
                raise ValueError(f"Failed to parse response: {response}")

    def get_workflows(
        self,
        query: str,
        if_llm: bool = False,
        if_naive: bool = False,
        if_domain: bool = False,
        if_role: bool = False,
        if_domain_role: bool = False,
        top_k: int = 5,
        threshold: float = 12,
        pool_type: str = "text",
    ):
        scenario_index = self.scenario_indexes[pool_type]
        scenario_embeddings = self.scenario_embeddings_map[pool_type]
        if if_llm:
            pass
        else:
            if if_naive:
                ret = run_naive(
                    query,
                    scenario_index,
                    top_k=top_k,
                )
                ret = ret[: max(top_k * CANDIDATE_MULTIPLIER, top_k)]
            elif if_domain:
                ret = run_hier_domain(
                    query,
                    scenario_index,
                    self.domain_index,
                    self.domain_to_ids,
                    top_k=top_k,
                    domain_top_n=3,
                )
            elif if_role:
                ret = run_hier_role(
                    query,
                    scenario_index,
                    self.role_index,
                    self.role_to_ids,
                    top_k=top_k,
                    role_top_n=3,
                )
            elif if_domain_role:
                ret = run_hier_domain_role(
                    query,
                    scenario_index,
                    self.domain_index,
                    self.role_index,
                    self.domain_role_to_ids,
                    top_k=top_k,
                    domain_top_n=3,
                    role_top_n=3,
                )
            else:
                raise ValueError("Please specify the retrieval method")

            scenario_embeddings = self.scenario_embeddings_map[pool_type]
            pred_pairs = rerank_candidates(
                query,
                ret,
                self.reranker_model,
                scenario_embeddings,
                top_k,
            )
            preds = [doc_id for doc_id, _ in pred_pairs]
            bm25_top_score = ret[0][1] if ret else float("-inf")
            if bm25_top_score < threshold:
                preds = []

            return preds
