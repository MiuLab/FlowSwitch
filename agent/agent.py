import json
import torch
from loguru import logger
from request_llm import request_llm_vllm_chat, arequest_llm_vllm_chat
from request_qwen import request_qwen_chat
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from retrieval_wo_llm_new import (
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
    run_naive_e5,
    run_hier_domain_e5,
    run_hier_role_e5,
    run_hier_domain_role_e5,
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
        self.system_message_wf_desc = """### Insturctions:
        Your goal is to help the user complete their task according to different workflow SOPs.
        In order to accomplish this, you will need to understand the user's intention and determine the appropriate workflow SOPs to follow.
        Specifically, given the current dialogue context and current workflwo SOP, you will need to decide:
        1. Whether the user's intention is aligned with the current workflow SOPs.
            a. If the answer is no, based on your own knowledge, does current user's intention has to do with any potential tasks that might be described as a workflow SOP?
                i. If yes, you will need generate a suitable search query to find the appropriate workflow SOP.
                ii. If the answer is partially yes, you will need to search for any other workflow SOP that may be relevant to the user's intention.
                iii. If no, you will need to stay with the current workflow, if the current context has nothing to do with any possible workflow SOPs and is out of current workflow's scope.
            b. If the answer is yes, you will need to stay with the current workflow.
        2. Note that All you have to do is to decide which action to take, you do not need to take any other actions such as calling functions.
            You only have 2 actions to choose from: 
            a. search: search for a suitable workflow SOP
            b. stay: stay with the current workflow
        3. It is possible that the current workflow SOPs is empty, then you will need to determine whether to search for suitable workflow SOPs or stay with the current workflow.
        4. If you decide to search, the search query should be a clear and precise descripion of such workflow that can be used to tackle the user's intention, this should include the following information:
                a. Potential Name of the workflow
                b. Task description of the workflow
                c. the action you need to take to solve the task
            For example, the search query should be a string as follows:
            "Potential Name of the workflow: Workflow Name, Task description of the workflow: Task Description, the action you need to take to solve the task: Action"
        ### Current Workflow SOPs:
        {current_workflow_sop}

        ## Dialogue Context:
        {dialogue_context}
        
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
        self.system_message_wo_thinking_wf_desc = """### Insturctions:
        Your goal is to help the user complete their task according to different workflow SOPs.
        In order to accomplish this, you will need to understand the user's intention and determine the appropriate workflow SOPs to follow.
        Specifically, given the current dialogue context and current workflwo SOP, you will need to decide:
        1. Whether the user's intention is aligned with the current workflow SOPs.
            a. If the answer is no, based on your own knowledge, does current user's intention has to do with any potential tasks that might be described as a workflow SOP?
                i. If yes, you will need generate a suitable search query to find the appropriate workflow SOP.
                ii. If the answer is partially yes, you will need to search for any other workflow SOP that may be relevant to the user's intention.
                iii. If no, you will need to stay with the current workflow, if the current context has nothing to do with any possible workflow SOPs and is out of current workflow's scope.
            b. If the answer is yes, you will need to stay with the current workflow.
        2. Note that All you have to do is to decide which action to take, you do not need to take any other actions such as calling functions.
            You only have 2 actions to choose from: 
            a. search: search for a suitable workflow SOP
            b. stay: stay with the current workflow
        3. It is possible that the current workflow SOPs is empty, then you will need to determine whether to search for suitable workflow SOPs or stay with the current workflow.
        4. If you decide to search, the search query should be a clear and precise descripion of such workflow that can be used to tackle the user's intention, this should include the following information:
                a. Potential Name of the workflow
                b. Task description of the workflow
                c. the action you need to take to solve the task
            For example, the search query should be a string as follows:
            "Potential Name of the workflow: Workflow Name, Task description of the workflow: Task Description, the action you need to take to solve the task: Action"
        ### Current Workflow SOPs:
        {current_workflow_sop}

        ## Dialogue Context:
        {dialogue_context}
        
        ### Output Format:
        Follow the below format in every response under any circumstances:
        ```json
        {{
            "reasoning": "<provide_your_reasoning_here>",
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
    async def aget_response(
        self,
        context,
        current_workflow_sop,
        enable_thinking=False,
        retry=10,
        with_reason_output=True,
        turn_id=None,
    ):
        config = (
            self.agent_kwargs_w_thinking
            if enable_thinking
            else self.agent_kwargs_wo_thinking
        )
        context_str = "\n".join([f"{item['role']}: {item['content']}" for item in context])
        if enable_thinking:
            system_message = self.system_message_wf_desc.format(
                current_workflow_sop=current_workflow_sop, dialogue_context=context_str
            )
        else:
            system_message = self.system_message_wo_thinking_wf_desc.format(
                current_workflow_sop=current_workflow_sop, dialogue_context=context
            )
        messages = [{"role": "system", "content": system_message}]

        # response = (
        #     await arequest_llm_vllm_chat(messages, enable_thinking=enable_thinking, **config)
        # )
        response = await request_qwen_chat(messages, model="qwen/qwen3-14b",enable_thinking=enable_thinking, **config)
        reasoning = response["response"]['message']["reasoning"] if enable_thinking else ""
        response = response["response"]["message"]["content"].split("assistant:")[-1].strip()
        response = response.replace("```json", "").replace("```", "")
        try:
            response = json.loads(response)
            if with_reason_output:
                response["reason"] = reasoning
            return response, turn_id
        except:
            if retry > 0:
                return await self.aget_response(
                    context,
                    current_workflow_sop,
                    enable_thinking=enable_thinking,
                    retry=retry - 1,
                    turn_id=turn_id,
                )
            else:
                logger.error(f"Failed to parse response: {response}")
                return {
                    "action": "None",
                    "search_query": "",
                    "user_intention": "",
                    "reason": "",
                }, turn_id

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
                response["reason"] = reasoning
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
                logger.error(f"Failed to parse response: {response}")
                return {
                    "action": "None",
                    "search_query": "",
                    "user_intention": "",
                    "reason": "",
                }

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
        if_bm25: bool = False,
    ):
        scenario_index = self.scenario_indexes[pool_type]
        if if_llm:
            pass
        elif not if_bm25:
            if if_naive:
                ret = run_naive_e5(
                    query,
                    scenario_index,
                    top_k=top_k,
                )
            elif if_domain:
                ret = run_hier_domain_e5(
                    query,
                    scenario_index,
                    self.domain_index,
                    self.domain_to_ids,
                    top_k=top_k,
                    domain_top_n=3,
                )
            elif if_role:
                ret = run_hier_role_e5(
                    query,
                    scenario_index,
                    self.role_index,
                    self.role_to_ids,
                    top_k=top_k,
                    role_top_n=3,
                )
            elif if_domain_role:
                ret = run_hier_domain_role_e5(
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

        preds = [doc_id for doc_id, _ in ret[:top_k]]
        return preds
