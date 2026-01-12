import os
import uuid
import asyncio
import argparse
import random
import json
import jsonlines
from loguru import logger
from agent import WorkflowAgent
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from utils import save_jsonl, load_jsonl
from retrieval_wo_llm import build_full_history, build_last_history
# from retrieval_w_llm import HierarchicalRetriever
from retrieval_w_llm_new import HierarchicalRetriever
QUERY_TYPE = ["full", "last1", "last2", "last3", "search_query"]
POOL_TYPE = ["text", "code", "flowchart", "summary"]
METHODS = ["hier_domain", "hier_role", "hier_domain_role"]
def search_w_all_settings_wo_llm(agent, sample, pool_type, if_bm25=False):
    ret = {}
    for query_type in QUERY_TYPE:
        ret[query_type] = {}
        if query_type == "search_query":
            query = sample["response"]["search_query"]
        elif query_type == "full":
            query = build_full_history(sample["messages"])
        elif query_type == "last1":
            query = build_last_history(sample["messages"], max_turns=1)
        elif query_type == "last2":
            query = build_last_history(sample["messages"], max_turns=2)
        elif query_type == "last3":
            query = build_last_history(sample["messages"], max_turns=3)
        ret[query_type]["naive"] = agent.get_workflows(
            str(query),
            if_llm=False,
            if_naive=True,
            threshold=12,
            pool_type=pool_type,
            if_bm25=if_bm25,
        )
        ret[query_type]["hier_domain"] = agent.get_workflows(
            query,
            if_llm=False,
            if_domain=True,
            threshold=12,
            pool_type=pool_type,
            if_bm25=if_bm25,
        )
        ret[query_type]["hier_role"] = agent.get_workflows(
            query,
            if_llm=False,
            if_role=True,
            threshold=12,
            pool_type=pool_type,
            if_bm25=if_bm25,
        )
        ret[query_type]["hier_domain_role"] = agent.get_workflows(
            query,
            if_llm=False,
            if_domain_role=True,
            threshold=12,
            pool_type=pool_type,
            if_bm25=if_bm25,
        )
    return ret


async def run_all(args):
    agent = WorkflowAgent()
    sample_input = jsonlines.open(args.input_path)
    workflows = json.load(open(args.workflows_path))
    output = []
    tasks_inference = []
    id2sample = {}
    for sample in sample_input:
        messages = sample["messages"]
        last_turn_workflow = "\n".join(
            [
                workflows[last_workflow]["workflow"]
                for last_workflow in sample["last_turn_workflow"]
            ]
        )
        turn_id = sample["turn_id"]
        id2sample[turn_id] = sample
        if last_turn_workflow.strip() == "":
            last_turn_workflow = "empty"
        if args.enable_thinking == "True":
            tasks_inference.append(agent.aget_response(
                                    messages, last_turn_workflow, enable_thinking=True, turn_id=turn_id
                                ))
        else:
            tasks_inference.append(agent.aget_response(
                                    messages, last_turn_workflow, enable_thinking=False, turn_id=turn_id
                                ))
    final_output = []
    batch_size = 10
    n = len(tasks_inference)
    pbar = tqdm(total=n, desc=f"Running with Workflow Agent Async in format {args.pool_type}...")
    for i in range(0, n, batch_size):
        # 這裡會自動處理最後一批不足 batch_size 的情況
        coros_batch = tasks_inference[i:i + batch_size]

        # 重要：把每個 coroutine 包成「這一批專用」的 Task，避免重複 await
        batch_tasks = [asyncio.create_task(c) for c in coros_batch]

        # 先完成先處理
        for fut in asyncio.as_completed(batch_tasks):
            response, turn_id = await fut
            sample = id2sample[turn_id]
            sample["response"] = response
            output.append(sample)
            pbar.update(1)  # 更新全域進度條

        # 批與批之間節流（可視需要移除/調整）
        await asyncio.sleep(3)

    pbar.close()
    save_jsonl(output, args.output_path)

    for sample in tqdm(output,total=len(output), desc="Running Retrieval..."):
        response = sample["response"]
        if response["action"] == "search":
            if args.if_llm == "True":
                raise NotImplementedError
            else:
                sample["preds_all"] = search_w_all_settings_wo_llm(agent, sample, args.pool_type)
            pbar.update(1)
        else:
            logger.info("Skipped retrieval since the agent decided to not search")
            pbar.update(1)
        final_output.append(sample)
    save_jsonl(final_output, args.output_path)

def run_single(args):
    agent = WorkflowAgent()
    sample_input = jsonlines.open(args.input_path)
    workflows = json.load(open(args.workflows_path))
    sample_input = random.sample([sample for sample in sample_input if sample["answer_type"] == args.answer_type], 1)
    for sample in random.sample(list(sample_input), 1):
        messages = sample["messages"]
        last_turn_workflow = "\n".join(
            [
                workflows[last_workflow]["workflow"]
                for last_workflow in sample["last_turn_workflow"]
            ]
        )
        print(f"Chat History:\n{'\n'.join([f'{msg['role']}: {msg['content']}' for msg in messages])}")
        print(f"Last workflow:{sample['last_turn_workflow']}")
        print(f"Last workflow name: {' '.join([workflows[last_workflow]['scenario_name'] for last_workflow in sample['last_turn_workflow']])}")
        if last_turn_workflow.strip() == "":
            last_turn_workflow = "empty"
        response = agent.get_response(
            messages, last_turn_workflow, enable_thinking=True
        )
        print(f"Prediction: {json.dumps(response, indent=4)}")
        print(f"Ground Truth: {sample['answer']}")

async def run_all_wo_llm(args):
    agent = WorkflowAgent()
    input = load_jsonl(args.input_path)
    final_output = []
    for sample in tqdm(input,total=len(input), desc="Running Retrieval..."):
        response = sample["response"]
        if response["action"] == "search":
            if args.if_llm == "True":
                raise NotImplementedError
            else:
                sample["preds_all_bm25"] = search_w_all_settings_wo_llm(agent, sample, args.pool_type, if_bm25=True)
                sample["preds_all_e5"] = search_w_all_settings_wo_llm(agent, sample, args.pool_type, if_bm25=False)
        else:
            logger.info("Skipped retrieval since the agent decided to not search")
        final_output.append(sample)
    save_jsonl(final_output, args.output_path)

async def run_all_w_llm(args):
# this function only works after you have all the output from the agent and do the retrieval with LLM
    format2retriever = {}
    format2data = {}
    method_mapping = {
        "hier_domain": ("hier_domain", "retrieve_2layer_domain_scenario"),
        "hier_role": ("hier_role", "retrieve_2layer_role_scenario"),
        "hier_domain_role": ("hier_domain_role", "retrieve_3layer_domain_role_scenario")
    }
    
    methods = [(method_mapping[m][0], method_mapping[m][1]) for m in METHODS if m in method_mapping]
    format = args.pool_type
    test_data = load_jsonl(args.input_path)
    retriever_bm25 = HierarchicalRetriever(pool_name=f"workflow_{format}", use_reranker=False, retrieval_model="bm25")
    retriever_e5 = HierarchicalRetriever(pool_name=f"workflow_{format}", use_reranker=False, retrieval_model="e5")
    final_output = []
    for i, item in enumerate(tqdm(test_data, total=len(test_data), desc=f"Processing {format}")):
        # try:
        turn_id = item.get('turn_id', f'turn_{i}')
        messages = item.get('messages', [])
        res_bm25 = {}
        res_e5 = {}
        if_search = item["response"]["action"] == "search"
        search_task = []
        task2meta = {}
        if if_search:
            for query_type in QUERY_TYPE:
                if query_type == "search_query":
                    query = item["response"]["search_query"]
                    query_bm25 = query
                    query_e5 = query
                elif query_type == "full":
                    query_bm25 = retriever_bm25.format_messages_to_query(messages, 0)
                    query_e5 = retriever_e5.format_messages_to_query(messages, 0)
                elif query_type == "last1":
                    query_bm25 = retriever_bm25.format_messages_to_query(messages, 1)
                    query_e5 = retriever_e5.format_messages_to_query(messages, 1)
                elif query_type == "last2":
                    query_bm25 = retriever_bm25.format_messages_to_query(messages, 2)
                    query_e5 = retriever_e5.format_messages_to_query(messages, 2)
                elif query_type == "last3":
                    query_bm25 = retriever_bm25.format_messages_to_query(messages, 3)
                    query_e5 = retriever_e5.format_messages_to_query(messages, 3)
                # print(f"Query e5: {query_e5}, query bm25: {query_bm25}")
                if not query_e5.strip() and not query_bm25.strip():
                    logger.warning(f"Empty query for turn {turn_id}")
                    continue
                res_bm25[query_type] = {}
                res_e5[query_type] = {}
                for method_name, method_func in methods:
                    method_bm25 = getattr(retriever_bm25, method_func)
                    method_e5 = getattr(retriever_e5, method_func)
                    task_bm25 = asyncio.create_task(method_bm25(query_bm25, top_k=args.topk),name=f"{method_name}-{query_type}-bm25")
                    task_e5 = asyncio.create_task(method_e5(query_e5, top_k=args.topk),name=f"{method_name}-{query_type}-e5")
                    task2meta[task_bm25] = (method_name, query_type)
                    task2meta[task_e5] = (method_name, query_type)
                    search_task.append(task_bm25)
                    search_task.append(task_e5)

            for task in tqdm(search_task, total=len(search_task), desc=f"Processing {turn_id}"):
                predictions = await task
                method_name, query_type, retrieval_model = task.get_name().split("-")[0], task.get_name().split("-")[1], task.get_name().split("-")[2]
                if retrieval_model == "bm25":
                    res_bm25[query_type][method_name] = predictions[:5] if predictions else []
                elif retrieval_model == "e5":
                    res_e5[query_type][method_name] = predictions[:5] if predictions else []
            item["preds_all_w_llm_bm25"] = res_bm25
            item["preds_all_w_llm_e5"] = res_e5
            final_output.append(item)
            await asyncio.sleep(3)
        else:
            logger.warning(f"Agent did not search for turn {turn_id}")
            final_output.append(item)
        # if len(final_output) == 3:
        #     save_jsonl(final_output, args.output_path)
        #     return
    logger.info(f"Finished processing {format} with {len(final_output)} samples")
    save_jsonl(final_output, args.output_path)



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["all", "single", "all_w_llm", "all_wo_llm"])
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--input_folder", type=str, default="./output", help="the folder that contains all the output from the agent")
    parser.add_argument("--workflows_path", type=str)
    parser.add_argument("--output_path", type=str, default="./output.jsonl")
    parser.add_argument("--answer_type", type=str, default="UNK")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--pool_type", type=str, choices=POOL_TYPE)
    parser.add_argument("--if_llm", type=str, default="False")
    parser.add_argument("--enable_thinking", type=str, default="True")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    if args.mode == "all":
        logger.info(f"start running all with the following args: {" ".join([f"{k}={v}" for k, v in vars(args).items()])}")
        asyncio.run(run_all(args))
    elif args.mode =="all_w_llm":
        logger.info(f"start running all with the following args: {" ".join([f"{k}={v}" for k, v in vars(args).items()])}")
        asyncio.run(run_all_w_llm(args))
    elif args.mode == "all_wo_llm":
        logger.info(f"start running all with the following args: {" ".join([f"{k}={v}" for k, v in vars(args).items()])}")
        asyncio.run(run_all_wo_llm(args))
    elif args.mode == "single":
        run_single(args)    
    else:
        raise ValueError("Wrong mode")
