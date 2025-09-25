import argparse
import random
import json
import jsonlines
from loguru import logger
from agent import WorkflowAgent
from tqdm import tqdm
from utils import save_jsonl
from retrieval_wo_llm import build_full_history, build_last_history
QUERY_TYPE = ["full", "last1", "last2", "last3", "search_query"]
POOL_TYPE = ["text", "code", "flowchart", "summary"]

def search_w_all_settings_wo_llm(agent, sample, pool_type):
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
        ret[query_type]["preds_naive"] = agent.get_workflows(
            query,
            if_llm=False,
            if_naive=True,
            threshold=12,
            pool_type=pool_type,
        )
        ret[query_type]["preds_hier_domain"] = agent.get_workflows(
            query,
            if_llm=False,
            if_domain=True,
            threshold=12,
            pool_type=pool_type,
        )
        ret[query_type]["preds_hier_role"] = agent.get_workflows(
            query,
            if_llm=False,
            if_role=True,
            threshold=12,
            pool_type=pool_type,
        )
        ret[query_type]["preds_hier_domain_role"] = agent.get_workflows(
            query,
            if_llm=False,
            if_domain_role=True,
            threshold=12,
            pool_type=pool_type,
        )
    return ret


def run_all(args):
    agent = WorkflowAgent()
    sample_input = jsonlines.open(args.input_path)
    workflows = json.load(open(args.workflows_path))
    pbar = tqdm(sample_input, desc="Running with Workflow Agent")
    for sample in pbar:
        messages = sample["messages"]
        last_turn_workflow = "\n".join(
            [
                workflows[last_workflow]["workflow"]
                for last_workflow in sample["last_turn_workflow"]
            ]
        )
        if last_turn_workflow.strip() == "":
            last_turn_workflow = "empty"
        if args.enable_thinking == "True":
            response = agent.get_response(
                messages, last_turn_workflow, enable_thinking=True
            )
        else:
            response = agent.get_response(
                messages, last_turn_workflow, enable_thinking=False
            )
        sample["response"] = response
        # run retrieval in different settings
        if response["action"] == "search":
            if args.if_llm == "True":
                raise NotImplementedError
            else:
                sample["preds_all"] = search_w_all_settings_wo_llm(agent, sample, args.pool_type)
            pbar.update(1)
        else:
            logger.info("Skipped retrieval since the agent decided to not search")
            pbar.update(1)
            continue
    save_jsonl(sample_input, args.output_path)

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


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["all", "single"])
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--workflows_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./output.jsonl")
    parser.add_argument("--answer_type", type=str, default="UNK")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pool_type", type=str, required=True, choices=POOL_TYPE)
    parser.add_argument("--if_llm", type=str, default="False")
    parser.add_argument("--enable_thinking", type=str, default="True")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    if args.mode == "all":
        logger.info(f"start running all with the following args: {" ".join([f"{k}={v}" for k, v in vars(args).items()])}")
        run_all(args)
    elif args.mode == "single":
        run_single(args)    
    else:
        raise ValueError("Wrong mode")
