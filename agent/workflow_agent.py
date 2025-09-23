import random
import json
import jsonlines
from agent import WorkflowAgent


def main():
    # start agent
    agent = WorkflowAgent()
    sample_input = jsonlines.open("../data/turn_level_data_final_w_last_workflow.jsonl")
    workflows = json.load(open("../pools/workflow_text.json"))
    correct_cnt = 0
    total_cnt = 0
    for sample in sample_input:
        if sample["answer_type"] != "UNK":
            continue
        messages = sample["messages"]
        # print(
        #     "\n".join(
        #         [
        #             f"{message['role'].upper()}: {message['content']}"
        #             for message in messages
        #         ]
        #     ),
        # )
        # print("Last workflow:", sample["last_turn_workflow"])
        # print(
        #     "Last workflow name",
        #     [
        #         workflows[last_workflow]["scenario_name"]
        #         for last_workflow in sample["last_turn_workflow"]
        #     ],
        # )
        last_turn_workflow = "\n".join(
            [
                workflows[last_workflow]["workflow"]
                for last_workflow in sample["last_turn_workflow"]
            ]
        )
        if last_turn_workflow.strip() == "":
            last_turn_workflow = "empty"
        response = agent.get_response(
            messages, last_turn_workflow, enable_thinking=True
        )
        if response["action"] == "stay" and response["if_out_of_scope"]:
            correct_cnt += 1
        total_cnt += 1
        # print("Ans:", sample["answer"])
        # print("GT Action", sample["ground_truth"]["thought"])
    print(f"UNK Acc: {correct_cnt / total_cnt}")


if __name__ == "__main__":
    main()
