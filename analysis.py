from agent.utils import load_jsonl
import ipdb

org_output = load_jsonl("./output/output_wo_llm_text.jsonl")
new_output = load_jsonl("./output_workflow_desc_w_org_prompt/output_wo_llm_text.jsonl")
# sort by turn id
org_output = sorted(org_output, key=lambda x: x["turn_id"])
new_output = sorted(new_output, key=lambda x: x["turn_id"])
for i in range(len(org_output)):
    org_response = org_output[i]["response"]
    new_response = new_output[i]["response"]
    context = "\n".join([f"{item['role']}: {item['content']}" for item in org_output[i]["messages"]])
    if org_response["action"] == "stay" and new_response["action"] != "stay" and org_output[i]["answer_type"] == "UNK":
        print(f"Org id: {org_output[i]['turn_id']}")
        print(f"New id: {new_output[i]['turn_id']}")
        print(f"Context: {context}")
        print(f"Answer: {org_output[i]['answer']}")
        print(f"Last workflow: {org_output[i]['last_turn_workflow']}")
        print(f"New response: {new_response['reason']}")
        print(f"Org response: {org_response['reason']}")
        ipdb.set_trace()
