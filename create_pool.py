import json
import jsonlines
import os


# from retrieval pools from "./workflow_mix.jsonl"
# Here is the pools being created
# - workflow_text
# - workflow_code
# - workflow_flowchart (skipped)
# - workflow_summary (created from workflow_text by gpt-4.1)


def create_pools():
    """
    Creates different pools from the workflow_mix.jsonl file.
    """
    workflow_text_pool = {}
    workflow_code_pool = {}
    workflow_flowchart_pool = {}
    workflow_summary_pool = {}

    with jsonlines.open("./data/workflow_final.jsonl") as reader:
        for obj in reader:
            uuid = obj["uuid"]
            scenario, domain, role = (
                obj["scenario"],
                obj["domain"],
                obj["role"],
            )
            workflow_text_pool[uuid] = {
                "domain": domain,
                "role": role,
                "scenario_name": scenario,
                "workflow": obj["workflow"]["text"],
            }
            workflow_code_pool[uuid] = {
                "domain": domain,
                "role": role,
                "scenario_name": scenario,
                "workflow": obj["workflow"]["code"],
            }
            workflow_flowchart_pool[uuid] = {
                "domain": domain,
                "role": role,
                "scenario_name": scenario,
                "workflow": obj["workflow"]["flowchart"],
            }
            workflow_summary_pool[uuid] = {
                "domain": domain,
                "role": role,
                "scenario_name": scenario,
                "workflow": obj["workflow"]["summary"],
            }

    # Create a directory to store the pools
    if not os.path.exists("pools"):
        os.makedirs("pools")

    with open("pools/workflow_text.json", "w") as f:
        json.dump(workflow_text_pool, f, indent=4)

    with open("pools/workflow_code.json", "w") as f:
        json.dump(workflow_code_pool, f, indent=4)

    with open("pools/workflow_summary.json", "w") as f:
        json.dump(workflow_summary_pool, f, indent=4)

    with open("pools/workflow_flowchart.json", "w") as f:
        json.dump(workflow_flowchart_pool, f, indent=4)


if __name__ == "__main__":
    create_pools()
    print("Pools created successfully in the 'pools' directory.")
    print(
        "Note: workflow_summary.json is a placeholder. You need to populate it with summaries."
    )
    print(
        "Note: workflow_flowchart pool was not created as no structured flowchart data was found."
    )
