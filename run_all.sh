python ./agent/workflow_agent.py \
    --input_path ./data/turn_level_data_final_w_last_workflow.jsonl \
    --workflows_path ./pools/workflow_text.json \
    --mode all \
    --output_path ./output/output_wo_llm_text.jsonl \
    --top-k 5 \
    --pool_type text \
    --if_llm "False"

python ./agent/workflow_agent.py \
    --input_path ./data/turn_level_data_final_w_last_workflow.jsonl \
    --workflows_path ./pools/workflow_text.json \
    --workflows_path ./pools/workflow_code.json \
    --mode all \
    --output_path ./output/output_wo_llm_code.jsonl \
    --top-k 5 \
    --pool_type code \
    --if_llm "False"

python ./agent/workflow_agent.py \
    --input_path ./data/turn_level_data_final_w_last_workflow.jsonl \
    --workflows_path ./pools/workflow_text.json \
    --workflows_path ./pools/workflow_flowchart.json \
    --mode all \
    --output_path ./output/output_wo_llm_flowchart.jsonl \
    --top-k 5 \
    --pool_type flowchart \
    --if_llm "False"

python ./agent/workflow_agent.py \
    --input_path ./data/turn_level_data_final_w_last_workflow.jsonl \
    --workflows_path ./pools/workflow_text.json \
    --workflows_path ./pools/workflow_summary.json \
    --mode all \
    --output_path ./output/output_wo_llm_summary.jsonl \
    --top-k 5 \
    --pool_type summary \
    --if_llm "False"

