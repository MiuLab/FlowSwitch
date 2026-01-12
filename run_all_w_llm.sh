export OPENROUTER_API_KEY=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
# python ./agent/workflow_agent.py --input_path ./output/output_wo_llm_text.jsonl --pool_type text --mode all_w_llm --output_path ./output/output_w_llm_text.jsonl 2>&1 | tee ./log/text_w_llm_retrieval.log &
# python ./agent/workflow_agent.py --input_path ./output/output_wo_llm_code.jsonl --pool_type code --mode all_w_llm --output_path ./output/output_w_llm_code.jsonl 2>&1 | tee ./log/code_w_llm_retrieval.log &
# python ./agent/workflow_agent.py --input_path ./output/output_wo_llm_flowchart.jsonl --pool_type flowchart --mode all_w_llm --output_path ./output/output_w_llm_flowchart.jsonl 2>&1 | tee ./log/flowchart_w_llm_retrieval.log &
# python ./agent/workflow_agent.py --input_path ./output/output_wo_llm_summary.jsonl --pool_type summary --mode all_w_llm --output_path ./output/output_w_llm_summary.jsonl 2>&1 | tee ./log/summary_w_llm_retrieval.log &
# wait
python ./agent/workflow_agent.py --input_path ./output_workflow_desc_w_org_prompt/output_wo_llm_text.jsonl --pool_type text --mode all_w_llm --output_path ./output_workflow_desc_w_org_prompt/output_w_llm_text.jsonl 2>&1 | tee ./output_workflow_desc_w_org_prompt/output_w_llm_text.log &
python ./agent/workflow_agent.py --input_path ./output_workflow_desc_w_org_prompt/output_wo_llm_code.jsonl --pool_type code --mode all_w_llm --output_path ./output_workflow_desc_w_org_prompt/output_w_llm_code.jsonl 2>&1 | tee ./output_workflow_desc_w_org_prompt/output_w_llm_code.log &
python ./agent/workflow_agent.py --input_path ./output_workflow_desc_w_org_prompt/output_wo_llm_flowchart.jsonl --pool_type flowchart --mode all_w_llm --output_path ./output_workflow_desc_w_org_prompt/output_w_llm_flowchart.jsonl 2>&1 | tee ./output_workflow_desc_w_org_prompt/output_w_llm_flowchart.log &
python ./agent/workflow_agent.py --input_path ./output_workflow_desc_w_org_prompt/output_wo_llm_summary.jsonl --pool_type summary --mode all_w_llm --output_path ./output_workflow_desc_w_org_prompt/output_w_llm_summary.jsonl 2>&1 | tee ./output_workflow_desc_w_org_prompt/output_w_llm_summary.log &
wait
