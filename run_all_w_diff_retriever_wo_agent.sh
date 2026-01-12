# 只用BM25，不用reranker(all_turn)
# python ./agent/retrieval_w_llm_new.py --input data/turn_level_data_final.jsonl --output-dir wo_reranker_bm25_all --use-reranker false --retrieval-model bm25
# 用E5模型，不用reranker(all_turn)
python ./agent/retrieval_w_llm_new.py --input data/turn_level_data_final.jsonl --output-dir wo_reranker_e5_all --use-reranker false --retrieval-model e5 --last-n-turns 2
python ./agent/retrieval_w_llm_new.py --input data/turn_level_data_final.jsonl --output-dir wo_reranker_e5_all --use-reranker false --retrieval-model e5 --last-n-turns 3
