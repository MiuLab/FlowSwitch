## Workflow Agent
WIP...
### Data Preparation
```bash

export <your openai api key>
# create domain and role descriptions
python create_desc4domainrole.py
# create pools for different formats
python create_pool.py
# the data will be saved in ./pools/
```
### Utils
Use the function in `utils.py` for the following:
- basic i/o (load_json, load_jsonl, save_json, save_jsonl)
- load retriever (load_bm25_retriever, load_qwen_retriever, load_reranker)
- load pool (load_pool)
### Prediction/Evaluation Instructions
After the retrieval, the final prediction should be appended as a key "prediction" into the original jsonl file.
And each of the json objects in the jsonl file must contain the following keys, before running the `eval.py`:
- answer: the gt workflows' uuid
- answer_type: the type of answer (AND, OR, SINGLE, UNK)
- prediction_top1: the predicted workflows' uuid at top 1
- prediction_top2: the predicted workflows' uuid at top 2
.... til topk

For example:
input.jsonl must be like:
```bash
{..., "answer": ["uuid1", "uuid2"], "prediction": ["uuid1", "uuid3"]}
{..., "answer": ["uuid1", "uuid5"], "prediction": ["uuid4", "uuid3"]}
...
```



