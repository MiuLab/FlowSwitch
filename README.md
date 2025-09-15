## Workflow Agent

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
    - load retriever (load_bm25_retriever, load_qwen_retriever, load_reranker)
    - load pool (load_pool)
```python
