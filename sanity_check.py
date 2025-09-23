import jsonlines

turn_data = []
with jsonlines.open("./data/turn_level_data_final.jsonl") as reader:
    for obj in reader:
        turn_data.append(obj)

print(f"total number of turns: {len(turn_data)}")
for turn in turn_data:
    if turn["answer_type"] == "UNK":
        assert turn["answer"] == ["Out of scope"]
    if turn["answer_type"] == "SINGLE":
        assert len(turn["answer"]) == 1
    if turn["answer_type"] == "AND" or turn["answer_type"] == "OR":
        assert len(turn["answer"]) >= 1 and len(turn["answer"]) < 3
