import jsonlines

turn_data = []
with jsonlines.open("./data/turn_level_data_final_w_last_workflow.jsonl") as reader:
    for obj in reader:
        turn_data.append(obj)

print(f"total number of turns: {len(turn_data)}")
cnt_dict = {
    "SINGLE": 0,
    "AND": 0,
    "OR": 0,
    "UNK": 0,
    "stay": 0
}
for turn in turn_data:
    if  not isinstance(turn["messages"], list):
        raise ValueError("messages is not a list")
        context = "\n".join([f"{item['role']}: {item['content']}" for item in turn["messages"]])
        # for message in turn["messages"]:
        #     if not isinstance(message, dict):
                # print(message)
    if turn["last_turn_workflow"] == turn["answer"] and turn["answer_type"] != "UNK":
        cnt_dict["stay"] += 1
    if "answer_type" not in turn:
        raise ValueError("turn does not have answer_type")
    elif turn["answer_type"] == "UNK":
        cnt_dict["UNK"] += 1
    elif turn["answer_type"] == "SINGLE":
        cnt_dict["SINGLE"] += 1
    elif turn["answer_type"] == "AND":
        cnt_dict["AND"] += 1
    elif turn["answer_type"] == "OR":
        cnt_dict["OR"] += 1
print(cnt_dict)

