# Evalutation script for retrieval
"""
Usage:
python eval.py --input <input_file> --output <output_file>

Your input file should be in jsonl format, and each line should be a json object with the following keys:
- answer: a list of uuid of the gt workflows
- answer type: the answer type of the gt workflows (AND, OR, SINGLE, UNK)
- prediction: a list of uuid of the predicted workflows

The output file will be in json format, and each line will be a json object as the following:
[
    {
        "answer_type": <answer type>,
        "accuracy": <accuracy>
    },
    ...

]
"""

import argparse
from utils import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.input, args.output)


if __name__ == "__main__":
    main()
