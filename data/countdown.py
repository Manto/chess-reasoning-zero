"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
Taken from Jiayi-Pan/TinyZero github repo
"""

import os
from datasets import load_dataset
from random import randint, seed
from typing import List, Tuple
from tqdm import tqdm
import argparse



def build_prompt(dp):
    target = dp["target"]
    numbers = dp["nums"]

    return [
        {"role": "system", "content": "You are a helpful assistant. You first think about applying a reasoning process then provide the user with the answer."},
        {
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Put your thought process in <think> </think> tag. Show the final answer in <answer> </answer> tag, for example <answer> (1 + 2) / 3 </answer>",
        },
        {"role": "assistant", "content": "Let me solve this step by step. <think>"},
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./countdown")
    parser.add_argument("--train_size", type=int, default=327680)
    parser.add_argument("--test_size", type=int, default=1024)

    args = parser.parse_args()

    data_source = "countdown"
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # This is the dataset used in Jiayi-Pan/TinyZero project
    raw_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = build_prompt(example)
            solution = {"target": example["target"], "numbers": example["nums"]}
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
