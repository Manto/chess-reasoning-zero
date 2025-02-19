# pip install vllm
# pip install --upgrade wandb
# pip install trl[peft]
# pip install -U bitsandbytes loralib transformer
# pip install --upgrade typing_extensions

import wandb
wandb.login(key="5d52c7f836e8b019dacc5e60ab12e5e621d3b389")

# PREPPING DATASET

from datasets import load_dataset, Dataset# Building dataset
def build_prompt(dp):
    target = dp["target"]
    numbers = dp["nums"]

    return [
        {"role": "system", "content": "You are a helpful assistant. You first think about applying a reasoning process then provide the user with the answer."},
        {
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /). Each number must be used, and used only once. Put your thought process in <reasoning> </reasoning> tag. Show the final answer in <answer> </answer> tag without the equal sign, for example <answer>(1+2)/3</answer>",
        },
        {"role": "assistant", "content": "Let me solve this step by step. <reasoning>"},
    ]


data_source = "countdown"
TRAIN_SIZE = 51200
TEST_SIZE = 1024

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
            "prompt": prompt,
            "ground_truth": solution,
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }
        return data

    return process_fn

train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

# Reward Function

import re
import random


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, format_score=0.1, wrong_result_score=0.2, score=1.):
    target = ground_truth['target']
    numbers = ground_truth['numbers']

    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"===============================")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")
        print(f"--------------------------------")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return wrong_result_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, ground_truth, **kwargs) -> list[float]:
    scores = []
    for prompt, completion, truth in zip(prompts, completions, ground_truth):
        score = compute_score(completion[0]["content"], truth)
        scores.append(score)

    print(scores)
    return scores

# MODEL

from peft import LoraConfig
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
)

# TRL

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, 
    logging_steps=1,
    bf16 = True,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_generations = 2,
    max_prompt_length = 256,
    max_completion_length = 1526,
    num_train_epochs = 1,
    save_steps = 50,
    vllm_gpu_memory_utilization=0.4,
    report_to = "wandb",
    label_names=[model_name],
    output_dir="Qwen2.5-0.5B-countdown-grpo",
)

# Train

trainer = GRPOTrainer(
    model=model,
    reward_funcs=countdown_reward_func,
    args=training_args,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()