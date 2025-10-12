import json
import os
import re
import concurrent.futures
from tqdm import tqdm
from evalplus.data import write_jsonl
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))

def GEN_SOLUTION(task_describe, prompt, model):
    attempts = 0
    total_tokens_used = 0  # Initialize token counter
    model = genai.GenerativeModel(model)
    while attempts < 5:
        prompt = f"System: {task_describe}\nUser: {prompt}\nAssistant:"
        response = model.generate_content(
            prompt
        )

        # Add token usage for this request
        text = response.text
        # Approximate tokens as number of words
        total_tokens_used += len(text.split())

        # Use regular expression to match content starting with ```python and ending with ```
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip(), total_tokens_used  # Return Python code and token usage
        attempts += 1

    # If no result after 5 attempts, return an empty string and total token count
    print(f"Failed to extract valid Python code after 5 attempts for prompt: {prompt}")
    return "", total_tokens_used


def process_task(task_id, task_describe, prompt, model):
    completion, tokens_used = GEN_SOLUTION(task_describe, prompt, model)
    return dict(task_id=task_id, completion=completion), tokens_used


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)


def main(model, testset_path, mutated_prompt_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Read mutated prompt file
    mutated_prompts = {task["prompt_id"]: task["mutated_prompt"] for task in read_jsonl(mutated_prompt_path)}

    total_tokens = 0  # Initialize total tokens counter

    for prompt_id, task_describe in mutated_prompts.items():
        samples = []

        # Use multithreading to speed up task processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(
                    process_task,
                    task["id"],
                    task_describe,
                    task["input"],
                    model
                )
                for task in read_jsonl(testset_path)
            ]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc=f"Processing tasks for prompt_id {prompt_id}"):
                result, tokens_used = future.result()
                total_tokens += tokens_used  # Accumulate total tokens
                if result['completion']:
                    samples.append(result)
                else:
                    print(f"No valid completion for task_id: {result['task_id']} with prompt_id: {prompt_id}")

        output_file = os.path.join(output_path, f"train_set_{model}_{prompt_id}.jsonl")
        write_jsonl(output_file, samples)

    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    # Set command-line arguments
    parser = argparse.ArgumentParser(description="Run prompt evaluation using OpenAI API.")
    parser.add_argument('--model', type=str, required=True, help="The model to use (e.g., 'gpt-3.5-turbo').")
    parser.add_argument('--trainset_path', type=str, required=True, help="Path to the train set JSONL file.")
    parser.add_argument('--mutated_prompt_path', type=str, required=True, help="Path to the mutated prompt JSONL file.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the output files.")

    # Parse command-line arguments
    args = parser.parse_args()

    main(args.model, args.trainset_path, args.mutated_prompt_path, args.output_path)