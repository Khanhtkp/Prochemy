import os
import json
import re
import httpx
import concurrent.futures
from tqdm import tqdm
import argparse
import random
from evalplus.data import write_jsonl
from dotenv import load_dotenv
from google import generativeai as genai
load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))


task_describe = """
You are an expert prompt engineer.
"""

information = """
Please help me improve the given prompt to get a more helpful and harmless response.
Suppose I need to complete a Python program based on previously given code snippet.
The completed code should be fully functional, logically consistent, and integrate seamlessly with the existing code.\n
"""

format = """
You may add any information you think will help improve the task's effectiveness during the prompt optimization process.
If you find certain expressions and wording in the original prompt inappropriate, you can also modify these usages.
Ensure that the optimized prompt includes a detailed task description and clear process guidance added to the original prompt.
Wrap the optimized prompt in {{}}.
"""


def GEN_ANSWER(prompt, model="gemini-2.5-flash"):
    model = genai.GenerativeModel(model)
    prompt = f"System: {task_describe}\nUser: {information + prompt + format}\nAssistant:"
    response = model.generate_content(
        prompt
    )
    text = response.text
    # Approximate tokens as number of words
    tokens_used = len(text.split())
    return text, tokens_used


def extract_wrapped_content(text):
    match = re.search(r'\{\{(.*?)\}\}', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def process_task(task_id, prompt, model):
    total_tokens = 0
    while True:
        completion, tokens_used = GEN_ANSWER(prompt, model)
        total_tokens += tokens_used
        wrapped_content = extract_wrapped_content(completion)
        if wrapped_content:
            return dict(prompt_id=task_id, mutated_prompt=wrapped_content), total_tokens
        else:
            print(f"Task {task_id}: No wrapped content found. Retrying...")


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]  # Read all lines and parse them as JSON objects


def main(model, prompt_path, output_path):
    # Read the content of the jsonl file and randomly select one entry
    data = read_jsonl(prompt_path)
    random_entry = random.choice(data)

    if 'mutated_prompt' not in random_entry:
        print(f"No 'mutated_prompt' field found in the selected entry. Exiting...")
        return

    prompt = random_entry['mutated_prompt']  # Use the randomly selected 'mutated_prompt' field as the prompt

    samples = []
    total_tokens = 0
    problems = {i: {"prompt": prompt} for i in range(10)}  # Create 10 tasks

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_task, task_id, problem["prompt"], model) for task_id, problem in
                   problems.items()]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tasks"):
            result, tokens_used = future.result()
            samples.append(result)
            total_tokens += tokens_used

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    write_jsonl(output_path, samples)
    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description="Mutate prompts and save the results.")
    parser.add_argument('--model', type=str, required=True,
                        help="The model to use for prompt generation (e.g., 'gpt-3.5-turbo').")
    parser.add_argument('--prompt_path', type=str, required=True,
                        help="The file path for the prompt (.jsonl file) to be processed.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="The file path where the output JSONL will be saved.")

    # Parse command line arguments
    args = parser.parse_args()

    # Run the main logic
    main(args.model, args.prompt_path, args.output_path)