import sys
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.map.llm_handler import LLMHandler
from src.map.strategy_executor import run_strategy
# [MODIFIED] Import central data loaders
from src.utils.data_loader import (
    load_gsm8k, load_drop, load_hotpotqa, load_game_of_24,
    load_mbpp, load_humaneval, load_trivia_cw
)


def main(benchmark_name: str, limit: int):
    # This strategy name should correspond to the prompt file name, e.g., 'tot.md'
    strategy_to_run = "tot" 
    print(f"Starting Baseline Experiment: Strategy='{strategy_to_run}', Benchmark='{benchmark_name}'")

    # [MODIFIED] Use the central data loader
    benchmark_name_lower = benchmark_name.lower()
    loader_map = {
        'gsm8k': lambda: load_gsm8k(split="test"),
        'drop': lambda: load_drop(split="validation"),
        'hotpotqa': lambda: load_hotpotqa(split="validation"),
        'game_of_24': lambda: load_game_of_24(split="test"),
        'mbpp': lambda: load_mbpp(split="test"),
        'humaneval': lambda: load_humaneval(split="test"),
        'trivia_cw': lambda: load_trivia_cw(split="test")
    }
    loader = loader_map.get(benchmark_name_lower)
    if not loader:
        raise ValueError(f"Unknown or unsupported benchmark: {benchmark_name}")
    problems = loader()

    if not problems:
        print("No problems loaded. Aborting experiment.")
        return

    if limit > 0 and len(problems) > limit:
        print(f"Limiting benchmark from {len(problems)} to the first {limit} problems.")
        problems = problems[:limit]

    llm_handler = LLMHandler()

    print(f"Running '{strategy_to_run}' strategy on {len(problems)} problems...")
    results = []
    for problem in tqdm(problems, desc=f"Running {strategy_to_run}"):
        try:
            question = problem['question']
            context = problem.get('context')
            correct_answer = problem.get('answer', 'N/A')
            
            # [MODIFIED] Handle tuple return and record tokens
            generated_answer, tokens = run_strategy(llm_handler, strategy_to_run, question, context)
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "total_tokens": json.dumps(tokens) # Add token info
            })
        except Exception as e:
            print(f"\nAn error occurred while processing a problem: {e}. Skipping to the next one.")
            results.append({
                "question": problem.get('question', 'N/A'),
                "correct_answer": problem.get('answer', 'N/A'),
                "generated_answer": f"EXECUTION_ERROR: {e}",
                "total_tokens": "{}" # Add token info
            })
            continue

    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"results_{strategy_to_run.replace(' ', '_')}_{benchmark_name}_{timestamp}.csv"
    file_path = results_dir / file_name

    try:
        # [MODIFIED] Add total_tokens to the CSV fieldnames
        fieldnames = ["question", "correct_answer", "generated_answer", "total_tokens"]
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Experiment finished. Results saved to '{file_path}'")
    except Exception as e:
        print(f"Failed to save results. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a baseline experiment for a single strategy.")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        required=True,
        # [MODIFIED] Add choices for user convenience
        choices=['gsm8k', 'drop', 'hotpotqa', 'game_of_24', 'mbpp', 'humaneval', 'trivia_cw'],
        help="The benchmark to use."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0,
        help="Limit the number of problems to run. Default is 0 (run all)."
    )
    
    args = parser.parse_args()
    main(args.benchmark, args.limit)