import sys
import csv
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.map.llm_handler import LLMHandler
from src.map.strategy_executor import run_strategy
from src.utils.data_loader import (
    load_gsm8k, load_drop, load_hotpotqa, load_game_of_24,
    load_mbpp, load_humaneval, load_trivia_cw
)


def main(benchmark_name: str, limit: int):
    print(f"Starting MRP (Meta-Reasoning Prompting) Experiment on Benchmark: '{benchmark_name}'")

    # 1. Initialize Handlers and Load MRP-specific Prompt
    llm_handler = LLMHandler()
    try:
        prompt_dir = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"
        # MRP 전용 프롬프트 파일을 불러옵니다.
        meta_prompt_template = (prompt_dir / "mrp_evaluation.md").read_text(encoding='utf-8')
        print("MRP-specific prompt template loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find the MRP prompt file. {e}")
        return

    # 2. Load Benchmark Data
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

    # 3. Run MRP loop
    print(f"Running MRP on {len(problems)} problems...")
    results = []
    for problem in tqdm(problems, desc=f"Running MRP on {benchmark_name}"):
        total_tokens = {}
        try:
            question = problem['question']
            context = problem.get('context')
            correct_answer = problem.get('answer', 'N/A')
            
            # === MRP Phase 1: Meta-Reasoning and Selection ===
            input_query = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question
            meta_prompt = meta_prompt_template.replace("{{user_query}}", input_query)
            
            selection_output_str, selection_tokens = llm_handler.invoke(meta_prompt)
            total_tokens.update(selection_tokens)

            # 단순 텍스트 파싱 로직
            selected_strategy = "cot" # 파싱 실패 시 기본값
            mrp_log = {"raw_output": selection_output_str}
            try:
                # ">> FINAL CHOICE:" 라인에서 전략 이름을 추출
                match = re.search(r'>> FINAL CHOICE:\s*([a-zA-Z_ -]+)', selection_output_str)
                if match:
                    selected_strategy = match.group(1).strip()
                mrp_log["selected_strategy"] = selected_strategy
            except Exception as e:
                mrp_log["error"] = f"Failed to parse FINAL CHOICE: {e}"

            # === MRP Phase 2: Execution ===
            generated_answer, execution_tokens = run_strategy(llm_handler, selected_strategy, question, context)
            total_tokens.update(execution_tokens)
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "mrp_log": json.dumps(mrp_log),
                "total_tokens": json.dumps(total_tokens)
            })
        except Exception as e:
            print(f"\nAn error occurred while processing a problem: {e}. Skipping.")
            results.append({
                "question": problem.get('question', 'N/A'),
                "correct_answer": problem.get('answer', 'N/A'),
                "generated_answer": f"EXECUTION_ERROR: {e}",
                "mrp_log": json.dumps({"error": str(e)}),
                "total_tokens": "{}"
            })
            continue

    # 4. Save Results
    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%HM%S")
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"results_MRP_{benchmark_name}_{timestamp}.csv"
    file_path = results_dir / file_name

    try:
        fieldnames = ["question", "correct_answer", "generated_answer", "mrp_log", "total_tokens"]
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Experiment finished. Results saved to '{file_path}'")
    except Exception as e:
        print(f"Failed to save results. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MRP (Meta-Reasoning Prompting) baseline experiment.")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        required=True,
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