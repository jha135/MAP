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
    load_humaneval, load_trivia_cw
)


def main(benchmark_name: str, limit: int):
    print(f"Starting MRP (Meta-Reasoning Prompting) Experiment on Benchmark: '{benchmark_name}'")

    # 1. Initialize Handlers and Load MRP-specific Prompt
    llm_handler = LLMHandler()
    try:
        prompt_dir = Path(__file__).resolve().parent.parent.parent / "data" / "prompts" / "execution_methods"
        meta_prompt_template = (prompt_dir / "mrp.md").read_text(encoding='utf-8')
        print("MRP-specific prompt template loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find the MRP prompt file. {e}")
        return

    # 2. Load Benchmark Data
    benchmark_name_lower = benchmark_name.lower()
    
    # [FIXED] Removed lambda and split arguments to match the correct loading method.
    loader_map = {
        'gsm8k': load_gsm8k,
        'drop': load_drop,
        'hotpotqa': load_hotpotqa,
        'game_of_24': load_game_of_24,
        'humaneval': load_humaneval,
        'trivia_cw': load_trivia_cw
    }
    
    loader = loader_map.get(benchmark_name_lower)
    if not loader:
        raise ValueError(f"Unknown or unsupported benchmark: {benchmark_name}")
    
    # Load problems by calling the loader function without arguments.
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

            selected_strategy = "chain_of_thought" 
            mrp_log = {"raw_output": selection_output_str}
            mrp_log = {"raw_output": selection_output_str}
            selected_strategy = "chain_of_thought"  # 기본 전략으로 우선 설정

            try:
                # LLM의 응답 문자열에서 JSON 부분만 찾기 (가장 큰 JSON 블록을 찾음)
                json_match = re.search(r'\{.*\}', selection_output_str, re.DOTALL)
                if not json_match:
                    raise json.JSONDecodeError("No JSON object found in the output.", selection_output_str, 0)
                
                # 찾은 JSON 문자열을 파이썬 딕셔너리로 변환
                parsed_json = json.loads(json_match.group(0))
                
                # 'selected_strategy' 키를 이용해 값 추출
                selected_strategy = parsed_json['selected_strategy']
                
                # 로그에는 파싱된 전체 JSON 데이터를 저장하여 더 풍부한 정보 기록
                mrp_log['parsed_data'] = parsed_json
                mrp_log['selected_strategy'] = selected_strategy # 명시적으로도 저장

            except (json.JSONDecodeError, KeyError) as e:
                # JSON 파싱에 실패하거나, 필요한 키가 없는 경우
                mrp_log["error"] = f"Failed to parse JSON or find key: {e}"

            # === MRP Phase 2: Execution ===
            generated_answer, execution_tokens, *_ = run_strategy(llm_handler, selected_strategy, question, context)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # [MODIFIED] Standardized the save path to match other scripts.
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "baseline" / "mrp"
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
        choices=['gsm8k', 'drop', 'hotpotqa', 'game_of_24', 'humaneval', 'trivia_cw'],
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