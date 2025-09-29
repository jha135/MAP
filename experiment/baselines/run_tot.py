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
from src.utils.data_loader import (
    load_gsm8k, load_drop, load_hotpotqa, load_game_of_24,
    load_humaneval, load_trivia_cw
)


def main(benchmark_name: str, limit: int):
    strategy_to_run = "tree_of_thought"
    print(f"Starting Baseline Experiment: Strategy='{strategy_to_run}', Benchmark='{benchmark_name}'")

    benchmark_name_lower = benchmark_name.lower()
    
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
    
    # 인자 없이 로더 함수를 직접 호출
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
            
            generated_answer, tokens = run_strategy(llm_handler, strategy_to_run, question, context)
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "total_tokens": json.dumps(tokens) 
            })
        except Exception as e:
            print(f"\n문제 처리 중 오류 발생: {e}. 다음 문제로 넘어갑니다.")
            results.append({
                "question": problem.get('question', 'N/A'),
                "correct_answer": problem.get('answer', 'N/A'),
                "generated_answer": f"EXECUTION_ERROR: {e}",
                "total_tokens": "{}"
            })
            continue

    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # [경로 수정] MAP 에이전트와 동일하게 results/outputs/main 디렉토리에 저장하도록 변경
    results_dir = Path(__file__).resolve().parent.parent / "results" / "outputs" / "baseline" / "tree_of_thought"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # [파일명 수정] Baseline 이라는 것을 명확히 표시
    file_name = f"results_Baseline_{strategy_to_run.replace(' ', '_')}_{benchmark_name}_{timestamp}.csv"
    file_path = results_dir / file_name

    try:
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