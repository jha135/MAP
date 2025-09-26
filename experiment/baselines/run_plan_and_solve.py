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
# [수정] 중앙 데이터 로더 import
from src.utils.data_loader import (
    load_gsm8k, load_drop, load_hotpotqa, load_game_of_24,
    load_humaneval, load_trivia_cw
)


def main(benchmark_name: str, limit: int):
    # 'plan_and_solve'를 'plan-and-solve'로 수정하여 프롬프트 파일명과 일치
    strategy_to_run = "plan-and-solve"
    print(f"Starting Baseline Experiment: Strategy='{strategy_to_run}', Benchmark='{benchmark_name}'")

    # [수정] 중앙 데이터 로더를 사용하여 데이터 로드
    benchmark_name_lower = benchmark_name.lower()
    loader_map = {
        'gsm8k': lambda: load_gsm8k(split="test"),
        'drop': lambda: load_drop(split="validation"),
        'hotpotqa': lambda: load_hotpotqa(split="validation"),
        'game_of_24': lambda: load_game_of_24(split="test"),
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
            
            # [수정] 튜플 반환값 처리 및 토큰 기록
            generated_answer, tokens = run_strategy(llm_handler, strategy_to_run, question, context)
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "total_tokens": json.dumps(tokens) # 토큰 정보 추가
            })
        except Exception as e:
            print(f"\n문제 처리 중 오류 발생: {e}. 다음 문제로 넘어갑니다.")
            results.append({
                "question": problem.get('question', 'N/A'),
                "correct_answer": problem.get('answer', 'N/A'),
                "generated_answer": f"EXECUTION_ERROR: {e}",
                "total_tokens": "{}" # 토큰 정보 추가
            })
            continue

    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 이름 생성 시 strategy_to_run의 공백을 밑줄로 변경
    file_name = f"results_{strategy_to_run.replace(' ', '_')}_{benchmark_name}_{timestamp}.csv"
    file_path = results_dir / file_name

    try:
        # [수정] CSV 필드명에 total_tokens 추가
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
        # [수정] choices 추가하여 사용자 편의성 증대
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