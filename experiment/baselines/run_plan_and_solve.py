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

def load_preprocessed_benchmark(benchmark_name: str, split: str = "test") -> list:

    # 데이터 파일 경로를 동적으로 구성
    file_path = Path(__file__).resolve().parent.parent.parent / "data" / "benchmarks" / benchmark_name / f"{split}.json"
    print(f"Loading pre-processed benchmark data from: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"표준화된 벤치마크 파일({file_path})을 찾을 수 없습니다. 모든 벤치마크를 QA JSON 형식으로 사전 처리했는지 확인하세요.")

    # 표준 JSON 파일을 읽어 파싱
    with open(file_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    return problems


def main(benchmark_name: str, limit: int):
    strategy_to_run = "plan_and_solve"
    print(f"Starting Baseline Experiment: Strategy='{strategy_to_run}', Benchmark='{benchmark_name}'")

    try:
        problems = load_preprocessed_benchmark(benchmark_name.lower())
    except FileNotFoundError as e:
        print(f"오류: {e}")
        return

    if not problems:
        print("No problems loaded. Aborting experiment.")
        return

    # 문제 수 제한
    if limit > 0 and len(problems) > limit:
        print(f"Limiting benchmark from {len(problems)} to the first {limit} problems.")
        problems = problems[:limit]

    # 2. LLM 핸들러 초기화
    llm_handler = LLMHandler()

    # 3. 벤치마크 문제 순회 및 결과 기록
    print(f"Running '{strategy_to_run}' strategy on {len(problems)} problems...")
    results = []
    for problem in tqdm(problems, desc=f"Running {strategy_to_run}"):
        try:
            question = problem['question']
            context = problem.get('context')
            correct_answer = problem.get('answer', 'N/A')
            
            generated_answer = run_strategy(llm_handler, strategy_to_run, question, context)
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
            })
        except Exception as e:
            print(f"\n문제 처리 중 오류 발생: {e}. 다음 문제로 넘어갑니다.")
            results.append({
                "question": problem.get('question', 'N/A'),
                "correct_answer": problem.get('answer', 'N/A'),
                "generated_answer": f"EXECUTION_ERROR: {e}",
            })
            continue

    # 4. 결과 파일로 저장
    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"results_{strategy_to_run.replace(' ', '_')}_{benchmark_name}_{timestamp}.csv"
    file_path = results_dir / file_name

    try:
        fieldnames = ["question", "correct_answer", "generated_answer"]
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
        help="The benchmark folder name to use (e.g., 'gsm8k')."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0,
        help="Limit the number of problems to run. Default is 0 (run all)."
    )
    
    args = parser.parse_args()
    main(args.benchmark, args.limit)

