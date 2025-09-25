import sys
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm # tqdm 임포트 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.map.agent import MapAgent
from src.utils.data_loader import (
    load_gsm8k,
    load_drop,
    load_game_of_24,
    load_hotpotqa,
    load_mbpp,
    load_trivia_cw
)

def main(benchmark_name: str, limit: int):
    print(f"Starting MAP Agent Experiment on Benchmark: '{benchmark_name}'")

    # 1. 벤치마크 데이터 로드
    print(f"Loading benchmark data: {benchmark_name}")
    benchmark_name_lower = benchmark_name.lower()
    
    loader_map = {
        'gsm8k': load_gsm8k,
        'drop': load_drop,
        'game_of_24': load_game_of_24,
        'hotpotqa': load_hotpotqa,
        'mbpp': load_mbpp,
        'trivia_cw': load_trivia_cw,
    }
    
    if benchmark_name_lower in loader_map:
        problems = loader_map[benchmark_name_lower](split="test")
    else:
        raise ValueError(f"Unknown or unsupported benchmark: {benchmark_name}")

    if not problems:
        print("No problems loaded. Aborting experiment.")
        return

    if limit > 0 and len(problems) > limit:
        print(f"Limiting benchmark from {len(problems)} to the first {limit} problems.")
        problems = problems[:limit]

    # 2. MAP 에이전트 초기화
    print("Initializing MAP Agent...")
    agent = MapAgent()

    # 3. 벤치마크 문제 순회 및 결과 기록
    print(f"Running MAP Agent on {len(problems)} problems...")
    results = []
    # [수정] tqdm을 사용하여 진행 상황 표시
    for problem in tqdm(problems, desc=f"Benchmarking {benchmark_name}"):
        
        try:
            question = problem['question']
            context = problem.get('context')
            correct_answer = problem.get('answer', 'N/A')
            response_dict = agent.run(question, context=context)
            
            generated_answer = response_dict.get("final_answer", "ERROR: No answer generated.")
            execution_log = response_dict.get("execution_log", {})
            total_tokens = response_dict.get("total_tokens", 0) # [추가] 토큰 사용량 추출
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "execution_log": json.dumps(execution_log),
                "total_tokens": total_tokens # [추가] 토큰 사용량 저장
            })
        except Exception as e:
            print(f"\nError processing a problem: {e}. Skipping to the next one.")
            # 실패한 경우에도 로그를 남길 수 있도록 결과 추가
            results.append({
                "question": problem.get('question', 'N/A'),
                "correct_answer": problem.get('answer', 'N/A'),
                "generated_answer": f"EXECUTION_ERROR: {e}",
                "execution_log": "{}",
                "total_tokens": 0
            })
            continue

    # 4. 결과 파일로 저장
    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent / "results" / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"results_MAP_{benchmark_name}_{timestamp}.csv"
    file_path = results_dir / file_name

    try:
        fieldnames = ["question", "correct_answer", "generated_answer", "execution_log", "total_tokens"]
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Experiment finished. Results saved to '{file_path}'")
    except Exception as e:
        print(f"Failed to save results. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MAP Agent experiment.")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        required=True, 
        choices=['gsm8k', 'drop', 'hotpotqa', 'human_eval', 'game_of_24','trivia_cw'],
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