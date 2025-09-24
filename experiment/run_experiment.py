import sys
import csv
import argparse
from datetime import datetime
from pathlib import Path

# src 폴더가 파이썬 경로에 포함되도록 설정
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.map_agent.agent import MapAgent
from src.utils.data_loader import load_gsm8k

def main(benchmark_name: str):
    """
    MAP 에이전트를 지정된 벤치마크로 실행하고 결과를 저장합니다.
    """
    print(f"🚀 Starting MAP Agent Experiment on Benchmark: '{benchmark_name}'")

    # 1. 벤치마크 데이터 로드
    print(f"🔄 Loading benchmark data: {benchmark_name}")
    if benchmark_name.lower() == 'gsm8k':
        problems = load_gsm8k(split="test")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    if not problems:
        print("❌ No problems loaded. Aborting experiment.")
        return

    # 2. MAP 에이전트 초기화
    print("🧠 Initializing MAP Agent...")
    agent = MapAgent()

    # 3. 벤치마크 문제 순회 및 결과 기록
    print(f"⚙️ Running MAP Agent on {len(problems)} problems...")
    results = []
    for i, problem in enumerate(problems):
        print(f"  - Processing problem {i+1}/{len(problems)}...")
        
        question = problem['question']
        correct_answer = problem['answer']
        
        generated_answer = agent.run(question)
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "generated_answer": generated_answer,
        })

    # 4. 결과 파일로 저장
    print("💾 Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent / "results" / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"results_MAP_{benchmark_name}_{timestamp}.csv"
    file_path = results_dir / file_name

    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "correct_answer", "generated_answer"])
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Experiment finished. Results saved to '{file_path}'")
    except Exception as e:
        print(f"❌ Failed to save results. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MAP Agent experiment.")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        required=True, 
        choices=['gsm8k'],
        help="The benchmark to use (e.g., 'gsm8k')."
    )
    
    args = parser.parse_args()
    main(args.benchmark)