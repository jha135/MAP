import sys
import csv
import json
import re
from pathlib import Path
from tqdm import tqdm # 진행 상황을 보여주는 라이브러리

# src 폴더가 파이썬 경로에 포함되도록 설정
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.map_agent.llm_handler import LLMHandler
from src.utils.data_loader import load_gsm8k

def evaluate_gsm8k(generated_answer: str, correct_answer: str) -> str:
    """GSM8K 정답을 간단하게 채점하는 함수. 최종 숫자만 비교합니다."""
    try:
        # 정답에서 숫자 추출 (e.g., "#### 18" -> "18")
        correct_num = re.findall(r'####\s*(-?\d+\.?\d*)', correct_answer)[-1]
        # 생성된 답변에서 숫자 추출
        generated_num = re.findall(r'(-?\d+\.?\d+)', generated_answer)[-1]

        if float(correct_num) == float(generated_num):
            return "success"
        else:
            return "failure"
    except (IndexError, ValueError):
        # 숫자 추출에 실패하면 일단 실패로 처리
        return "failure"

def main():
   
    # 1. 유틸리티 및 프롬프트 초기화
    llm = LLMHandler()
    prompt_dir = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"
    stage1_prompt_template = (prompt_dir / "metacognitive_evaluation.md").read_text(encoding='utf-8')

    # 2. 검증용 데이터 로드 
    print("Loading validation dataset...")
    validation_problems = load_gsm8k(split="train")
    if not validation_problems:
        print("Validation data not found. Aborting.")
        return

    # 3. 데이터 수집 루프
    results_data = []
    print(f"Collecting data from {len(validation_problems)} problems...")
    for problem in tqdm(validation_problems):
        question = problem['question']
        correct_answer = problem['answer']

        # --- MAP 1단계 실행하여 메타인지 신호 추출 ---
        stage1_prompt = stage1_prompt_template.replace("{{user_query}}", question)
        stage1_output_str = llm.invoke(stage1_prompt)

        try:
            if '```json' in stage1_output_str:
                json_str = stage1_output_str.split('```json\n')[1].split('\n```')[0]
            else:
                json_str = stage1_output_str
            stage1_data = json.loads(json_str)

            # status가 REQUEST_SYNTHESIS인 경우는 건너뛰거나 별도 처리 (여기선 실패로 간주)
            if stage1_data.get("status") == "REQUEST_SYNTHESIS":
                final_outcome = "failure"
                selected_strategy = "N/A"
            else:
                # --- 실제 성공 여부(Ground Truth) 확인 ---
                selected_strategy = stage1_data.get("selected_strategy", "Unknown")
                execution_prompt = f"Please solve the following math problem using the '{selected_strategy}' method:\n\nProblem: {question}"
                generated_answer = llm.invoke(execution_prompt)
                final_outcome = evaluate_gsm8k(generated_answer, correct_answer)

            # --- 결과 기록 ---
            scores = stage1_data.get("strategy_scores", {})
            score_values = [v for v in scores.values() if isinstance(v, (int, float))]
            score_values.sort(reverse=True)

            max_score = score_values[0] if len(score_values) > 0 else 0
            score_gap = (max_score - score_values[1]) if len(score_values) > 1 else 0

            results_data.append({
                "question": question,
                "max_score": max_score,
                "score_gap": score_gap,
                "confidence_score": stage1_data.get("confidence_score"),
                "selected_strategy": selected_strategy,
                "final_outcome": final_outcome
            })

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"\nSkipping a problem due to parsing error: {e}")
            continue

    # 4. CSV 파일로 저장
    output_dir = Path(__file__).parent
    output_file = output_dir / "validation_results.csv"
    print(f"\nSaving collected data to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "max_score", "score_gap", "confidence_score", "selected_strategy", "final_outcome"])
        writer.writeheader()
        writer.writerows(results_data)

    print("Data collection finished successfully.")

if __name__ == "__main__":
    main()