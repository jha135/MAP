import sys
import csv
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.map.llm_handler import LLMHandler
from src.map.strategy_executor import run_strategy 
from src.utils.data_loader import (
    load_gsm8k,
    load_drop,
    load_game_of_24,
    load_hotpotqa,
    load_humaneval,
    load_trivia_cw
)

def _evaluate_gsm8k(generated_answer: str, correct_answer: str) -> str:
    """GSM8K 답변을 최종 숫자 비교를 통해 평가합니다."""
    try:
        correct_num = re.findall(r'####\s*(-?\d+\.?\d*)', correct_answer)[-1]
        generated_num = re.findall(r'(-?\d+\.?\d*)', generated_answer)[-1]
        return "success" if float(correct_num) == float(generated_num) else "failure"
    except (IndexError, ValueError):
        return "failure"

def evaluate_answer(benchmark_name: str, question: str, generated_answer: str, correct_answer: str, llm_judge: LLMHandler) -> tuple[str, dict]:
    """
    벤치마크에 맞는 평가 방식을 선택하고, 결과와 토큰 사용량을 함께 반환합니다.
    """
    if benchmark_name.lower() == 'gsm8k':
        return _evaluate_gsm8k(generated_answer, correct_answer), {}
    else:
        judge_prompt = f"""
        Is the following generated answer correct for the question?
        Question: "{question}"
        Correct Answer: "{correct_answer}"
        Generated Answer: "{generated_answer}"
        Respond only with "success" or "failure".
        """
        response, tokens = llm_judge.invoke(judge_prompt)
        outcome = "success" if "success" in response.lower() else "failure"
        return outcome, tokens


def main(benchmark_name: str, limit: int):
    print(f"벤치마크 '{benchmark_name}'에 대한 데이터 수집을 시작합니다.")

    # 1. 유틸리티 및 프롬프트 초기화
    llm = LLMHandler()
    prompt_dir = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"
    scoring_prompt_template = (prompt_dir / "1_scoring_prompt.md").read_text(encoding='utf-8')
    selection_prompt_template = (prompt_dir / "2_selection_prompt.md").read_text(encoding='utf-8')

    # 2. 검증용 데이터 로드
    print(f"검증용 데이터셋 로딩 중: {benchmark_name}")
    benchmark_name_lower = benchmark_name.lower()
    
    loader_map = {
        'gsm8k': lambda: load_gsm8k(),
        'drop': lambda: load_drop(),
        'hotpotqa': lambda: load_hotpotqa(),
        'game_of_24': lambda: load_game_of_24(),
        'humaneval': lambda: load_humaneval(),
        'trivia_cw': lambda: load_trivia_cw()
    }
    loader = loader_map.get(benchmark_name_lower)
    if not loader:
        raise ValueError(f"지원하지 않는 벤치마크: '{benchmark_name}'")
    problems = loader()
    
    if not problems:
        print("검증용 데이터를 찾을 수 없습니다. 중단합니다.")
        return
        
    if limit > 0 and len(problems) > limit:
        print(f"문제 수를 {len(problems)}개에서 처음 {limit}개로 제한합니다.")
        problems = problems[:limit]

    # 3. 데이터 수집 루프
    results_data = []
    print(f"{len(problems)}개의 문제에서 데이터 수집 중...")
    for problem in tqdm(problems, desc=f"Collecting data for {benchmark_name}"):
        question = problem['question']
        correct_answer = problem.get('answer', 'N/A')
        context = problem.get('context')
        input_query = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question
        
        total_tokens = {}

        try:
            # 1차 호출: 점수 생성
            scoring_prompt = scoring_prompt_template.replace("{{user_query}}", input_query)
            scores_output_str, scoring_tokens = llm.invoke(scoring_prompt)
            total_tokens.update(scoring_tokens)
            scores_json_str = re.search(r'```json\n(.*?)\n```', scores_output_str, re.DOTALL).group(1) if '```json' in scores_output_str else scores_output_str
            scores_data = json.loads(scores_json_str)

            # 2차 호출: 선택 및 확신도 생성
            selection_prompt = selection_prompt_template.replace("{{strategy_scores_json}}", json.dumps(scores_data))
            selection_output_str, selection_tokens = llm.invoke(selection_prompt)
            total_tokens.update(selection_tokens)
            selection_json_str = re.search(r'```json\n(.*?)\n```', selection_output_str, re.DOTALL).group(1) if '```json' in selection_output_str else selection_output_str
            selection_data = json.loads(selection_json_str)
            
            stage1_data = {**scores_data, **selection_data}

            # stage1_data 생성 직후에 점수 계산 수행
            scores = stage1_data.get("strategy_scores", {})
            score_values = sorted([float(v) for v in scores.values() if str(v).replace('.','',1).isdigit()], reverse=True)
            max_score = score_values[0] if score_values else 0
            score_gap = (max_score - score_values[1]) if len(score_values) > 1 else 0
            
            # 변수 초기화
            selected_strategy = stage1_data.get("selected_strategy", "N/A")
            final_outcome = "failure"

            if stage1_data.get("status") != "REQUEST_SYNTHESIS":
                # 3, 4차 호출
                generated_answer, exec_tokens = run_strategy(llm, selected_strategy, question, context)
                total_tokens.update(exec_tokens)
                final_outcome, judge_tokens = evaluate_answer(benchmark_name, question, generated_answer, correct_answer, llm)
                total_tokens.update(judge_tokens)

            results_data.append({
                "question": question, 
                "max_score": max_score,
                "score_gap": score_gap,
                "confidence_score": stage1_data.get("confidence_score"),
                "selected_strategy": selected_strategy, 
                "final_outcome": final_outcome,
                "total_tokens": json.dumps(total_tokens)
            })

        except (json.JSONDecodeError, AttributeError, IndexError, KeyError, ValueError) as e:
            print(f"\n파싱 오류 또는 실행 오류로 인해 문제를 건너뜁니다: {e}")
            continue

    # 4. 결과를 CSV 파일로 저장
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "outputs" / "validations"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"validation_results_{benchmark_name}.csv"
    print(f"\n수집된 데이터를 다음 파일에 저장합니다: {output_file}...")
    
    fieldnames = ["question", "max_score", "score_gap", "confidence_score", "selected_strategy", "final_outcome", "total_tokens"]
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print("데이터 수집이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="규칙 수립을 위한 데이터를 수집합니다.")
    parser.add_argument("--benchmark", type=str, required=True, 
                        choices=['gsm8k', 'drop', 'hotpotqa', 'game_of_24', 'humaneval', 'trivia_cw'],
                        help="사용할 벤치마크 이름.")
    parser.add_argument("--limit", type=int, default=0, help="실행할 문제 수를 제한합니다. 기본값은 0 (전체 실행).")
    
    args = parser.parse_args()
    main(args.benchmark, args.limit)