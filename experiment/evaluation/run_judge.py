import sys
import csv
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.map.llm_handler import LLMHandler

def parse_judge_response(response_str: str, is_map_model: bool) -> dict:
    try:
        json_str = re.search(r'```json\n(.*?)\n```', response_str, re.DOTALL).group(1) if '```json' in response_str else response_str
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError, IndexError):
        base_failure = {"task_success": {"is_correct": False, "is_catastrophic_failure": True, "reasoning": "Failed to parse judge response."}}
        if is_map_model:
            base_failure["strategy_quality"] = {}
            base_failure["decision_rationality"] = {}
        return base_failure

def main(input_file_path: Path, judge_models: list):
    print(f"\n'{input_file_path.name}' 파일에 대한 다중 심판 채점을 시작합니다.")
    print(f"사용될 심판 모델: {judge_models}")
    try:
        model_name = input_file_path.stem.split('_')[1].upper()
    except IndexError:
        print(f"오류: '{input_file_path.name}' 파일명에서 모델 이름을 식별할 수 없습니다.")
        return
    
    is_map_model = (model_name == 'MAP')
    rubric_filename = "rubric.md" if is_map_model else "baseline_rubric.md"
    
    try:
        rubric_path = Path(__file__).parent / "rubrics" / rubric_filename
        rubric_template = rubric_path.read_text(encoding='utf-8')
        print(f"'{model_name}' 모델을 위한 '{rubric_filename}' 기준표를 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: '{rubric_path}'에서 평가 기준표 파일을 찾을 수 없습니다.")
        return

    # 여러 심판 모델 초기화
    llm_judges = {name: LLMHandler(model_name=name) for name in judge_models}

    # 2. 실험 결과 파일 읽기
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"오류: '{input_file_path}'에서 입력 파일을 찾을 수 없습니다.")
        return

    evaluated_results = []
    
    # 3. 각 결과에 대해 평가 수행
    print(f"{len(rows)}개의 결과를 {len(llm_judges)}명의 심판으로 평가합니다...")
    for row in tqdm(rows):
        # 평가에 필요한 공통 프롬프트 부분 생성
        if is_map_model:
            base_prompt = rubric_template.format(
                question=row.get('question', ''),
                correct_answer=row.get('correct_answer', ''),
                generated_answer=row.get('generated_answer', ''),
                execution_log=row.get('execution_log', '{}')
            )
        else:
            base_prompt = rubric_template.format(
                question=row.get('question', ''),
                correct_answer=row.get('correct_answer', ''),
                generated_answer=row.get('generated_answer', '')
            )
        
        # 각 심판 모델로 평가 실행
        for judge_name, llm_judge in llm_judges.items():
            response_str = llm_judge.invoke(base_prompt)
            eval_data = parse_judge_response(response_str, is_map_model)
            # 각 심판의 평가 결과를 별도의 열에 저장
            row[f'evaluation_{judge_name}'] = json.dumps(eval_data)
        
        evaluated_results.append(row)

    # 4. 채점 결과가 추가된 새 파일 저장
    output_dir = input_file_path.parent
    output_file = output_dir / f"evaluated_{input_file_path.name}"
    print(f"\n상세 평가 결과를 '{output_file.name}' 파일에 저장했습니다.")
    
    fieldnames = list(evaluated_results[0].keys()) if evaluated_results else []
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluated_results)
        
    print("채점이 성공적으로 완료되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge를 사용하여 실험 결과 파일을 평가합니다.")
    parser.add_argument("input_file", type=str, help="평가할 단일 실험 결과 CSV 파일의 경로.")
    parser.add_argument("--judges", nargs='+', default=['gpt-5'], help="평가에 사용할 심판 LLM 모델 이름 목록 (예: gpt-4o, gpt-5)")
    args = parser.parse_args()
    
    main(Path(args.input_file), args.judges)