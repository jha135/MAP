import sys
import csv
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.map.llm_handler import LLMHandler
    from src.map.strategy_executor import run_strategy
    from src.utils.data_loader import (
        load_gsm8k, load_drop, load_game_of_24,
        load_hotpotqa, load_humaneval, load_trivia_cw
    )
except ImportError as e:
    print(f"Error: 필요한 모듈을 찾을 수 없습니다. 경로를 확인해주세요. {e}")
    sys.exit(1)


# --- 전역 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = BASE_DIR / "data" / "prompts"


# --- MapAgentDsp 클래스 정의 ---
class MapAgentDsp:
    def __init__(self):
        print("Initializing MapAgentDsp...")
        self.llm_handler = LLMHandler()
        
        try:
            self.scoring_prompt_template = (PROMPT_DIR / "1_scoring_prompt.md").read_text(encoding='utf-8')
            self.selection_prompt_template = (PROMPT_DIR / "2_selection_prompt.md").read_text(encoding='utf-8')
            self.self_correction_prompt_template = (PROMPT_DIR / "b_path_self_correction.md").read_text(encoding='utf-8')
            self.synthesis_unified_prompt_template = (PROMPT_DIR / "execution_method" /"dsp.md").read_text(encoding='utf-8')
            print("Prompt templates loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: A required prompt file was not found. {e}")
            raise

    def run(self, question: str, context: str = None) -> Dict[str, Any]:
        total_tokens = {}
        input_query = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question
        scores_output_str = ""
        selection_output_str = ""

        try:
            # === 1단계: 점수 평가 (Scoring) ===
            scoring_prompt = self.scoring_prompt_template.replace("{{user_query}}", input_query)
            scores_output_str, scoring_tokens = self.llm_handler.invoke(scoring_prompt)
            total_tokens.update(scoring_tokens)
            scores_json_str = re.search(r'```json\n(.*?)\n```', scores_output_str, re.DOTALL).group(1) if '```json' in scores_output_str else scores_output_str
            scores_data = json.loads(scores_json_str)

            # === 2단계: 선택 및 메타데이터 생성 (Selection) ===
            selection_prompt = self.selection_prompt_template.replace("{{strategy_scores_json}}", json.dumps(scores_data))
            selection_output_str, selection_tokens = self.llm_handler.invoke(selection_prompt)
            total_tokens.update(selection_tokens)
            selection_json_str = re.search(r'```json\n(.*?)\n```', selection_output_str, re.DOTALL).group(1) if '```json' in selection_output_str else selection_output_str
            selection_data = json.loads(selection_json_str)

            stage1_data = {**scores_data, **selection_data}

        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            raw_output = f"Scoring Output: {scores_output_str}\nSelection Output: {selection_output_str}"
            return {"final_answer": "Error: Could not decode Stage 1.", "execution_log": {"error": str(e), "raw_output": raw_output}}

        scores = stage1_data.get("strategy_scores", {})
        score_values = sorted([float(v) for v in scores.values() if str(v).replace('.','',1).isdigit()], reverse=True)
        max_score = score_values[0] if score_values else 0
        confidence_score = float(stage1_data.get("confidence_score", 0.0))

        path = ''
        if (max_score >= 7 and confidence_score >= 0.9):
            path = 'A'
        elif max_score <= 4 or confidence_score <= 0.77 or stage1_data.get("status") == "REQUEST_SYNTHESIS":
            path = 'C'
        else:
            path = 'B'

        final_answer = ""
        execution_log = {"path_chosen": path, "stage1_data": stage1_data}

        if path == 'A':
            print("Path A: Confident Execution")
            selected_strategy = stage1_data.get("selected_strategy")
            final_answer, path_tokens = run_strategy(self.llm_handler, selected_strategy, question, context)
            total_tokens.update(path_tokens)
            execution_log['path_A_log'] = {"strategy_used": selected_strategy}

        elif path == 'C':
            print("Path C: Metacognitive Synthesis")
            unified_prompt = self.synthesis_unified_prompt_template.replace("{{user_query}}", input_query)
            final_answer, path_tokens = self.llm_handler.invoke(unified_prompt)
            total_tokens.update(path_tokens)
            execution_log['path_    log'] = {"method": "Unified Synthesis"}

        else: # Path B
            print("Path B: Guarded Execution with Self-Correction")
            selected_strategy = stage1_data.get("selected_strategy")
            mitigation_plan = stage1_data.get("mitigation_plan")
            
            draft_answer, draft_tokens = run_strategy(self.llm_handler, selected_strategy, question, context)
            total_tokens.update(draft_tokens)

            correction_prompt = self.self_correction_prompt_template.replace("{{question}}", input_query)
            correction_prompt = correction_prompt.replace("{{draft_answer}}", draft_answer)
            check_result_str, check_tokens = self.llm_handler.invoke(correction_prompt)
            total_tokens.update(check_tokens)
            
            try:
                check_result_json_str = re.search(r'```json\n(.*?)\n```', check_result_str, re.DOTALL).group(1) if '```json' in check_result_str else check_result_str
                check_result_json = json.loads(check_result_json_str)
                checks_passed = check_result_json.get("checks_passed", False)
            except (json.JSONDecodeError, AttributeError, IndexError):
                checks_passed = False

            if checks_passed:
                print("Self-correction check passed. Using draft answer.")
                final_answer = draft_answer
            else:
                print("Self-correction check failed. Executing mitigation plan.")
                if not mitigation_plan:
                    print("Warning: Mitigation plan is empty. Falling back to the draft answer.")
                    final_answer = draft_answer
                else:
                    final_answer, mitigation_tokens = run_strategy(self.llm_handler, mitigation_plan, question, context)
                    total_tokens.update(mitigation_tokens)

            execution_log['path_B_log'] = {
                "initial_strategy": selected_strategy, "draft_answer": draft_answer,
                "correction_check_result": check_result_str, "mitigation_used": not checks_passed
            }

        return {
            "final_answer": final_answer,
            "execution_log": execution_log,
            "total_tokens": total_tokens
        }

# --- 실험 실행 함수 ---
def main(benchmark_name: str, limit: int):
    print(f"Starting MAP Agent Experiment on Benchmark: '{benchmark_name}'")

    print(f"Loading benchmark data: {benchmark_name}")
    benchmark_name_lower = benchmark_name.lower()
    
    loader_map = {
        'gsm8k': load_gsm8k, 'drop': load_drop, 'game_of_24': load_game_of_24,
        'hotpotqa': load_hotpotqa, 'humaneval': load_humaneval, 'trivia_cw': load_trivia_cw,
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

    print("Initializing Agent for experiment...")
    # MapAgent가 아닌 MapAgentDsp를 사용하도록 수정
    agent = MapAgentDsp()

    print(f"Running Agent on {len(problems)} problems...")
    results = []
    for problem in tqdm(problems, desc=f"Benchmarking {benchmark_name}"):
        try:
            question = problem['question']
            context = problem.get('context')
            correct_answer = problem.get('answer', 'N/A')
            response_dict = agent.run(question, context=context)
            
            generated_answer = response_dict.get("final_answer", "ERROR: No answer generated.")
            execution_log = response_dict.get("execution_log", {})
            total_tokens = response_dict.get("total_tokens", {})
            stage1_data = execution_log.get("stage1_data", {})
            confidence_score = stage1_data.get("confidence_score", None)
            
            results.append({
                "question": question, "correct_answer": correct_answer,
                "generated_answer": generated_answer, "confidence_score": confidence_score,
                "execution_log": json.dumps(execution_log), "total_tokens": json.dumps(total_tokens) 
            })
        except Exception as e:
            print(f"\nError processing a problem: {e}. Skipping to the next one.")
            results.append({
                "question": problem.get('question', 'N/A'), "correct_answer": problem.get('answer', 'N/A'),
                "generated_answer": f"EXECUTION_ERROR: {e}",
                "execution_log": "{}", "total_tokens": "{}"
            })
            continue

    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent / "results" / "outputs" / "baselines"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"results_MAP-DSP_{benchmark_name}_{timestamp}.csv"
    file_path = results_dir / file_name

    try:
        fieldnames = ["question", "correct_answer", "generated_answer", "confidence_score", "execution_log", "total_tokens"]
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Experiment finished. Results saved to '{file_path}'")
    except Exception as e:
        print(f"Failed to save results. Error: {e}")

# --- 스크립트 실행 지점 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the merged MAP Agent experiment.")
    parser.add_argument(
        "--benchmark", type=str, required=True,
        choices=['gsm8k', 'drop', 'hotpotqa', 'game_of_24', 'trivia_cw', 'humaneval'],
        help="The benchmark to use."
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit the number of problems to run. Default is 0 (run all)."
    )
    
    args = parser.parse_args()
    main(args.benchmark, args.limit)