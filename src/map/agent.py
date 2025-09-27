import json
import re
from pathlib import Path
from typing import Dict, Any

from .llm_handler import LLMHandler
from .strategy_executor import run_strategy

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = BASE_DIR / "data" / "prompts"

class MapAgent:
    def __init__(self):
        print("Initializing MapAgent...")
        self.llm_handler = LLMHandler()
        
        try:
            # 2단계 프롬프트 체이닝을 위해 2개의 프롬프트를 불러옵니다.
            self.scoring_prompt_template = (PROMPT_DIR / "1_scoring_prompt.md").read_text(encoding='utf-8')
            self.selection_prompt_template = (PROMPT_DIR / "2_selection_prompt.md").read_text(encoding='utf-8')
            
            # 경로 B와 C를 위한 프롬프트는 그대로 유지합니다.
            self.self_correction_prompt_template = (PROMPT_DIR / "b_path_self_correction.md").read_text(encoding='utf-8')
            self.synthesis_unified_prompt_template = (PROMPT_DIR / "c_path_synthesis.md").read_text(encoding='utf-8')
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

            # 두 결과를 합쳐 최종 stage1_data를 구성합니다.
            stage1_data = {**scores_data, **selection_data}

        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            raw_output = f"Scoring Output: {scores_output_str}\nSelection Output: {selection_output_str}"
            return {"final_answer": "Error: Could not decode Stage 1.", "execution_log": {"error": str(e), "raw_output": raw_output}}

        # stage1_data 생성 직후에 점수 계산 수행
        scores = stage1_data.get("strategy_scores", {})
        score_values = sorted([float(v) for v in scores.values() if str(v).replace('.','',1).isdigit()], reverse=True)
        max_score = score_values[0] if score_values else 0
        confidence_score = float(stage1_data.get("confidence_score", 0.0))

        path = ''
        # 데이터 기반으로 확정된 최종 규칙을 여기에 적용해야 합니다.
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
            execution_log['path_C_log'] = {"method": "Unified Synthesis"}

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
                # mitigation_plan이 null이거나 비어있는 경우를 대비한 fallback
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