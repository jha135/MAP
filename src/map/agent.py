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
            self.stage1_prompt_template = (PROMPT_DIR / "metacognitive_evaluation.md").read_text(encoding='utf-8')
            self.self_correction_prompt_template = (PROMPT_DIR / "b_path_self_correction.md").read_text(encoding='utf-8')
            self.synthesis_unified_prompt_template = (PROMPT_DIR / "c_path_synthesis.md").read_text(encoding='utf-8')
            print("Prompt templates loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: A required prompt file was not found. {e}")
            raise

    

    def run(self, question: str, context: str = None) -> Dict[str, Any]:
        total_tokens = {} # 토큰 사용량 기록 시작

        input_query = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question
        stage1_prompt = self.stage1_prompt_template.replace("{{user_query}}", input_query)
        stage1_output_str, tokens = self.llm_handler.invoke(stage1_prompt)
        total_tokens.update(tokens) # 토큰 사용량 누적

        try:
            json_str = re.search(r'```json\n(.*?)\n```', stage1_output_str, re.DOTALL).group(1) if '```json' in stage1_output_str else stage1_output_str
            stage1_data = json.loads(json_str)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            return {"final_answer": "Error: Could not decode Stage 1.", "metacognitive_data": None, "execution_log": {"error": str(e), "raw_output": stage1_output_str}}

        scores = stage1_data.get("strategy_scores", {})
        score_values = sorted([v for v in scores.values() if isinstance(v, (int, float))], reverse=True)
        max_score = score_values[0] if score_values else 0
        score_gap = (max_score - score_values[1]) if len(score_values) > 1 else max_score

        path = ''
        if (max_score >= 8 and score_gap >= 3) or stage1_data.get("confidence_score", 0) >= 0.95:
            path = 'A'
        elif max_score <= 4 or stage1_data.get("status") == "REQUEST_SYNTHESIS":
            path = 'C'
        else:
            path = 'B'

        final_answer = ""
        execution_log = {"path_chosen": path, "stage1_data": stage1_data}

        if path == 'A':
            print("Path A: Confident Execution")
            selected_strategy = stage1_data.get("selected_strategy")
            final_answer, tokens = run_strategy(self.llm_handler, selected_strategy, question, context) # 수정된 호출
            total_tokens['completion_tokens'] = total_tokens.get('completion_tokens', 0) + tokens.get('completion_tokens', 0)
            total_tokens['prompt_tokens'] = total_tokens.get('prompt_tokens', 0) + tokens.get('prompt_tokens', 0)
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
            total_tokens.update(draft_tokens) # 초안 생성 토큰 누적

            correction_prompt = self.self_correction_prompt_template.replace("{{question}}", input_query)
            correction_prompt = correction_prompt.replace("{{draft_answer}}", draft_answer)
            check_result_str, check_tokens = self.llm_handler.invoke(correction_prompt)
            total_tokens.update(check_tokens) # 교정 단계 토큰 누적

            
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
                final_answer, mitigation_tokens = run_strategy(self.llm_handler, mitigation_plan, question, context)
                total_tokens.update(mitigation_tokens)

            execution_log['path_B_log'] = {
                "initial_strategy": selected_strategy, 
                "draft_answer": draft_answer,
                "correction_check_result": check_result_str, 
                "mitigation_used": not checks_passed
            }

        return {
            "final_answer": final_answer,
            "execution_log": execution_log,
            "total_tokens": total_tokens # [수정] total_tokens 반환 추가
        }

