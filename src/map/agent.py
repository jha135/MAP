import json
from pathlib import Path
from .llm_handler import LLMHandler



BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = BASE_DIR / "data" / "prompts"

class MapAgent:
    def __init__(self):
        print("Initializing MapAgent...")
        self.llm_handler = LLMHandler()
        
        # 프롬프트 템플릿을 파일에서 한 번만 읽어옵니다.
        try:
            self.stage1_prompt_template = (PROMPT_DIR / "metacognitive_evaluation.md").read_text(encoding='utf-8')
            self.stage2_prompt_template = (PROMPT_DIR / "execution_engine.md").read_text(encoding='utf-8')
            print("Prompt templates loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: Prompt file not found. {e}")
            raise
            

    def run(self, question: str) -> str:
        
        # --- 1단계: 메타인지 평가 ---
        print("\nPerforming Metacognitive Evaluation")
        stage1_prompt = self.stage1_prompt_template.replace("{{user_query}}", question)
        
        stage1_output_str = self.llm_handler.invoke(stage1_prompt)
        print("Evaluation received from LLM.")
        try:
            if '```json' in stage1_output_str:
                stage1_output_json_str = stage1_output_str.split('```json\n')[1].split('\n```')[0]
            else:
                stage1_output_json_str = stage1_output_str
            
            json.loads(stage1_output_json_str) 
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error: Failed to parse Stage 1 output as JSON. Error: {e}")
            return "Error: Could not decode the metacognitive evaluation from the LLM."

        print("\nFinal Response Generation")
        stage2_prompt = self.stage2_prompt_template.replace("{{user_query}}", question)
        stage2_prompt = stage2_prompt.replace("{{evaluation_json}}", stage1_output_json_str)

        final_result = self.llm_handler.invoke(stage2_prompt)
        print("Final result generated.")
        
        return final_result