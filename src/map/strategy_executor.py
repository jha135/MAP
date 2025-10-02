from pathlib import Path
from .llm_handler import LLMHandler
PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"

def run_strategy(llm_handler: LLMHandler, strategy_name: str, question: str, context: str = None): # 반환 타입 힌트도 수정
    if not strategy_name:
        # ✅ (문자열, 빈 딕셔너리) 형태로 반환
        return "Error: Mitigation plan strategy name is missing in the Stage 1 JSON.", {}
        
    try:
        recipe_file_name = f"{strategy_name.lower().replace(' ', '_')}.md"
        recipe_path = PROMPT_DIR / "execution_methods" / recipe_file_name
        strategy_instructions = recipe_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        # ✅ (문자열, 빈 딕셔너리) 형태로 반환
        return f"Error: Recipe file for strategy '{strategy_name}' not found.", {}

    input_query = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question
    
    final_prompt = f"""
You must solve the following problem using the '{strategy_name}' strategy.

--- STRATEGY INSTRUCTIONS ---
{strategy_instructions}
-----------------------------

--- PROBLEM ---
{input_query}
---------------

Now, solve the problem following the instructions precisely.
"""
    return llm_handler.invoke(final_prompt)