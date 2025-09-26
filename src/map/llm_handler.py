import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.outputs import Generation

class LLMHandler:

    def __init__(self, model_name: str = "gpt-5", temperature: float = 0.0):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        self.client = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        print(f"LLMHandler initialized with model: {model_name}")

    def invoke(self, prompt: str) -> tuple[str, dict]:
        print(prompt)
        messages = [HumanMessage(content=prompt)]

        try:
            result: LLMResult = self.client.generate([messages])
            generation: Generation = result.generations[0][0]
            token_usage = result.llm_output.get("token_usage", {}) if result.llm_output else {}

            return generation.text, token_usage
        except Exception as e:
            return f"Error: {e}", {}