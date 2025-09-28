import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini 라이브러리 추가
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult, Generation

class LLMHandler:
    def __init__(self, model_name: str = "gpt-5", temperature: float = 0.0):
        self.provider = self._get_provider(model_name)
        
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
            self.client = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )
        elif self.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
                
            self.client = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key
            )
        else:
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}. 'gpt' 또는 'gemini'로 시작해야 합니다.")
            
        print(f"LLMHandler initialized with model: {model_name} (Provider: {self.provider})")

    def _get_provider(self, model_name: str) -> str:
        """모델 이름에 따라 'openai' 또는 'google'을 반환합니다."""
        if "gemini" in model_name.lower():
            return "google"
        elif "gpt" in model_name.lower():
            return "openai"
        return "unknown"

    def invoke(self, prompt: str) -> tuple[str, dict]:
        print(prompt) # 필요시 주석 해제하여 사용
        messages = [HumanMessage(content=prompt)]

        try:
            result: LLMResult = self.client.generate([messages])
            generation: Generation = result.generations[0][0]
            
            # --- Provider별 토큰 사용량 파싱 로직 ---
            token_usage = {}
            if self.provider == "openai" and result.llm_output:
                token_usage = result.llm_output.get("token_usage", {})
            elif self.provider == "google" and result.llm_output:
                usage_metadata = result.llm_output.get("usage_metadata", {})
                token_usage = {
                    "prompt_tokens": usage_metadata.get("prompt_token_count", 0),
                    "completion_tokens": usage_metadata.get("candidates_token_count", 0),
                    "total_tokens": usage_metadata.get("total_token_count", 0),
                }
            # ----------------------------------------

            return generation.text, token_usage
        except Exception as e:
            return f"Error: {e}", {}