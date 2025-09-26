import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "benchmarks"

def load_gsm8k(split: str = "test") -> List[Dict[str, Any]]:
    """
    gsm.jsonl 파일에서 GSM8K 데이터를 로드합니다.
    """
    # [수정] 파일 경로를 실제 파일명으로 직접 지정
    file_path = DATA_DIR / "gsm8k" / "gsm.jsonl"
    print(f"Loading GSM8K data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]
        
        print(f"Successfully loaded {len(problems)} problems from GSM8K.")
        return problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []

def load_game_of_24(split: str = "test") -> List[Dict[str, Any]]:
    """
    game_of_24.jsonl 파일에서 Game of 24 데이터를 로드합니다.
    """
    # [수정] 파일 경로를 실제 파일명으로 직접 지정
    file_path = DATA_DIR / "game_of_24" / "game_of_24.jsonl"
    print(f"Loading Game of 24 data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []
    
    # [수정] CSV 로더가 아닌 JSONL 로더로 로직 변경
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = []
            for line in f:
                data = json.loads(line)
                numbers = data.get("Puzzles", "")
                question = f"Use the numbers {numbers} and the operations (+, -, *, /) to get 24. Each number must be used exactly once."
                problems.append({"question": question, "answer": "24"}) # answer 키 추가
        
        print(f"Successfully loaded {len(problems)} problems from Game of 24.")
        return problems
    except Exception as e:
        print(f"An error occurred while loading the Game of 24 JSONL file: {e}")
        return []

def load_drop(split: str = "validation") -> List[Dict[str, Any]]:
    """
    drop.jsonl 파일에서 DROP 데이터를 로드합니다.
    """
    # [수정] 파일 경로를 실제 파일명으로 직접 지정
    file_path = DATA_DIR / "drop" / "drop.jsonl"
    print(f"Loading DROP data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]

        for p in problems:
            if 'context' not in p and 'passage' in p:
                p['context'] = p['passage']
        
        print(f"Successfully loaded {len(problems)} problems from DROP.")
        return problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []

def load_hotpotqa(split: str = "validation") -> List[Dict[str, Any]]:
    """
    hotpotqa.jsonl 파일에서 HotpotQA 데이터를 로드합니다.
    """
    # [수정] 파일 경로를 실제 파일명으로 직접 지정
    file_path = DATA_DIR / "hotpotqa" / "hotpotqa.jsonl"
    print(f"Loading HotpotQA data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]

        for p in problems:
            if isinstance(p.get('context'), list):
                all_sentences = [sent for title, sents in p['context'] for sent in sents]
                p['context'] = " ".join(all_sentences)
        
        print(f"Successfully loaded {len(problems)} problems from HotpotQA.")
        return problems
    except Exception as e:
        print(f"An error occurred while loading or processing the JSONL file: {e}")
        return []

def load_humaneval(split: str = "test") -> List[Dict[str, Any]]:
    """
    humaneval.jsonl 파일에서 HumanEval 데이터를 로드합니다.
    """
    # [수정] 파일 경로를 실제 파일명으로 직접 지정하고 humaneval 폴더 대신 mbpp_humaneval 사용 가능성 고려
    file_path = DATA_DIR / "humaneval" / "humaneval.jsonl"
    print(f"Loading HumanEval data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]
        
        formatted_problems = []
        for p in problems:
            formatted_problems.append({
                "question": p.get("text", ""),
                "answer": p.get("canonical_solution", "")
            })

        print(f"Successfully loaded {len(formatted_problems)} problems from HumanEval.")
        return formatted_problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []

def load_trivia_cw(split: str = "test") -> List[Dict[str, Any]]:
    """
    trivia_cw.jsonl 파일에서 TriviaQA 데이터를 로드합니다.
    """
    # [수정] 파일 경로를 실제 파일명으로 직접 지정
    file_path = DATA_DIR / "trivia_cw" / "trivia_cw.jsonl"
    print(f"Loading TriviaQA data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]
        
        print(f"Successfully loaded {len(problems)} problems from TriviaQA.")
        return problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []