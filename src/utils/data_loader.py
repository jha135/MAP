import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import random
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "benchmark"

def load_gsm8k(split: str = "test") -> List[Dict[str, Any]]:
    """
    gsm.jsonl 파일에서 GSM8K 데이터를 로드합니다.
    """
    file_path = DATA_DIR / "gsm8k" / "gsm.jsonl"
    print(f"Loading GSM8K data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]
        
        print(f"Successfully loaded {len(problems)} problems from GSM8K.")
        random.shuffle(problems)
        return problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []

def load_game_of_24(split: str = "test") -> List[Dict[str, Any]]:
    """
    game_of_24.jsonl 파일에서 Game of 24 데이터를 로드합니다.
    """
    # split 매개변수는 현재 사용되지 않지만, 함수의 시그니처를 유지합니다.
    file_path = DATA_DIR / "game_of_24" / "game_of_24.jsonl"
    print(f"Loading Game of 24 data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        print(f"Please ensure the file exists at: {file_path}")
        return []
    
    problems = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 빈 줄이나 공백만 있는 줄은 건너뜁니다.
                if not line.strip():
                    continue  
                data = json.loads(line)
                question = data.get("question", "")
                answer = data.get("answer", "")
                if question and answer:
                    problems.append({"question": question, "answer": answer})
                else:
                     print(f"Warning: Skipping invalid line (missing key): {line.strip()}", file=sys.stderr)
        print(f"Successfully loaded {len(problems)} problems from Game of 24.")
        random.shuffle(problems)
        return problems
    except json.JSONDecodeError as e:
        print(f"An error occurred while parsing the JSONL file: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading the Game of 24 data: {e}", file=sys.stderr)
        return []

def load_drop(split: str = "validation") -> List[Dict[str, Any]]:
    """
    drop.jsonl 파일에서 DROP 데이터를 로드합니다.
    """
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
        random.shuffle(problems)
        return problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []

def load_hotpotqa(split: str = "validation") -> List[Dict[str, Any]]:
    """
    hotpotqa.jsonl 파일에서 HotpotQA 데이터를 로드합니다.
    """
    file_path = DATA_DIR / "hotpotqa" / "hotpotqa.jsonl"
    print(f"Loading HotpotQA data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]

        for p in problems:
            # context가 딕셔너리이고 'sentences' 키를 가지고 있는지 확인
            if isinstance(p.get('context'), dict) and 'sentences' in p['context']:
                # sentences는 리스트의 리스트이므로, 이를 펼쳐서 하나의 리스트로 만듭니다.
                all_sentences = [sent for sublist in p['context']['sentences'] for sent in sublist]
                p['context'] = " ".join(all_sentences)
        
        print(f"Successfully loaded {len(problems)} problems from HotpotQA.")
        random.shuffle(problems)
        
        # 처리된 데이터 확인
        for p in problems:
            print("\\n--- Loaded Problem ---")
            print("Question:", p['question'])
            print("Answer:", p['answer'])
            print("Processed Context:", p['context'])

        return problems
    except Exception as e:
        print(f"An error occurred while loading or processing the JSONL file: {e}")
        return []

def load_humaneval(split: str = "test") -> List[Dict[str, Any]]:
    """
    humaneval.jsonl 파일에서 HumanEval 데이터를 로드합니다.
    """
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
        random.shuffle(formatted_problems)
        return formatted_problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []

def load_trivia_cw(split: str = "test") -> List[Dict[str, Any]]:
    """
    trivia_cw.jsonl 파일에서 TriviaQA 데이터를 로드합니다.
    """
    file_path = DATA_DIR / "trivia_cw" / "trivia_cw.jsonl"
    print(f"Loading TriviaQA data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]
        
        print(f"Successfully loaded {len(problems)} problems from TriviaQA.")
        random.shuffle(problems)
        return problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []