import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "benchmarks"

def load_gsm8k(split: str = "test") -> List[Dict[str, Any]]:
    """
    Hugging Face Parquet 형식의 GSM8K 데이터를 로드합니다.
    prepare_benchmarks.py가 .jsonl로 저장하므로, 호환을 위해 수정이 필요할 수 있습니다.
    """
    file_path = DATA_DIR / "gsm8k" / f"{split}.jsonl"
    print(f"Loading GSM8K data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        print("Please run `prepare_benchmarks.py --name gsm8k` first.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]
        
        print(f"Successfully loaded {len(problems)} problems from GSM8K {split} set.")
        return problems
    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []

def load_game_of_24(split: str = "test") -> List[Dict[str, Any]]:
    """
    '24.csv' 파일에서 Game of 24 데이터를 로드합니다.
    """
    file_path = DATA_DIR / "game_of_24" / "24.csv"

    print(f"Loading Game of 24 data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    df = pd.read_csv(file_path)
    
    df = df.rename(columns={"Puzzles": "numbers"})
    df['question'] = df['numbers'].apply(lambda x: f"Use the numbers {x} and the operations (+, -, *, /) to get 24. Each number must be used exactly once.")
    
    problems = df[['question']].to_dict('records')

    print(f"Successfully loaded {len(problems)} problems from Game of 24.")
    return problems
    
def load_drop(split: str = "validation") -> List[Dict[str, Any]]:
    """
    DROP 데이터를 로드합니다. prepare_benchmarks.py는 dev.jsonl로 저장합니다.
    """
    # 'validation'을 'dev'로 매핑
    split_name = "dev" if split == "validation" else split
    file_path = DATA_DIR / "drop" / f"{split_name}.jsonl"

    print(f"Loading DROP data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        print("Please run `prepare_benchmarks.py --name drop` first.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]

        # passage 키를 context로 변경하여 일관성 유지
        for p in problems:
            if 'context' not in p and 'passage' in p:
                p['context'] = p['passage']
        
        print(f"Successfully loaded {len(problems)} problems from DROP {split} set.")
        return problems

    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []

def load_hotpotqa(split: str = "validation") -> List[Dict[str, Any]]:
    """
    HotpotQA 데이터를 로드합니다. prepare_benchmarks.py는 dev.jsonl로 저장합니다.
    """
    # 'validation'을 'dev'로 매핑
    split_name = "dev" if split == "validation" else split
    file_path = DATA_DIR / "hotpotqa" / f"{split_name}.jsonl"

    print(f"Loading HotpotQA data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        print("Please run `prepare_benchmarks.py --name hotpotqa` first.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]

        # 중첩된 context 구조를 단일 문자열로 변환
        for p in problems:
            if isinstance(p.get('context'), list):
                all_sentences = [sent for title, sents in p['context'] for sent in sents]
                p['context'] = " ".join(all_sentences)
        
        print(f"Successfully loaded {len(problems)} problems from HotpotQA {split} set.")
        return problems

    except Exception as e:
        print(f"An error occurred while loading or processing the JSONL file: {e}")
        return []

def load_mbpp(split: str = "test") -> List[Dict[str, Any]]:
    """
    'sanitized-mbpp.json' 파일에서 MBPP 데이터를 로드하고,
    'question'과 'answer' 키로 표준화합니다.
    """
    file_path = DATA_DIR / "mbpp_humaneval" / "sanitized-mbpp.json"
    print(f"Loading MBPP data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        print("Please run `prepare_benchmarks.py --name mbpp` first.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_problems = json.load(f)

        # 'task_id'를 기준으로 원하는 split의 데이터만 필터링
        prefix = f"mbpp-{split}"
        split_problems = [p for p in all_problems if p.get("task_id", "").startswith(prefix)]

        # 다른 로더와 형식을 맞추기 위해 키 이름 변경
        formatted_problems = []
        for p in split_problems:
            formatted_problems.append({
                "question": p.get("text", ""),
                "answer": p.get("canonical_solution", "")
            })

        print(f"Successfully loaded and filtered {len(formatted_problems)} problems from MBPP {split} set.")
        return formatted_problems

    except Exception as e:
        print(f"An error occurred while loading or processing the JSON file: {e}")
        return []

def load_humaneval(split: str = "test") -> List[Dict[str, Any]]:
    """
    'humaneval.jsonl' 파일에서 HumanEval 데이터를 로드하고,
    'question'과 'answer' 키로 표준화합니다.
    HumanEval은 'test' 스플릿만 존재합니다.
    """
    if split != "test":
        print(f"Warning: HumanEval only has a 'test' split. Ignoring requested split '{split}'.")
    
    file_path = DATA_DIR / "mbpp_humaneval" / "humaneval.jsonl"
    print(f"Loading HumanEval data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        print("Please run `prepare_benchmarks.py --name humaneval` first.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]
        
        # 다른 로더와 형식을 맞추기 위해 키 이름 변경
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
    TriviaQA 데이터를 로드합니다. (trivia_cw 폴더)
    """
    split_map = {
        "validation": "dev.jsonl",
        "test": "test.jsonl",
        "train": "train.jsonl"
    }
    file_name = split_map.get(split)
    if not file_name:
        raise ValueError(f"Invalid split for TriviaQA: {split}. Choose from 'train', 'validation', 'test'.")

    file_path = DATA_DIR / "trivia_cw" / file_name
    print(f"Loading TriviaQA data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        print("Please run `prepare_benchmarks.py --name trivia` first.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            problems = [json.loads(line) for line in f]
        
        print(f"Successfully loaded {len(problems)} problems from TriviaQA {split} set.")
        return problems

    except Exception as e:
        print(f"An error occurred while loading the JSONL file: {e}")
        return []