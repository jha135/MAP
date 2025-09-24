import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "benchmarks"

def load_gsm8k(split: str = "test") -> List[Dict[str, Any]]:
    file_name = f"{split}-00000-of-00001.parquet"
    file_path = DATA_DIR / file_name

    print(f"Loading GSM8K data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        print("Please ensure the GSM8K Parquet file is in the 'data/benchmarks/gsm8k' directory.")
        return []

    try:
        df = pd.read_parquet(file_path)
        
        problems = df.to_dict('records')

        print(f"Successfully loaded {len(problems)} problems from GSM8K {split} set.")
        return problems

    except Exception as e:
        print(f"An error occurred while loading the Parquet file: {e}")
        return []

def load_game_of_24(split: str = "test") -> List[Dict[str, Any]]:
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
    file_name = f"{split}-00000-of-00001.parquet"
    file_path = DATA_DIR / "drop" / file_name

    print(f"Loading DROP data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        df = pd.read_parquet(file_path)
        df['answer'] = df['answers_spans'].apply(lambda x: x['spans'][0] if x['spans'] else "")
        problems = df[['passage', 'question', 'answer']].to_dict('records')

        print(f"Successfully loaded {len(problems)} problems from DROP {split} set.")
        return problems

    except Exception as e:
        print(f"An error occurred while loading the Parquet file: {e}")
        return []
def load_hotpotqa(split: str = "validation") -> List[Dict[str, Any]]:
    # Parquet 파일명을 완성합니다.
    file_name = f"{split}-00000-of-00001.parquet"
    file_path = DATA_DIR / "hotpotqa" / file_name

    print(f"Loading HotpotQA data from: {file_path}")

    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return []

    try:
        df = pd.read_parquet(file_path)
        def flatten_context(context_dict):
            all_sentences = [sentence for paragraph in context_dict['sentences'] for sentence in paragraph]
            return " ".join(all_sentences)

        df['context'] = df['context'].apply(flatten_context)
        
        problems = df[['question', 'answer', 'context']].to_dict('records')

        print(f"Successfully loaded {len(problems)} problems from HotpotQA {split} set.")
        return problems

    except Exception as e:
        print(f"An error occurred while loading or processing the Parquet file: {e}")
        return []
