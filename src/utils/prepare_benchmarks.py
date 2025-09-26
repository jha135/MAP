# tools/prepare_benchmarks.py
# Hugging Face에서 벤치마크를 내려받아
# data/benchmark/<name>/ 아래 JSONL/JSON로 저장합니다.

from pathlib import Path
import argparse, json
from typing import Any, Dict, List
from datasets import load_dataset

ROOT = Path("data/benchmark")
ROOT.mkdir(parents=True, exist_ok=True)

def save_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- GSM8K ----------------
def prep_gsm8k() -> None:
    ds = load_dataset("gsm8k", "main")  # splits: train/test
    for split in ["train", "test"]:
        recs = []
        for i, row in enumerate(ds[split]):
            recs.append({
                "id": f"gsm8k-{split}-{i}",
                "question": row.get("question", ""),
                "answer": row.get("answer", "")  # 설명 포함 원문. 숫자만 쓰려면 후처리 가능
            })
        save_jsonl(recs, ROOT / "gsm8k" / f"{split}.jsonl")
    print("[gsm8k] saved → data/benchmark/gsm8k/{train,test}.jsonl")

# ---------------- HotpotQA ----------------
def prep_hotpotqa() -> None:
    ds = load_dataset("hotpot_qa", "distractor")  # splits: train/validation
    # train
    tr = []
    for i, row in enumerate(ds["train"]):
        tr.append({
            "_id": row.get("_id", f"hpqa-train-{i}"),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "context": row.get("context", []),  # [[title,[sent...]], ...]
            "type": row.get("type", "")
        })
    save_jsonl(tr, ROOT / "hotpotqa" / "train.jsonl")
    # dev (= validation)
    dv = []
    for i, row in enumerate(ds["validation"]):
        dv.append({
            "_id": row.get("_id", f"hpqa-dev-{i}"),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "context": row.get("context", []),
            "type": row.get("type", "")
        })
    save_jsonl(dv, ROOT / "hotpotqa" / "dev.jsonl")
    print("[hotpotqa] saved → data/benchmark/hotpotqa/{train,dev}.jsonl")

# ---------------- MBPP ----------------
def prep_mbpp() -> None:
    ds = load_dataset("mbpp")  # splits: train/validation/test
    out = []
    for split in ["train", "validation", "test"]:
        for i, row in enumerate(ds[split]):
            out.append({
                "task_id": row.get("task_id", f"mbpp-{split}-{i}"),
                "text": row.get("text") or row.get("prompt") or row.get("task_description") or "",
                "canonical_solution": row.get("code") or row.get("solution") or ""
            })
    p = ROOT / "mbpp_humaneval" / "sanitized-mbpp.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[mbpp] saved → data/benchmark/mbpp_humaneval/sanitized-mbpp.json")

# ---------------- HumanEval ----------------
def prep_humaneval() -> None:
    ds = load_dataset("openai_humaneval")  # split: test
    recs = []
    for i, row in enumerate(ds["test"]):
        recs.append({
            "task_id": row.get("task_id", f"humaneval-{i}"),
            "text": row.get("prompt", ""),
            "canonical_solution": row.get("canonical_solution", ""),
            "tests": row.get("test", "")
        })
    save_jsonl(recs, ROOT / "mbpp_humaneval" / "humaneval.jsonl")
    print("[humaneval] saved → data/benchmark/mbpp_humaneval/humaneval.jsonl")

# ---------------- TriviaQA (trivia_cw 대체) ----------------
def prep_trivia() -> None:
    ds = load_dataset("trivia_qa", "rc")  # splits: train/validation/test
    mapping = [("train", "train.jsonl"), ("validation", "dev.jsonl"), ("test", "test.jsonl")]
    for split, fname in mapping:
        recs = []
        for i, row in enumerate(ds[split]):
            ans = row.get("verified_answers") or row.get("answer", {}).get("value", "")
            recs.append({
                "id": row.get("question_id", f"trivia-{split}-{i}"),
                "question": row.get("question", ""),
                "answer": ans,
                "context": row.get("evidence", "")
            })
        save_jsonl(recs, ROOT / "trivia_cw" / fname)
    print("[trivia_qa] saved → data/benchmark/trivia_cw/{train,dev,test}.jsonl")

# ---------------- DROP ----------------
def prep_drop() -> None:
    """
    HF 'drop' 데이터셋을 받아서 data/benchmark/drop/{train,dev}.jsonl 로 저장.
    answer는 validated_answers > spans > number > date 순으로 선택.
    """
    ds = load_dataset("drop")   # splits: 'train', 'validation'
    def pick_answer(row):
        # 1) validated_answers (리스트) 우선
        va = row.get("validated_answers")
        if isinstance(va, list) and va:
            return va[0]
        # 2) spans(텍스트)
        spans = row.get("answers_spans", {}).get("spans", [])
        if isinstance(spans, list) and spans:
            return spans[0]
        # 3) number
        num = row.get("answers_spans", {}).get("number")
        if num not in (None, ""):
            return str(num)
        # 4) date(dict)
        date = row.get("answers_spans", {}).get("date")
        if isinstance(date, dict):
            y = str(date.get("year", "")).strip()
            m = str(date.get("month", "")).strip()
            d = str(date.get("day", "")).strip()
            cand = "-".join(x for x in [y, m, d] if x)
            if cand:
                return cand
        return ""

    # train
    tr = []
    for i, row in enumerate(ds["train"]):
        tr.append({
            "id": row.get("query_id", f"drop-train-{i}"),
            "question": row.get("question", ""),
            "context": row.get("passage", ""),
            "answer": pick_answer(row),
            "split": "train",
        })
    save_jsonl(tr, ROOT / "drop" / "train.jsonl")

    # dev (= validation)
    dv = []
    for i, row in enumerate(ds["validation"]):
        dv.append({
            "id": row.get("query_id", f"drop-dev-{i}"),
            "question": row.get("question", ""),
            "context": row.get("passage", ""),
            "answer": pick_answer(row),
            "split": "dev",
        })
    save_jsonl(dv, ROOT / "drop" / "dev.jsonl")
    print("[drop] saved → data/benchmark/drop/{train,dev}.jsonl")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--name",
        required=True,
        choices=["gsm8k", "hotpotqa", "mbpp", "humaneval", "trivia", "drop", "all"],
    )
    args = ap.parse_args()
    if args.name in ("gsm8k", "all"):       prep_gsm8k()
    if args.name in ("hotpotqa", "all"):    prep_hotpotqa()
    if args.name in ("mbpp", "all"):        prep_mbpp()
    if args.name in ("humaneval", "all"):   prep_humaneval()
    if args.name in ("trivia", "all"):      prep_trivia()
    if args.name in ("drop", "all"):        prep_drop()

if __name__ == "__main__":
    main()
