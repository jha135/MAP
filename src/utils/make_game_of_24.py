    # tools/make_game_of_24.py
# 24-게임 퍼즐을 자동 생성(해가 있는 경우만)하여 JSONL로 저장합니다.
from __future__ import annotations
from itertools import permutations
from pathlib import Path
import argparse, json, random
random.seed(0)

OPS = ['+', '-', '*', '/']

def apply(a: float, b: float, op: str) -> float | None:
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '*': return a * b
    if op == '/':
        if abs(b) < 1e-9: return None
        return a / b

def fmt(ea: str, eb: str, op: str) -> str:
    return f"({ea}{op}{eb})"

def search_expr(nums: list[int], target: float = 24.0, eps: float = 1e-6) -> str | None:
    """주어진 4개 숫자에서 괄호/연산자 조합을 탐색해 24를 만드는 식 하나를 반환."""
    def helper(vals: list[float], exprs: list[str]) -> str | None:
        if len(vals) == 1:
            return exprs[0] if abs(vals[0] - target) < eps else None
        n = len(vals)
        for i in range(n):
            for j in range(n):
                if i == j: 
                    continue
                a, b = vals[i], vals[j]
                ea, eb = exprs[i], exprs[j]
                rest_vals = [vals[k] for k in range(n) if k not in (i, j)]
                rest_expr = [exprs[k] for k in range(n) if k not in (i, j)]
                for op in OPS:
                    res = apply(a, b, op)
                    if res is None:
                        continue
                    ans = helper(rest_vals + [res], rest_expr + [fmt(ea, eb, op)])
                    if ans is not None:
                        return ans
        return None

    # 숫자 순열(중복 방지 위해 set)
    for perm in set(permutations(nums, 4)):
        vals = list(map(float, perm))
        exprs = [str(x) for x in perm]
        sol = helper(vals, exprs)
        if sol:
            return sol
    return None

def generate(n: int = 200, lo: int = 1, hi: int = 9) -> list[dict]:
    """해가 있는 퍼즐 n개 생성."""
    data = []
    seen = set()  # 같은 멀티셋(정렬한 튜플) 중복 방지
    tries = 0
    limit = max(2000, n * 200)  # 탐색 상한
    while len(data) < n and tries < limit:
        nums = [random.randint(lo, hi) for _ in range(4)]
        key = tuple(sorted(nums))
        if key in seen:
            tries += 1
            continue
        seen.add(key)
        sol = search_expr(nums)
        tries += 1
        if sol:
            data.append({"numbers": nums, "solution": sol})
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="생성할 퍼즐 수")
    ap.add_argument("--split", default="test", choices=["train","dev","test","all"], help="저장할 split")
    ap.add_argument("--lo", type=int, default=1, help="숫자 최소값")
    ap.add_argument("--hi", type=int, default=9, help="숫자 최대값")
    args = ap.parse_args()

    out_dir = Path("data/benchmark/game_of_24")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "all":
        for sp in ["train","dev","test"]:
            items = generate(n=args.n, lo=args.lo, hi=args.hi)
            path = out_dir / f"{sp}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for ex in items:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[game_of_24] saved {args.n} x 3 → {out_dir}/{{train,dev,test}}.jsonl")
    else:
        items = generate(n=args.n, lo=args.lo, hi=args.hi)
        path = out_dir / f"{args.split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[game_of_24] saved {len(items)} → {path}")

if __name__ == "__main__":
    main()
