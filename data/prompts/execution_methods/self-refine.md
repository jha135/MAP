
You are an expert in {domain}.
Your task is to solve the following problem carefully.

Problem: {input}

First, provide your draft answer step by step.
Then, critically reflect on your draft answer and identify a principle to improve it.
Be honest and precise in pointing out issues such as logic gaps, arithmetic slips, missing conditions, or unclear explanation.

Output in JSON format:
{
  "draft_answer": "...",
  "principle": "..."
}

Step 2: Refined Final Answer 

You are an expert in {domain}.
Re-examine the previous draft answer with a critical eye and revise it only if necessary.

Problem: {input}
Previous Draft Answer: {draft_answer}

Guidelines:
- Consider the following principle extracted earlier: {principle}
- Double-check correctness, completeness, and adherence to the required format.
- If the draft is already correct, keep it.
- Perform all reasoning internally; do not reveal intermediate steps.

Final Answer (no explanations, only the required output):
