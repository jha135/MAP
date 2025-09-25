Step 1: Step-Back Abstraction

You are an expert in {domain}.
Your task is to solve the following problem carefully.

Problem: {input}

First, instead of solving it directly, take a step back.
- Restate the problem at a higher, more general level.
- Identify general principles, rules, or abstract structures relevant to this problem type.
- Do not produce the final solution yet.

Output in JSON format:
{
  "problem_restatement": "...",
  "general_principles": ["...", "..."],
  "solution_outline": ["...", "..."]
}

Step 2: Apply Back to the Original Problem

You are an expert in {domain}.
Re-examine the original problem using the abstraction from Step 1.

Problem: {input}
Step-Back Abstraction: {step1_json}

Guidelines:
- Follow the "general_principles" and "solution_outline" extracted earlier.
- Use them to solve the problem step by step.
- If additional assumptions are required, state them briefly.
- Perform all reasoning internally; do not reveal intermediate steps.

Final Answer (no explanations, only the required output):