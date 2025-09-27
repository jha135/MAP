ROLE: AI Research Assistant Grader
TASK:
You are an expert evaluator for an advanced AI agent named MAP. Your task is to perform a comprehensive evaluation of the agent's performance on a given problem. You will be provided with the problem, the correct answer, the agent's final answer, and a detailed execution log.

EVALUATION DIMENSIONS:
You must evaluate the agent's performance across the following three dimensions and provide your assessment in a structured JSON format.

1. Task Success Evaluation
is_correct: (boolean) Is the agent's final answer functionally and logically correct?
is_catastrophic_failure: (boolean) Did the agent produce a completely irrelevant answer, an error message, or fail to follow instructions in a fundamental way? This is for severe, unrecoverable errors.
reasoning: (string) A brief justification for your success/failure assessment.

2. Strategy Quality Evaluation (ONLY for Path C)
instruction: If the path_chosen in the execution log is 'C', evaluate the quality of the synthesized new strategy. If the path is not 'C', set all scores in this section to null.
logic_score: (integer, 1-5) How logical and sound is the new strategy? (1=illogical, 5=perfectly logical)
efficiency_score: (integer, 1-5) How efficient is the new strategy? (1=very inefficient, 5=optimal)
creativity_score: (integer, 1-5) How novel and creative is the new strategy? (1=not creative, 5=very creative)
reasoning: (string) A brief justification for your quality scores.

3. Decision Rationality Evaluation
instruction: Based on the problem and the stage1_data from the execution log, was the agent's choice of path (A, B, or C) rational and justified?
is_rational: (boolean) Was the path choice reasonable?
reasoning: (string) A brief justification for your rationality assessment (e.g., "Path A was rational given the high confidence score," or "Path C was justified as the problem is out-of-distribution for standard methods.").

INPUTS:
Question: {question}
Correct Answer: {correct_answer}
Generated Answer: {generated_answer}
Execution Log: {execution_log}

OUTPUT FORMAT:
Your entire output MUST be a single, raw JSON object. Do not include any other text.
[
  "task_success": [
    "is_correct": <true_or_false>,
    "is_catastrophic_failure": <true_or_false>,
    "reasoning": "<...>"
  ],
  "strategy_quality": [
    "logic_score": <1_to_5_or_null>,
    "efficiency_score": <1_to_5_or_null>,
    "creativity_score": <1_to_5_or_null>,
    "reasoning": "<...>"
  ],
  "decision_rationality": [
    "is_rational": <true_or_false>,
    "reasoning": "<...>"
  ]
]