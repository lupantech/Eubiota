"""
Utilities for Scientist Training
Includes LLM-based scoring and evaluation
"""

import re
import sys
import os
from pydantic import BaseModel

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
trainer_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(trainer_dir)

# Only add project_root to sys.path
# This ensures 'trainer' is recognized as a package (not trainer/trainer.py directly)
while project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from scientist.engine.openai import ChatOpenAI


class AnswerVerification(BaseModel):
    """Schema for LLM answer verification"""
    analysis: str
    true_false: bool


try:
    llm_scorer_engine = ChatOpenAI(
        model_string="gpt-4o",
        is_multimodal=False,
        use_cache=True
    )
    print(f"\nLLM Scorer engine '{llm_scorer_engine.model_string}' initialized successfully.\n")
except Exception as e:
    print(f"Failed to initialize LLM Scorer engine: {e}")
    llm_scorer_engine = None


def compute_score(question: str, groundtruth: str, answer_extracted: str) -> bool:
    """
    Uses gpt-4o to determine if the extracted answer matches the groundtruth.

    Args:
        question: The full question text, including options.
        answer_extracted: The answer provided by the model being evaluated.
        groundtruth: The correct answer label (e.g., "A").

    Returns:
        A boolean indicating whether the answer is correct.
    """
    if llm_scorer_engine is None:
        raise RuntimeError("LLM Scorer engine is not available.")

    query_prompt = f"""
You are a precise evaluator. Determine if the Model Response is equivalent to the Ground Truth.

**Instructions:**
1.  **Extract:** Isolate the final answer from the Model Response, ignoring reasoning. Look for `\\boxed{{...}}` or concluding statements.
2.  **Normalize & Compare:** The extracted answer and Ground Truth must be equivalent after normalization:
    - **Math:** Mathematically identical (e.g., `\\frac{{1}}{{2}}` == `0.5`).
    - **Numbers/Text:** Ignore formatting, case, and currency/units (e.g., `1,000` == `1000`).
    - **MCQ:** Match option content (e.g., "Paris") or number (e.g., `3rd` option) to the correct letter.
3.  **Verdict:** "True" only for semantically or mathematically equivalent answers.

**Inputs:**
Question: {question}
Model Response: {answer_extracted}
Ground Truth: {groundtruth}

**Format:**
<analysis>: Brief analysis of the comparison.
<true_false>: "True" or "False".
"""

    verification_result = llm_scorer_engine(query_prompt, response_format=AnswerVerification)

    return verification_result.true_false


def eval(question: str, groundtruth: any, answer_extracted: any, val: bool = False) -> float:
    """
    Evaluates if the extracted answer is correct by calling an LLM judge (gpt-4o).
    It strip(), and matches the final answer.
    """
    question_str = str(question)
    groundtruth_str = str(groundtruth)
    answer_extracted_str = str(answer_extracted)

    is_correct = compute_score(question_str, groundtruth_str, answer_extracted_str)

    return 1.0 if is_correct else 0.0


async def main():
    """Test cases for the scorer"""
    # ==============================================================================
    # ==============================================================================
    print("--- Running Simple Case ---")
    simple_question = "What is the capital of France?\nA) Berlin\nB) Madrid\nC) Paris\nD) Rome"
    simple_groundtruth = "C"
    simple_model_answer = "The correct answer is C."
    score1 = eval(simple_question, simple_groundtruth, simple_model_answer)
    print(f"Question: {simple_question}")
    print(f"Model Answer: '{simple_model_answer}'")
    print(f"Ground Truth: '{simple_groundtruth}'")
    print(f"==> Score: {score1}\n")  # 1.0

    # ==============================================================================
    # ==============================================================================
    print("--- Running Case with LaTeX Formula ---")
    latex_question = r"""
Calculate the definite integral of $f(x) = 2x$ from $x=1$ to $x=3$.
A) 4
B) 6
C) 8
D) 10
"""
    latex_groundtruth = "C"
    latex_model_answer = r"""
To solve this, we need to compute the integral $\int_{1}^{3} 2x \,dx$.
The antiderivative of $2x$ is $x^2$.
Using the Fundamental Theorem of Calculus, we evaluate this at the bounds:
$F(b) - F(a) = 3^2 - 1^2 = 9 - 1 = 8$.
"""
    score2 = eval(latex_question, latex_groundtruth, latex_model_answer)
    print(f"Question: {latex_question.strip()}")
    print(f"Model Answer: '{latex_model_answer.strip()}'")
    print(f"Ground Truth: '{latex_groundtruth}'")
    print(f"==> Score: {score2}\n")  # 1.0

    # ==============================================================================
    # ==============================================================================
    print("--- Running Case with Multiple Intermediate Answers ---")
    multi_answer_question = """
A project has two phases. Phase 1 costs $5,000 and takes 3 months. Phase 2 costs $8,000 and takes 4 months. What is the total duration of the project?
A) $13,000
B) 4 months
C) 7 months
D) $5,000
"""
    multi_answer_groundtruth = "C"
    multi_answer_model_response = """
Let's analyze the problem.
The cost of Phase 1 is $5,000 and the duration is 3 months.
The cost of Phase 2 is $8,000 and the duration is 4 months.
The total cost would be $5,000 + $8,000 = $13,000.
The question asks for the total duration, which is 3 months + 4 months = 7 months.
Therefore, the final answer is 7 months. This matches option C.
"""
    score3 = eval(multi_answer_question, multi_answer_groundtruth, multi_answer_model_response)
    print(f"Question: {multi_answer_question.strip()}")
    print(f"Model Answer: '{multi_answer_model_response.strip()}'")
    print(f"Ground Truth: '{multi_answer_groundtruth}'")
    print(f"==> Score: {score3}\n")  # 1.0


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
