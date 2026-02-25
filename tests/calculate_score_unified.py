import concurrent.futures
import os, re
import json
import argparse
import tqdm
import sys
from pydantic import BaseModel
from difflib import SequenceMatcher
from scientist.engine.openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import ResultAnalyzer

class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool

class BinaryAnswerVerification(BaseModel):
    true_false: bool

class AnswerExtraction(BaseModel):
    analysis: str
    extracted_option: str

# Two-step scoring prompt for MCQ tasks
SCORING_PROMPT = """
You are an expert grader for multiple-choice questions. Your task is to extract the best-matching answer option (A, B, C, etc.) from a model's free-form response, based on the provided choices.

## Input:
- Model Response: {response}
- Answer Choices: {choices}

## Instructions:
1. Carefully read and interpret the model's response.
2. Determine the final answer *implied or explicitly stated* in the response.
3. Match it to the closest option in the provided choices.
4. Return your analysis and the extracted option string.

## Output Format (must match this exactly):
<analysis>: A brief explanation of how you identified the final answer in the model's response and how it maps to one of the given options.
<extracted_option>: The exact option text (e.g., "A. 0.9", "B. 0.7") from the choices that best matches the model's response.

## Example:
Input:
Model Response: "The model's response is that the answer is A."
Answer Choices: ["A. 0.9", "B. 0.7", "C. 0.1", "D. 0.3"]

Output:
<analysis>: The model's response implies that the answer is A, which is the best match to the provided choices.
<extracted_option>: "A. 0.9"

Make sure <extracted_option> is one of the choices: {choices}.
"""

def find_most_similar_candidate(target, candidates):
    """
    Find the candidate that is most similar to the target.
    Target is a string.
    Candidates is a list of strings.
    Returns the most similar candidate string.
    """
    if not candidates:
        return None
    
    # If candidates is a dict, convert to list of strings
    if isinstance(candidates, dict):
        candidates = [f"{k}. {v}" for k, v in candidates.items()]
    
    max_similarity = 0
    most_similar_candidate = candidates[0]  # Default to first candidate

    for candidate in candidates:
        # Calculate similarity between target and candidate
        similarity = SequenceMatcher(None, target.lower(), candidate.lower()).ratio()

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_candidate = candidate

    return most_similar_candidate

class ResultScorer:
    def __init__(self, llm_engine=None, task_name="gaia"):
        self.llm_engine = llm_engine or ChatOpenAI(model_string="gpt-4o", is_multimodal=False, enable_cache=True)
        self.task_name = task_name
        # Tasks that require two-step MCQ scoring
        self.mcq_tasks = ["bomixqa", "lab-bench-litqa2", "lab-bench-protocolqa",
                         "mmlu-bio", "pubmedqa", "wmdp-bio", "medqa", "gpqa", "microbio"]
        print(f"\nLocal OpenAI engine {self.llm_engine.model_string} initialized.")
        print(f"Task: {self.task_name}")
        print(f"Scoring method: {'Two-step MCQ' if self.task_name in self.mcq_tasks else 'Direct verification'}\n")

    def answer_verification_mcq(self, question, response, correct_answer, choices):
        """Two-step MCQ scoring: extract option, then compare"""
        # Extract <answer> tags if present
        all_matches = re.findall(r"<answer>(.*?)</answer>", str(response), re.DOTALL)
        if all_matches:
            response = all_matches[-1].strip()

        # Step 1: Extract the option from model's response
        prompt = SCORING_PROMPT.format(response=response, choices=choices)
        extraction = self.llm_engine(prompt, response_format=AnswerExtraction)

        extraction_analysis = extraction.analysis
        extracted_option = extraction.extracted_option

        # Step 2: Find the most similar candidate from choices
        if extracted_option in choices:
            normalized_extracted_option = extracted_option
        else:
            normalized_extracted_option = find_most_similar_candidate(extracted_option, choices)

        # Step 3: Compare with correct answer
        true_false = (normalized_extracted_option == correct_answer)

        # Build analysis
        analysis = {
            "extraction_analysis": extraction_analysis,
            "extracted_option": extracted_option,
            "normalized_extracted_option": normalized_extracted_option,
            "correct_answer": correct_answer,
            "true_false": true_false
        }

        return analysis, true_false

    def answer_verification_direct(self, question, response, correct_answer):
        """Direct verification for non-MCQ tasks (e.g., GAIA)"""
        all_matches = re.findall(r"<answer>(.*?)</answer>", str(response), re.DOTALL)
        if all_matches:
            response = all_matches[-1].strip()

        query_prompt = f"""
Given a Question, a Model Response, and its Correct Answer, determine whether the Model's prediction is correct.

The prediction is correct only if it **exactly matches** the correct answer after necessary normalization. Follow these instructions carefully:

1. Extract the core answer from the Model Response, ignoring irrelevant text or explanations.
2. Normalize both the predicted answer and the correct answer (e.g., lowercase, remove extra spaces).
3. Compare the normalized answers.

Question: {question}
Model response: {response}
Correct answer: {correct_answer}

Response Format:
<analysis>: First extract the answer, then explain the comparison
<true_false>: Return "True" only for exact matches, otherwise "False"
        """

        verification = self.llm_engine(query_prompt, response_format=AnswerVerification)

        analysis = verification.analysis.strip()
        true_false = verification.true_false

        return analysis, true_false

    def answer_verification(self, question, response, correct_answer, choices=None):
        """Route to appropriate verification method based on task type"""
        if self.task_name in self.mcq_tasks and choices is not None:
            return self.answer_verification_mcq(question, response, correct_answer, choices)
        else:
            return self.answer_verification_direct(question, response, correct_answer)

    def score_results(self, results, max_workers=10):
        correct = 0

        def process_single_result(pid_data):
            pid, question_data = pid_data
            question = question_data["question"]
            response = question_data["response"]
            correct_answer = question_data["correct_answer"]
            # Get choices if available (for MCQ tasks)
            choices = question_data.get("choices", None)
            analysis, true_false = self.answer_verification(question, response, correct_answer, choices)
            return pid, analysis, true_false

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_result, (pid, data))
                      for pid, data in results.items()]

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures),
                                  total=len(futures),
                                  desc="Scoring results"):
                pid, analysis, true_false = future.result()
                correct += 1 if true_false else 0
                results[pid].update({
                    "stepwise_analysis": analysis,
                    "true_false": true_false
                })

        return results, correct


def load_data(data_file, result_dir, response_type):
    # Load the benchmark data
    with open(data_file, 'r') as f:
        # convert the benchmark data to a dictionary
        benchmark_data = {data["pid"]: data for data in json.load(f)}

    # Load the results
    results = {}
    for file in os.listdir(result_dir):
        if file.endswith(".json") and "output_" in file:
            file_path = os.path.join(result_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)

                # Get the index of the result
                index = file.replace(".json", "").replace("output_", "") # "0", "1", "2", ...
                # Try using index as string first, if not found then try as int
                try:
                    pid = int(index)
                    if pid not in benchmark_data:
                        pid = str(int(index))
                except (ValueError, KeyError):
                    pid = index
                assert result["pid"] == benchmark_data[pid]["pid"]

                # Save the results
                results[pid] = benchmark_data[pid]
                assert response_type in result
                results[pid]["response"] = result[response_type]
                results[pid]["correct_answer"] = benchmark_data[pid]["answer"]
                # print(f"successfully read: {file}")

            except json.JSONDecodeError as e:
                print(f"JSON decode error, cannot parse the file: {file}, Error message: {e}")
            except Exception as e:
                print(f"Unknown error: {file}, Error message: {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Universal script to extract and score results from benchmark data for all tasks")
    parser.add_argument("--task_name", type=str, required=True,
                        help="The name of the task (e.g., aime24, bamboogle, gaia, gameof24)")
    parser.add_argument("--data_file", type=str, default=None,
                        help="The file containing the benchmark data (default: {task_name}/data/test.json)")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="The directory containing the results (default: {task_name}/results/{exp_name})")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="The experiment name (used to construct result_dir if not specified)")
    parser.add_argument("--output_file", type=str, default="final_results.json",
                        help="The file to save the extracted results")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="The directory containing the logs")
    parser.add_argument("--response_type", type=str, default="direct_output",
                        choices=["final_output", "direct_output", "base_response"],
                        help="The type of response to extract from the results")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="The maximum number of workers to use")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get the base directory (tasks folder)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(base_dir, args.task_name)

    # Set default paths if not provided
    if args.data_file is None:
        args.data_file = os.path.join(task_dir, "data", "test.json")

    if args.result_dir is None:
        if args.exp_name is None:
            raise ValueError("Either --result_dir or --exp_name must be specified")
        args.result_dir = os.path.join(task_dir, "results", args.exp_name)

    # Validate paths
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    if not os.path.exists(args.result_dir):
        raise FileNotFoundError(f"Result directory not found: {args.result_dir}")

    # Load and print the arguments
    print("#"*50)
    print(f"Task: {args.task_name}")
    print(f"Arguments: {args}")
    for arg, value in args.__dict__.items():
        print(f"# {arg}: {value}")
    print("#"*50)

    # Initialize scorer with task_name for proper scoring method selection
    scorer = ResultScorer(task_name=args.task_name)
    analyzer = ResultAnalyzer()

    # Load the results
    results = load_data(args.data_file, args.result_dir, args.response_type)

    # Score the results
    results, correct = scorer.score_results(results, max_workers=args.max_workers)

    # Calculate accuracy and wrong answers
    acc = round(correct / len(results) * 100, 2)
    print(f"\nAccuracy: {acc}% ({correct}/{len(results)})")

    # Save detailed results
    output_file = os.path.join(args.result_dir, args.output_file)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")

    # Calculate wrong answers
    wrong_pids = [pid for pid, data in results.items() if not data["true_false"]]
    wrong_pids = sorted(wrong_pids, key=lambda x: int(x))
    wrong_indices = [int(pid) for pid in wrong_pids]
    print(f"Wrong PIDs: {wrong_pids}")
    print(f"Wrong Indices: {wrong_indices}")

    scores = {
        "correct": correct,
        "total": len(results),
        "accuracy": acc,
        "wrong_pids": wrong_pids,
        "wrong_indices": wrong_indices
    }

    # Calculate additional statistics if log directory is provided
    log_dir = args.log_dir or args.result_dir.replace("results", "logs")
    if os.path.exists(log_dir):

        if args.response_type == "base_response":
            print("Base response is not supported for scoring.")
            print("Exited.\n")
            exit()

         # Calculate the average time and steps
        step_stats = analyzer.calculate_time_steps(log_dir)
        print(f"\nStep stats:")
        for key, value in step_stats.items():
            print(f"- {key}: \t{value}")

        # Calculate the usage of tools
        tool_usage = analyzer.calculate_tool_usage(args.result_dir)
        print(f"\nTool usage:")
        for tool, ratio in tool_usage.items():
            print(f"- {tool}: \t{ratio}")

        # Update the scores
        scores.update({
            "step_stats": step_stats,
            "tool_usage": tool_usage
        })


    # Save the scores
    score_file = os.path.join(args.result_dir, f"final_scores_{args.response_type}.json")
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=4)
        print(f"Scores saved to {score_file}")


if __name__ == "__main__":
    main()