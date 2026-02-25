import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any

from scientist.solver_scientist import construct_solver # this will be the new relative import path according to scientist-merge branch

class Solver:
    def __init__(
        self,
        agent,
        task: str,
        data_file: str,
        task_description: str,
        output_types: str = "direct",
        index: int = 0,
        verbose: bool = False,
        max_steps: int = 10,
        max_time: int = 60,
        max_tokens: int = 4000,
        output_json_dir: str = "results",
        root_cache_dir: str = "cache",
        temperature: float = 0.7,
        add_tag: bool = False
    ):
        self.agent = agent
        self.task = task
        self.data_file = data_file
        self.task_description = task_description
        self.index = index
        self.verbose = verbose
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.output_json_dir = output_json_dir
        self.root_cache_dir = root_cache_dir
        self.temperature = temperature
        self.add_tag = add_tag

        self.output_types = output_types.lower().split(',')
        assert all(output_type in ["direct", "final", "schema"] for output_type in self.output_types), \
            "Invalid output type. Supported types are 'direct', 'final', 'schema'."

        self.benchmark_data = self.load_benchmark_data()

    def load_benchmark_data(self) -> List[Dict[str, Any]]:
        # Add task description to the query
        if self.task_description:
            print(f"Task description: {self.task_description}")
            self.task_description = f"Task description: {self.task_description}\n"

        with open(self.data_file, 'r') as f:
            data = json.load(f)
        output_format = "When ready, output the final answer enclosed in <answer> and </answer> tags. Do not generate any content after the </answer> tag."
        for problem in data:
            problem['query'] = problem['query'] if 'query' in problem else problem['question']
            if self.add_tag:
                problem['query'] += output_format
            if self.task_description:
                problem['query'] = self.task_description + problem['query']

            if 'image' in problem and problem['image'] not in [None, ""]:
                # NOTE: This is a hack to make the absolute image path relative to the data file
                problem['image'] = os.path.abspath(os.path.join(os.path.dirname(self.data_file), problem['image']))
                assert os.path.exists(problem['image']), f"Error: Image file {problem['image']} does not exist."

        return data

    def solve(self):
        total_problems = len(self.benchmark_data)

        # Solve a single problem
        if self.index is not None:
            if not 0 <= self.index < total_problems:
                print(f"Error: Invalid problem index {self.index}. Valid indices are 0 to {total_problems-1}).")
            else:
                self.solve_single_problem(self.index)
            return

    def solve_single_problem(self, index: int):
        """
        Solve a single problem from the benchmark dataset.

        Args:
            index (int): Index of the problem to solve
        """
        # Create output directory and file path
        json_dir = os.path.join(self.output_json_dir)
        os.makedirs(json_dir, exist_ok=True)
        output_file = os.path.join(json_dir, f"output_{index}.json")

        # Create logging directory
        log_dir = os.path.join(self.root_cache_dir, f"{index}")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "agent.log")

        # Get the problem
        problem = self.benchmark_data[index]
        # use 'query' by default for LLM inputs
        question = problem.get("query") if "query" in problem else problem["question"]
        image_path = problem.get("image", None)
        pid = problem['pid']
        answer = problem['answer']

        if self.verbose:
            print("\n\n")
            print("#"*100)
            print(f"## Problem {index}:")
            print(f"Question:\n{question}")
            print(f"Image: {image_path}")
            print("#"*100)

        # Prepare input files if there's an image
        input_files = []
        if image_path and os.path.exists(image_path):
            input_files.append(image_path)

        # Run the agent
        start_time = time.time()
        try:
            result = self.agent.run(
                agent_query=question,
                query_instruction=None,
                input_files=input_files,
                output_schema={},
                output_types=','.join(self.output_types),
                max_steps=self.max_steps,
                max_time=self.max_time,
                logging_path=log_file,
                output_path=None  # We'll save manually
            )

            # Extract the outputs
            outputs = result.get("outputs", {})
            final_output = outputs.get("final_output", {})
            step_history = outputs.get("step_history", {})
            global_plan = outputs.get("global_plan", "")

            # Build json_data with all required information
            json_data = {
                "pid": pid,
                "query": question,
                "image": image_path,
                "answer": answer,
                "global_plan": global_plan,
                "step_history": step_history,
                "execution_time": round(time.time() - start_time, 2),
                "step_count": len(step_history),
            }

            # Add metadata if available
            if 'metadata' in problem:
                json_data['metadata'] = problem['metadata']

            # Add outputs based on requested types
            if 'direct' in self.output_types:
                json_data["direct_output"] = final_output.get("direct", "")

            if 'final' in self.output_types:
                json_data["final_output"] = final_output.get("final", "")

            if 'schema' in self.output_types:
                json_data["schema_output"] = final_output.get("schema", {})

            if self.verbose:
                print("\n## Execution Result:")
                print("#"*50)
                print(f"Global Plan: {global_plan}")
                print(f"Steps executed: {len(step_history)}")
                print(f"Execution time: {json_data['execution_time']} seconds")

                if 'direct' in self.output_types:
                    print("\n## Direct Output:")
                    print("#"*50)
                    print(f"{json_data.get('direct_output', '')}")
                    print("#"*50)

                if 'final' in self.output_types:
                    print("\n## Final Output:")
                    print("#"*50)
                    print(f"{json_data.get('final_output', '')}")
                    print("#"*50)

        except Exception as e:
            print(f"Error during agent execution: {str(e)}")
            import traceback
            traceback.print_exc()

            json_data = {
                "pid": pid,
                "query": question,
                "image": image_path,
                "answer": answer,
                "error": str(e),
                "execution_time": round(time.time() - start_time, 2),
            }
        finally:
            if hasattr(self.agent, 'cleanup'):
                self.agent.cleanup()

        # Save results
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=4)
            print(f"\n==>Output saved to: {output_file}")

        # Print execution statistics
        print(f"\n## Execution Statistics for Problem {index}:")
        print(f"==>Total steps executed: {json_data.get('step_count', 0)}")
        print(f"==>Total execution time: {json_data['execution_time']:.2f} seconds")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the scientist agent with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o-mini", help="LLM engine name for all modules (if module_engine not specified).")
    parser.add_argument("--module_engine", default=None, help="Comma-separated list of engines for [planner,executor,verifier,generator]. E.g., 'gpt-4o,gpt-4o,gpt-4o,gpt-4o'")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens for LLM generation.")
    parser.add_argument("--task", default="benchmark", help="Task to run.")
    parser.add_argument("--data_file", default="data/data.json", help="Data file to run.")
    parser.add_argument("--task_description", default="", help="Task description.")
    parser.add_argument(
        "--output_types",
        default="direct",
        help="Comma-separated list of required outputs (direct,final,schema)"
    )
    parser.add_argument("--enabled_tools", default="Base_Generator_Tool,Wikipedia_RAG_Tool,Google_Search_Tool", help="List of enabled tools.")
    parser.add_argument("--tool_engine", default=None, help="List of tool engines corresponding to enabled_tools, separated by commas.")
    parser.add_argument("--index", type=int, default=0, help="Index of the problem in the benchmark file.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--output_json_dir", default="results", help="Path to output JSON directory.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM sampling temperature.")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for the LLM API.")
    parser.add_argument("--add_tag", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to add additional prompt to the question.")
    return parser.parse_args()


def main(args):
    # Parse enabled tools and tool engines
    enabled_tools = args.enabled_tools.split(",") if args.enabled_tools else []

    # Parse tool_engine
    if args.tool_engine:
        tool_engine = args.tool_engine.split(",")
    else:
        tool_engine = ["Default"] * len(enabled_tools)

    # Ensure tool_engine matches enabled_tools length
    if len(tool_engine) < len(enabled_tools):
        tool_engine += ["Default"] * (len(enabled_tools) - len(tool_engine))
    elif len(tool_engine) > len(enabled_tools):
        tool_engine = tool_engine[:len(enabled_tools)]

    # Parse module_engine
    if args.module_engine:
        module_engine = args.module_engine.split(",")
        if len(module_engine) != 4:
            raise ValueError(f"module_engine must have 4 elements [planner,executor,verifier,generator], got {len(module_engine)}")
    else:
        module_engine = None  # Will use llm_engine_name for all modules

    print(f"LLM Engine: {args.llm_engine_name}")
    print(f"Module Engine: {module_engine}")
    print(f"Enabled Tools: {enabled_tools}")
    print(f"Tool Engine: {tool_engine}")
    print(f"Base URL: {args.base_url}")

    # Construct the scientist agent
    agent = construct_solver(
        role_prompt="You are an expert AI assistant specialized in scientific research and problem-solving.",
        enabled_tools=enabled_tools,
        llm_engine_name=args.llm_engine_name,
        module_engine=module_engine,
        tool_engine=tool_engine,
        verbose=args.verbose,
        base_url=args.base_url,
        temperature=args.temperature,
        check_model=False,
        is_validation=True
    )

    # Instantiate Solver
    solver = Solver(
        agent=agent,
        task=args.task,
        data_file=args.data_file,
        task_description=args.task_description,
        output_types=args.output_types,
        index=args.index,
        verbose=args.verbose,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        output_json_dir=args.output_json_dir,
        root_cache_dir=args.root_cache_dir,
        temperature=args.temperature,
        add_tag=args.add_tag
    )

    # Solve the task or problem
    solver.solve()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
