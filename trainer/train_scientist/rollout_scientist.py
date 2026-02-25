"""
Rollout for Scientist Agent Training
Adapted from AgentFlow's rollout.py to use scientist.solver_scientist
"""

import os
import re
import sys
from typing import Any, Optional


# Add paths for imports (ORDER MATTERS!)
current_dir = os.path.dirname(os.path.abspath(__file__))
trainer_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(trainer_dir)

# IMPORTANT: Only add project_root to sys.path
# This ensures 'trainer' is recognized as a package (not trainer/trainer.py directly)
# Remove project_root if it exists (to avoid duplicates)
while project_root in sys.path:
    sys.path.remove(project_root)

# Insert project_root at position 0
# This allows: from trainer import ... (imports trainer package correctly)
sys.path.insert(0, project_root)

from trainer import Trainer, LitAgent, NamedResources, LLM, reward, configure_logger

# Import from agentflow.scientist (uses second path: agentflow/)
from scientist.solver_scientist import construct_solver
from datetime import datetime
import uuid, json
from filelock import FileLock
import asyncio

from utils import compute_score

configure_logger()


@reward
async def eval(question: str, groundtruth: any, answer_extracted: any, val: bool = False) -> float:
    """
    Evaluates if the extracted answer is correct by calling an LLM judge (gpt-4o).
    It strip(), and matches the final answer.
    """
    question_str = str(question)
    groundtruth_str = str(groundtruth)
    answer_extracted_str = str(answer_extracted)

    is_correct = compute_score(question_str, groundtruth_str, answer_extracted_str)

    return 1.0 if is_correct else 0.0


class ScientistRollout:
    """
    Scientist Agent Rollout wrapper.
    Uses scientist.solver_scientist instead of agentflow.solver
    """
    def __init__(
        self,
        resources: NamedResources,
        llm_engine_name: str = "gpt-4o",
        enabled_tools: list[str] = ["Base_Generator_Tool"],
        tool_engine: list[str] = ["Default"],
        module_engine: list[str] = None,
        output_types: str = "final,direct",
        max_steps: int = 3,
        max_time: int = 500,
        max_tokens: int = 2048,
        base_url: str = None,
        verbose: bool = True,
        temperature: float = 0.0,
    ):
        """
        Initialize Scientist Rollout.

        Args:
            resources: Named resources from the trainer
            llm_engine_name: LLM model name (e.g., "gpt-4o", "qwen2.5-7b-instruct")
            enabled_tools: List of tool names to enable
            tool_engine: List of engines for each tool (or "Default")
            module_engine: List of engines for [planner, executor, verifier, generator]
                          "Trainable"/"Default" uses main model, otherwise uses specified model
            output_types: Comma-separated output types (e.g., "direct,final")
            max_steps: Maximum reasoning steps
            max_time: Maximum execution time in seconds
            max_tokens: Maximum tokens for base response
            base_url: Base URL for vLLM or custom API endpoint
            verbose: Whether to print verbose output
            temperature: LLM sampling temperature
        """
        assert len(tool_engine) == len(enabled_tools), \
            f"tool_engine length ({len(tool_engine)}) must match enabled_tools length ({len(enabled_tools)})"

        print(f"********MODEL {llm_engine_name} SERVED AT {base_url or 'default'}***********")

        self.resources = resources
        self.llm_engine = llm_engine_name

        # Default module_engine: all modules use trainable model
        if module_engine is None:
            module_engine = ["Trainable", "Trainable", "Trainable", "Trainable"]

        # Validate module_engine length
        assert len(module_engine) == 4, \
            f"module_engine must have 4 elements [planner, executor, verifier, generator], got {len(module_engine)}"

        # For vLLM models, add vllm- prefix if needed
        # This allows the engine factory to route to the correct backend
        prefix = "" if "gpt" in llm_engine_name or "claude" in llm_engine_name else "vllm-"
        model_string = prefix + llm_engine_name

        # Process module_engine: convert "Trainable"/"Default" to actual model string
        processed_module_engine = []
        for engine in module_engine:
            if engine.lower() in ["trainable", "default"]:
                processed_module_engine.append(model_string)
            else:
                processed_module_engine.append(engine)

        print(f"Module engines: planner={processed_module_engine[0]}, executor={processed_module_engine[1]}, "
              f"verifier={processed_module_engine[2]}, generator={processed_module_engine[3]}")

        # Construct the scientist solver
        self.solver = construct_solver(
            llm_engine_name=model_string,
            enabled_tools=enabled_tools,
            tool_engine=tool_engine,
            module_engine=processed_module_engine,
            verbose=verbose,
            base_url=base_url,
            temperature=temperature,
            check_model=False  # Skip model checking for faster initialization
        )

        self.output_types = output_types
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.verbose = verbose

    def solve(self, question: str, image_path: Optional[str] = None) -> dict:
        """
        Solve a question using the scientist agent.

        Args:
            question: The question/prompt to solve
            image_path: Optional path to image (for multimodal tasks)

        Returns:
            Dictionary containing:
            - direct_output: Direct output response (extracted for convenience)
            - agent_result: Complete Agent.run() result with all module I/O
              - configs: Agent configuration
              - inputs: Input parameters
              - outputs: All outputs including:
                - global_plan: Global planning result
                - step_history: Complete step-by-step execution history
                  - Each step contains Planner, Executor, Verifier details
                - final_output: Generated outputs for each requested type
                - final_status: 'success', 'max_steps_reached', 'timeout', or 'error'
                - total_execution_time: Total execution time
        """
        # Call Agent.run() method
        agent_result = self.solver.run(
            agent_query=question,
            query_instruction=None,
            input_files=[],
            output_schema={},
            output_types=self.output_types,
            max_steps=self.max_steps,
            max_time=self.max_time,
            logging_path=None,
            output_path=None
        )

        # Extract direct_output for backward compatibility
        direct_output = ""
        if "outputs" in agent_result and "final_output" in agent_result["outputs"]:
            final_output = agent_result["outputs"]["final_output"]
            # Try to get direct output first, fallback to final
            if "direct" in final_output and "response" in final_output["direct"]:
                direct_output = final_output["direct"]["response"]
            elif "final" in final_output and "response" in final_output["final"]:
                direct_output = final_output["final"]["response"]

        # Build result with both formats
        result = {
            "direct_output": direct_output,  # For backward compatibility
            "agent_result": agent_result     # Complete structured result
        }

        if self.verbose:
            print(f"\n==> 📝 Scientist Solver Result:")
            print(f"Direct Output: {direct_output[:200]}..." if len(direct_output) > 200 else f"Direct Output: {direct_output}")
            print(f"Status: {agent_result.get('outputs', {}).get('final_status', 'unknown')}")
            print(f"Steps: {agent_result.get('outputs', {}).get('execution_metadata', {}).get('steps_completed', 0)}")

        return result


def get_agent(
    model: str,
    openai_base_url: str,
    temperature: float,
    resources,
    tools: list[str],
    max_steps: int,
    tool_engine: str,
    module_engine: list[str],
    max_tokens: int,
    output_type: str,
    timeout: int,
):
    """
    Create a Scientist agent instance.

    Args:
        model: Model name
        openai_base_url: OpenAI-compatible API base URL (e.g., vLLM endpoint)
        temperature: Sampling temperature
        resources: Named resources
        tools: List of enabled tools
        max_steps: Maximum reasoning steps
        tool_engine: Tool engine configuration
        module_engine: Module engine configuration [planner, executor, verifier, generator]
        max_tokens: Maximum tokens
        output_type: Output types (comma-separated)
        timeout: Maximum execution time

    Returns:
        ScientistRollout instance
    """
    llm_engine_name = model

    # Handle vLLM base URL
    if openai_base_url and openai_base_url != "https://api.openai.com/v1":
        vllm_base_url = openai_base_url
    else:
        vllm_base_url = None

    # Note: `max_time` and `verbose` are set to constant values here.
    # If these need to be dynamic, you would also need to add them to the function parameters.
    agent = ScientistRollout(
        resources=resources,
        llm_engine_name=llm_engine_name,
        enabled_tools=tools,
        tool_engine=tool_engine,
        module_engine=module_engine,
        max_steps=max_steps,
        max_tokens=max_tokens,
        base_url=vllm_base_url,
        verbose=True,
        output_types=output_type,
        max_time=timeout,
        temperature=temperature
    )
    return agent


class Rollout(LitAgent):
    """
    Scientist Rollout Handler for Training.

    This class manages the rollout process during training:
    - Creates agents for training and validation
    - Manages rollout data collection and storage
    - Handles synchronization between training steps
    """

    def __init__(
        self,
        server_public_ip: str = "Default",
        exp_name: str = "scientist_exp",
        rollout_n: int = 8,
        batch_size: int = 16,
        enabled_tools: list[str] = ["Base_Generator_Tool", "Wikipedia_RAG_Tool"],
        tool_engine: list[str] = ["Default", "Default"],
        module_engine: list[str] = None,
        max_steps: int = 3,
        max_tokens: int = 2048,
        train_temperature: float = 0.7,
        test_temperature: float = 0.0,
        output_type: str = "direct",
        timeout: int = 300,
    ):
        """
        Initialize Rollout handler.

        Args:
            server_public_ip: Public IP of the training server
            exp_name: Experiment name for organizing rollout data
            rollout_n: Number of rollouts per task
            batch_size: Training batch size
            enabled_tools: List of tools to enable
            tool_engine: Engine for each tool
            module_engine: Engine for each reasoning module [planner, executor, verifier, generator]
            max_steps: Maximum reasoning steps
            max_tokens: Maximum tokens for generation
            train_temperature: Temperature for training rollouts
            test_temperature: Temperature for validation rollouts
            output_type: Output types to generate
            timeout: Maximum execution time per task
        """
        super().__init__()
        self.server_public_ip = server_public_ip

        # Agents will be initialized on the first call to their respective rollouts.
        self.training_agent = None
        self.validation_agent = None
        self.val_step_n = None

        self.output_type = output_type
        self.timeout = timeout

        self.rollout_dir = None
        self.train_rollout_dir = None
        self.val_rollout_dir = None
        self.train_lock_file = None
        self.val_lock_file = None

        self.train_temperature = train_temperature
        self.test_temperature = test_temperature

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_rollout_dir = f"./rollout_data/{self.server_public_ip}/{exp_name}_{timestamp}"
        self.tools = enabled_tools
        self.tool_engine = tool_engine
        self.module_engine = module_engine if module_engine else ["Trainable", "Trainable", "Trainable", "Trainable"]
        self._solve_call_count = 0

        self.run_info_file = os.path.join(self.base_rollout_dir, ".run_info")
        self.init_lock_file = os.path.join(self.base_rollout_dir, ".init.lock")

        # Added locks and state variables for async-safe step management.
        self.train_batch_size = batch_size
        self.rollout_num = rollout_n
        self.max_steps = max_steps
        self.max_tokens = max_tokens

        # Atomic counter for step slot reservation to prevent race conditions
        # Key: step_n, Value: number of reserved slots (workers that will write to this step)
        self._step_reserved_slots: dict[int, int] = {}

    async def _solve_and_evaluate(
        self,
        rollout: ScientistRollout,
        task: Any,
        step_n: int,
        val: bool = False
    ):
        """
        A helper function to run the agent, parse the result, and evaluate it.

        Args:
            rollout: ScientistRollout instance
            task: Task dictionary containing question, result, etc.
            step_n: Current step number
            val: Whether this is a validation rollout
        """
        result = {}
        try:
            # Add output format instruction
            output_format = "When ready, output the final answer enclosed in <answer> and </answer> tags. Do not generate any content after the </answer> tag."
            prompt = task["question"] + " " + output_format

            result = rollout.solve(question=prompt)

            # Safely check for and extract the final answer
            if "direct_output" in result and result["direct_output"]:
                final_output = result["direct_output"]
                # Try to extract answer from tags
                all_matches = re.findall(r"<answer>(.*?)</answer>", final_output, re.DOTALL)
                if all_matches:
                    answer = all_matches[-1].strip()
                else:
                    answer = final_output
            else:
                print("Warning: Result has no direct_output or direct_output is empty.")
                answer = "None"
        except Exception as e:
            print(f"Failure during agent execution: {str(e)}. Defaulting to 'None'.")
            answer = "None"

        # Evaluate the answer against the ground truth
        reward_value = await eval(task["question"], str(task["result"]), answer, val)
        print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward_value))

        idx = task.get("extra_info", {}).get("idx", "unknown_idx")

        # Extract tools with their corresponding engines from the solver's toolbox_metadata
        available_tools = []
        if hasattr(rollout.solver, 'toolbox_metadata'):
            for tool_name, tool_meta in rollout.solver.toolbox_metadata.items():
                available_tools.append({
                    "name": tool_name,
                    "engine": tool_meta.get("engine", "Unknown")
                })
        else:
            # Fallback: use self.tools if toolbox_metadata is not available
            for tool_name in self.tools:
                available_tools.append({
                    "name": tool_name,
                    "engine": "Unknown"
                })

        # Extract Python_Coder_Tool generated code if available
        python_generated_codes = []
        if "agent_result" in result and "outputs" in result["agent_result"]:
            step_history = result["agent_result"]["outputs"].get("step_history", {})
            for step_num, step_data in step_history.items():
                if "Executor" in step_data:
                    execution_results = step_data["Executor"].get("execution_results", [])
                    for exec_result in execution_results:
                        if "execution_code" in exec_result:
                            python_generated_codes.append({
                                "step": step_num,
                                "code": exec_result["execution_code"],
                                "printed_output": exec_result.get("printed_output", ""),
                                "variables": exec_result.get("variables", {})
                            })

        rollout_data = {
            "step": task.get("step", ""),
            "idx": idx,
            "id": task.get("id", ""),
            "prompt": task["question"],
            "model": rollout.llm_engine,
            "available_tools": available_tools,
            "python_generated_codes": python_generated_codes,
            "groundtruth": task.get("extra_info", {}).get("groundtruth", task["result"]),
            "answer_extracted": answer,
            "reward": reward_value,
            "total_result": result,
            "timestamp": datetime.now().isoformat(),
        }

        data_id = str(uuid.uuid4())
        filename = f"rollout_{data_id}.json"

        save_dir = self.val_rollout_dir if val else self.train_rollout_dir

        # This function now uses the `step_n` passed as an argument.
        step_dir = os.path.join(save_dir, f"step_{step_n}")

        idx_dir = os.path.join(step_dir, f"idx_{idx}")
        os.makedirs(idx_dir, exist_ok=True)

        # Check rollout count for this idx (with graceful overflow handling)
        json_count = sum(
            len([f for f in files if f.endswith(".json")])
            for root, dirs, files in os.walk(idx_dir)
        )
        if json_count >= self.rollout_num:
            print(f"Warning: Skipping save for idx {idx} in step {step_n} - already has {json_count} >= {self.rollout_num} rollouts")
            # Clean up and return early
            del rollout_data
            del result
            import gc
            gc.collect()
            return

        save_path = os.path.join(idx_dir, filename)

        with open(save_path, "w") as f:
            json.dump(rollout_data, f, indent=2)

        # ✅ CRITICAL FIX: Explicitly free rollout_data and result to prevent memory leak
        # rollout_data contains the full agent_result with step_history
        # With 256 concurrent rollouts, this can accumulate hundreds of GB
        # Each result can be 8MB (JSON) but contains Python objects that are much larger
        del rollout_data
        del result

        # Force garbage collection to immediately free memory
        import gc
        gc.collect()

        print(f"Rollout data saved and memory freed: {save_path}")


    async def _initialize_run_once(self, resources: NamedResources):
        """
        Ensures that the rollout directory is set up only once per run,
        in a process-safe way.
        """
        if self.rollout_dir is not None:
            return

        os.makedirs(self.base_rollout_dir, exist_ok=True)

        init_lock = FileLock(self.init_lock_file, timeout=50)
        with init_lock:
            if os.path.exists(self.run_info_file):
                with open(self.run_info_file, 'r') as f:
                    final_rollout_dir = f.read().strip()
            else:
                model_name = resources.get("main_llm").model
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                model_name = model_name.rsplit('/', 1)[-1]
                final_rollout_dir = os.path.join(
                    self.base_rollout_dir, f"{model_name}_{timestamp}"
                )

                with open(self.run_info_file, 'w') as f:
                    f.write(final_rollout_dir)
                print(f"Run directory created by process {os.getpid()}: {final_rollout_dir}")

        self.rollout_dir = final_rollout_dir
        self.train_rollout_dir = os.path.join(self.rollout_dir, "train")
        self.val_rollout_dir = os.path.join(self.rollout_dir, "validation")

        os.makedirs(self.train_rollout_dir, exist_ok=True)
        os.makedirs(self.val_rollout_dir, exist_ok=True)

        self.train_lock_file = os.path.join(self.train_rollout_dir, ".train.lock")
        self.val_lock_file = os.path.join(self.val_rollout_dir, ".val.lock")

    async def training_rollout_async(
        self,
        task: Any,
        rollout_id: str,
        resources: NamedResources,
        val: bool = False
    ) -> Any:
        """
        Execute a training rollout.

        Args:
            task: Task to execute
            rollout_id: Unique rollout identifier
            resources: Named resources from trainer
            val: Whether this is a validation rollout (should be False for training)
        """
        await self._initialize_run_once(resources)

        if self.training_agent is None:
            print("Initializing Scientist training agent...")
            llm: LLM = resources.get("main_llm")
            self.training_agent = get_agent(
                llm.model,
                llm.endpoint,
                temperature=self.train_temperature,
                tools=self.tools,
                max_steps=self.max_steps,
                tool_engine=self.tool_engine,
                module_engine=self.module_engine,
                resources=resources,
                max_tokens=self.max_tokens,
                output_type=self.output_type,
                timeout=self.timeout,
            )

        # filelock to determine step_n with atomic slot reservation ---
        # This prevents race conditions where multiple workers get the same step_n
        # before any of them saves their file.
        lock = FileLock(self.train_lock_file, timeout=30)
        with lock:
            step_dirs = [d for d in os.listdir(self.train_rollout_dir) if d.startswith("step_")]
            step_nums = [int(d.replace("step_", "")) for d in step_dirs if d.replace("step_", "").isdigit()]

            current_step_n = 1
            if step_nums:
                current_step_n = max(step_nums)

            current_step_dir = os.path.join(self.train_rollout_dir, f"step_{current_step_n}")
            expected_rollouts_per_step = self.train_batch_size * self.rollout_num

            if os.path.exists(current_step_dir):
                # Count actual saved JSON files
                json_count = sum(
                    len([f for f in files if f.endswith(".json")])
                    for root, dirs, files in os.walk(current_step_dir)
                )
                # Also count reserved slots (workers that are about to write but haven't yet)
                reserved_count = self._step_reserved_slots.get(current_step_n, 0)
                total_count = json_count + reserved_count

                if total_count >= expected_rollouts_per_step:
                    current_step_n += 1
                    # Create new step directory immediately
                    new_step_dir = os.path.join(self.train_rollout_dir, f"step_{current_step_n}")
                    os.makedirs(new_step_dir, exist_ok=True)
                    # Reset reserved slots for new step
                    self._step_reserved_slots[current_step_n] = 0

            # Reserve a slot for this worker BEFORE releasing the lock
            self._step_reserved_slots[current_step_n] = self._step_reserved_slots.get(current_step_n, 0) + 1
            step_n = current_step_n

        try:
            await self._solve_and_evaluate(self.training_agent, task, step_n, val)
        finally:
            # Release the reserved slot after saving (success or failure)
            with lock:
                if step_n in self._step_reserved_slots and self._step_reserved_slots[step_n] > 0:
                    self._step_reserved_slots[step_n] -= 1


    async def validation_rollout_async(
        self,
        task: Any,
        rollout_id: str,
        resources: NamedResources,
        val: bool = True
    ) -> Any:
        """
        Execute a validation rollout.

        Args:
            task: Task to execute
            rollout_id: Unique rollout identifier
            resources: Named resources from trainer
            val: Whether this is a validation rollout (should be True)
        """
        await self._initialize_run_once(resources)

        # Lazy initialization of the agent and one-time determination of the validation step number.
        # This lock ensures that only the first validation task of a run calculates the step number,
        # preventing the creation of thousands of folders.
        val_lock = FileLock(self.val_lock_file, timeout=50)
        with val_lock:
            if self.validation_agent is None:
                print("Initializing Scientist validation agent and determining validation step...")
                llm: LLM = resources.get("main_llm")
                self.validation_agent = get_agent(
                    llm.model,
                    llm.endpoint,
                    temperature=self.test_temperature,
                    tools=self.tools,
                    max_steps=self.max_steps,
                    tool_engine=self.tool_engine,
                    module_engine=self.module_engine,
                    resources=resources,
                    max_tokens=self.max_tokens,
                    output_type=self.output_type,
                    timeout=self.timeout,
                )

            print(f"Scanning '{self.train_rollout_dir}' to find current training step...")
            train_step_dirs = [d for d in os.listdir(self.train_rollout_dir) if d.startswith("step_")]
            train_step_nums = [int(d.replace("step_", "")) for d in train_step_dirs if d.replace("step_", "").isdigit()]

            current_train_step = max(train_step_nums) if train_step_nums else 0
            self.val_step_n = current_train_step
            print(f"Validation run started. Synchronizing with training progress. Saving results to validation step folder: {self.val_step_n}")

        await self._solve_and_evaluate(self.validation_agent, task, self.val_step_n, val)


if __name__ == "__main__":
    from util.parse_config import get_values_from_yaml
    from util.port_cleanup import kill_process_on_port
    from util.get_pub_ip import get_public_ip_with_fallback
    from pprint import pprint

    server_public_ip = get_public_ip_with_fallback()

    keys_to_retrieve = [
        "EXPERIMENT_NAME",
        'data.train_batch_size',
        'actor_rollout_ref.rollout.n',
        'agentflow.port',
        'N_WORKERS',
        'ENABLE_TOOLS',
        'TOOL_ENGINE',
        'MODULE_ENGINE',
        "TOOL_STEPS",
        "TRAIN_TEMPERATURE",
        "TEST_TEMPERATURE",
        "data.max_response_length",
        "OUTPUT_TYPE",
        "AGENT_MAX_TIMEOUT"
    ]

    config_file = 'trainer/train_scientist/config.yaml'

    values = get_values_from_yaml(config_file, keys_to_retrieve)

    config_keys_map = {
        "EXPERIMENT_NAME": "exp_name",
        "data.train_batch_size": "batch_size",
        "actor_rollout_ref.rollout.n": "rollout_n",
        "agentflow.port": "port",
        "N_WORKERS": "n_workers",
        "ENABLE_TOOLS": "enabled_tools",
        "TOOL_ENGINE": "tool_engine",
        "MODULE_ENGINE": "module_engine",
        "TOOL_STEPS": "max_steps",
        "TRAIN_TEMPERATURE": "train_temperature",
        "TEST_TEMPERATURE": "test_temperature",
        "data.max_response_length": "max_tokens",
        "OUTPUT_TYPE": "output_type",
        "AGENT_MAX_TIMEOUT": "timeout",
    }

    config_dict = dict(zip(config_keys_map.values(), values))

    port_to_use = config_dict.get("port")
    if port_to_use:
        print(f"[INFO] Checking and freeing port {port_to_use}...")
        kill_process_on_port(port_to_use)
    else:
        print("[WARNING] No port specified in config, skipping port cleanup.")

    print("Scientist Agent params:")
    pprint(config_dict, indent=2, width=80, compact=True)

    trainer = Trainer(n_workers=config_dict["n_workers"])
    agent = Rollout(
        server_public_ip=server_public_ip,
        **{k: v for k, v in config_dict.items() if k != "n_workers" and k != "port"}
    )
    trainer.fit(agent, f"http://localhost:{config_dict['port']}/")
