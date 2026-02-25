import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback

from scientist.engine.factory import create_llm_engine
from scientist.base_agent.memory import Memory
from scientist.base_agent.planner import Planner
from scientist.base_agent.executor import Executor
from scientist.base_agent.verifier import Verifier
from scientist.base_agent.generator import Generator
from scientist.utils.tool import get_all_tools_metadata
from scientist.utils.logger import setup_logger

from dotenv import load_dotenv
load_dotenv()

class Agent:
    def __init__(
        self,
        role_prompt: str,
        suggested_tools: List[str],
        llm_planner: Any,
        llm_executor: Any,
        llm_verifier: Any,
        llm_generator: Any = None,
        prompts_path: str = None,
        workspace_path: str = None,
        tool_engine: List[str] = None,
        is_validation: bool = False,
    ):
        # Initialize attributes
        self.role_prompt = role_prompt
        self.suggested_tools = suggested_tools
        self.tool_engine = tool_engine
        self.llm_planner = llm_planner
        self.llm_executor = llm_executor
        self.llm_verifier = llm_verifier
        self.llm_generator = llm_generator or llm_verifier
        self.prompts_path = prompts_path
        self.workspace_path = workspace_path # workspace path for running tools
        self.is_validation = is_validation # if is validation, tool usage name will use soft match

        # Initialize components (logger will be set after setup)
        self.memory = Memory()
        self.planner = Planner(llm=self.llm_planner, role_prompt=self.role_prompt, prompts_path=self.prompts_path, logger=None, is_validation=self.is_validation)
        self.executor = Executor(llm=self.llm_executor, role_prompt=self.role_prompt, prompts_path=self.prompts_path, logger=None)
        self.verifier = Verifier(llm=self.llm_verifier, role_prompt=self.role_prompt, prompts_path=self.prompts_path, logger=None)
        self.generator = Generator(llm=self.llm_generator, role_prompt=self.role_prompt, prompts_path=self.prompts_path, logger=None)

        # Get tool metadata using unified function from utils (with tool_engine support)
        self.toolbox_metadata = get_all_tools_metadata(self.suggested_tools, self.llm_executor, self.tool_engine)
        self.available_tools = list(self.toolbox_metadata.keys())
        
        # Set available_tools for planner tool normalization
        self.planner.available_tools = self.available_tools
        
        # Initialize unified logger
        self.logger = None # Unified Logger object for all logging

    def _setup_logger(self, logging_path: str):
        """Setup structured logger using utils function."""
        # Only setup logger if not already initialized or if logging_path changes
        if self.logger is None or (logging_path and logging_path != getattr(self, '_last_logging_path', None)):
            self.logger = setup_logger(f"Agent_{id(self)}", logging_path)
            self._last_logging_path = logging_path
            # Update components with the new logger
            self._update_components_logger()
    
    def cleanup(self):
        """Clean up agent resources to free memory."""
        # Close logger handlers to release file descriptors
        if self.logger and hasattr(self.logger, 'logger') and self.logger.logger:
            for handler in self.logger.logger.handlers:
                handler.close()
        
        # Clear memory to free up space
        if hasattr(self, 'memory') and self.memory:
            self.memory.clear()
        
        # Clear component references
        self.planner = None
        self.executor = None
        self.verifier = None
        self.generator = None
                
    def _update_components_logger(self):
        """Update components with the current logger."""
        if hasattr(self, 'planner'):
            self.planner.logger = self.logger
        if hasattr(self, 'executor'):
            self.executor.logger = self.logger
        if hasattr(self, 'verifier'):
            self.verifier.logger = self.logger
        if hasattr(self, 'generator'):
            self.generator.logger = self.logger
    
    def cleanup(self):
        """Cleanup resources."""
        if self.logger and self.logger.logger:
            for handler in self.logger.logger.handlers[:]:
                handler.close()
                self.logger.logger.removeHandler(handler)

    
    def _collect_llm_engines_info(self) -> Dict[str, Any]:
        """Collect LLM engine metadata from all components."""
        llm_engines_info = {}
        for component_name, component in [("planner", self.planner), ("executor", self.executor), ("verifier", self.verifier), ("generator", self.generator)]:
            if hasattr(component, 'llm') and component.llm:
                llm_engines_info[component_name] = {
                    "engine_type": type(component.llm).__name__,
                    "model": getattr(component.llm, 'model_string', 'unknown')
                }
        return llm_engines_info

    def _parse_output_types(self, output_types: str) -> List[str]:
        """Parse output_types string into a list of output types."""
        # Parse and validate output_types (support comma-separated values)
        valid_output_types = {"final", "direct", "schema"}
        if isinstance(output_types, str):
            output_types_list = [t.strip() for t in output_types.split(',')]
        else:
            output_types_list = [output_types] if output_types else ["direct"]
        
        # Validate each output type
        for ot in output_types_list:
            if ot not in valid_output_types:
                raise ValueError(f"Invalid output_types '{ot}'. Must be one of: {valid_output_types}")
        
        return output_types_list

    def _build_result_dict(self, final_output: Dict[str, Any], final_output_time: float, 
                          final_status: str, total_time: float, step_count: int) -> Dict[str, Any]:
        """Build the final result dictionary with configs, inputs and outputs."""
        return {
            "configs": {
                "role_prompt": self.role_prompt,
                "suggested_tools": self.suggested_tools,
                "workspace_path": self.workspace_path,
                "llm_engines": self._collect_llm_engines_info()
            },
            "inputs": {
                "agent_query": self.agent_query,
                "query_instruction": self.query_instruction,
                "input_files": self.input_files,
                "output_schema": self.output_schema,
                "output_types": self.output_types,
                "max_steps": self.max_steps,
                "max_time": self.max_time,
                "logging_path": self.logging_path,
                "output_path": self.output_path,
                "available_tools": self.available_tools,
                "toolbox_metadata_summary": {
                    "total_tools": len(self.toolbox_metadata),
                    "tool_names": list(self.toolbox_metadata.keys())
                }
            },
            "outputs": {
                "global_plan": self.memory.get_global_plan(),
                "step_history": self.memory.step_history,
                "final_output": final_output,
                "final_output_time": round(final_output_time, 2),
                "final_status": final_status,
                "total_execution_time": round(total_time, 2),
                "execution_metadata": {
                    "steps_completed": step_count + (1 if final_status == 'success' else 0),
                    "max_steps_reached": step_count == self.max_steps,
                    "timeout_occurred": final_status == 'timeout',
                    "has_execution_errors": any(
                        not result.get("success", True) 
                        for step_data in self.memory.step_history.values() 
                        for result in step_data.get("Executor", {}).get("execution_results", [])
                    ),
                    "total_errors": sum(
                        1 for step_data in self.memory.step_history.values() 
                        for result in step_data.get("Executor", {}).get("execution_results", [])
                        if not result.get("success", True)
                    ),
                    "success_rate": len([s for s in self.memory.step_history.values() if s.get("Executor", {}).get("execution_results", [{}])[0].get("success", False)]) / max(len(self.memory.step_history), 1)
                }
            }
        }
    
    def _save_output(self, result: Dict[str, Any], output_path: str):
        """Save output to JSON file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            self.logger.log(f"==> Output saved to {output_path}")
        except Exception as e:
            self.logger.log(f"==> Failed to save output: {str(e)}")
    
    def run(self, 
        agent_query: str = None,
        query_instruction: str = None, 
        input_files: List[Dict[str, Any]] = [], 
        output_schema: Dict[str, Any] = {}, 
        output_types: str = "direct",
        max_steps: int = 3,
        max_time: int = 300, # seconds
        logging_path: str = None,
        output_path: str = None,
        **kwargs) -> Dict[str, Any]:
        
        start_time = time.time()
        
        try:
            # Setup logging (console logging always enabled)
            self._setup_logger(logging_path)
            self.logger.log_header("Agent Initialization", separator="#")

            # CRITICAL: Reset Memory for each new run to prevent memory leaks
            # When Agent instances are reused across multiple rollouts (as in training),
            # the Memory object accumulates step history indefinitely, causing OOM
            self.memory = Memory()
            self.logger.log("==> Memory reset for new rollout to prevent accumulation")

            # Collect tool engine information
            tools_with_engines = []
            for tool_name in self.available_tools:
                engine_info = self.toolbox_metadata.get(tool_name, {}).get("engine", "Unknown")
                tools_with_engines.append(f"{tool_name} ({engine_info})")

            self.logger.log_section(
                "Agent Initialization Args",
                {
                    "Role Prompt": self.role_prompt,
                    "Suggested Tools": self.suggested_tools,
                    "Available Tools": self.available_tools,
                    "Tools with Engines": tools_with_engines,
                    "Planner LLM": (self.llm_planner, self.llm_planner.model_string if hasattr(self.llm_planner, 'model_string') else "N/A"),
                    "Executor LLM": (self.llm_executor, self.llm_executor.model_string if hasattr(self.llm_executor, 'model_string') else "N/A"),
                    "Verifier LLM": (self.llm_verifier, self.llm_verifier.model_string if hasattr(self.llm_verifier, 'model_string') else "N/A"),
                    "Prompts Path": self.prompts_path,
                    "Workspace Path": self.workspace_path,
                }
            )

            # Parse inputs
            self.agent_query = agent_query
            self.query_instruction = query_instruction
            self.input_files = {f: f for f in input_files} if input_files else {}
            self.output_schema = output_schema
            self.output_types = output_types  # Keep original string for logging
            self.max_steps = max_steps
            self.max_time = max_time
            self.logging_path = logging_path
            self.output_path = output_path

            # Log detailed initialization information
            self.logger.log_section(
                "Query Information",
                {
                    "query": self.agent_query,
                    "query_instruction": self.query_instruction,
                    "input_files": self.input_files,
                    "output_schema": self.output_schema,
                    "output_types": self.output_types,
                    "max_steps": self.max_steps,
                    "max_time": self.max_time,
                    "logging_path": self.logging_path,
                    "output_path": self.output_path
                }
            )

            # Handle output_types priority - user's explicit choice takes precedence
            self.output_types_list = self._parse_output_types(output_types)
            self.logger.log(f"Output types: {self.output_types_list}")
            
            # Only log a warning if schema is provided but no schema type is requested
            if output_schema and "schema" not in self.output_types_list:
                self.logger.warning(f"Schema provided but 'schema' not in output_types '{output_types}'. Schema will be ignored.")
            
            # Reconstruct the agent query
            self.agent_query = f"**User Request:**\n**Query:** {self.agent_query}"
            if self.query_instruction:
                self.agent_query += f"\n**Instructions:** {self.query_instruction}"

            self.logger.log_section(
                "Final Query",
                {
                    "agent_query": "\n" + self.agent_query if isinstance(self.agent_query, str) else self.agent_query
                }
            )

            # [0] Analyze query and generate a global plan
            self.logger.log_header("Global Plan Generation", separator="#")
            self.logger.log("Starting global plan generation...")

            query_start_time = time.time()
            global_plan_start_time = time.time()
            try:
                # Generate global plan
                global_plan = self.planner.generate_global_plan(
                    self.agent_query, 
                    self.input_files, 
                    self.available_tools, 
                    self.toolbox_metadata
                )
                global_plan_time = time.time() - global_plan_start_time
                # Log detailed global plan
                self.logger.log_section(
                    "Global Plan Generation",
                    {
                        "Execution time": f"{global_plan_time:.2f} seconds" if global_plan_time else "N/A",
                        "==> Generated global plan": "\n" + global_plan if isinstance(global_plan, str) else global_plan,
                    },
                )
                # Update memory
                self.memory.add_global_plan(global_plan, global_plan_time)
            except Exception as e:
                self.logger.error("Exception occurred during global plan generation:")
                self.logger.error(traceback.format_exc())
                raise e

            # Main Agent Loop
            step_count = 0
            final_status = None
            self.logger.log_header("Main Agent Loop", separator="#")
            try:
                while step_count < self.max_steps:
                    step_count += 1

                    # [1] Generate next action for the current query
                    self.logger.log_header(f"[Step {step_count}] [Planner]", width=70)
                    plan_start_time = time.time()
                    try:
                        # Generate one step plan
                        current_plan = self.planner.generate_one_step_plan(
                            self.agent_query, 
                            self.input_files,
                            self.available_tools, 
                            self.toolbox_metadata,
                            global_plan,
                            self.memory,
                            step_count, 
                            self.max_steps
                        )
                        plan_time = time.time() - plan_start_time
                        # Log detailed one step plan
                        self.logger.log_section(
                            f"[Step {step_count}] [Planner] Generated Step Plan",
                            {
                                "Execution time": f"{plan_time:.2f} seconds" if plan_time else "N/A",
                                "==> Step plan": "\n" + current_plan if isinstance(current_plan, str) else current_plan
                            }
                        )
                        # Parse one step plan
                        tool_to_use, step_goal, step_context = self.planner.parse_next_step(current_plan)
                        # Log parsed one step plan
                        self.logger.log_section(
                            f"[Step {step_count}] [Planner] Parsed Step Plan",
                            {
                                "==> Tool to use": tool_to_use,
                                "==> Step goal": step_goal,
                                "==> Step context": step_context
                            }
                        )
                        # Update memory
                        self.memory.add_step_plan(step_count, tool_to_use, step_goal, step_context, plan_time)
                    except Exception as e:
                        self.logger.error(f"Exception during step {step_count} planning:")
                        self.logger.error(traceback.format_exc())
                        raise e
                    
                    # [2] Executor: execute the next action by tool calling
                    execution_start_time = time.time()
                    try:
                        # Generate commands
                        self.logger.log_header(f"[Step {step_count}] [Executor] Generating Commands", width=70)
                        command_generation_start_time = time.time()
                        analysis, commands = self.executor.generate_commands(
                            tool_to_use, step_goal, step_context
                        )
                        command_generation_time = time.time() - command_generation_start_time
                        self.logger.log_section(
                            f"[Step {step_count}] [Executor] Generated Commands",
                            {
                                "Execution time": f"{command_generation_time:.2f} seconds" if command_generation_time else "N/A",
                                "==> Analysis": "\n" + analysis if isinstance(analysis, str) else analysis,
                                "==> Commands": commands
                            }
                        )

                        # Execute commands
                        self.logger.log_header(f"[Step {step_count}] [Executor] Executing Commands", width=70)
                        command_execution_start_time = time.time()
                        execution_results = self.executor.execute_commands(
                            tool_to_use, commands, self.workspace_path
                        )
                        command_execution_time = time.time() - command_execution_start_time
                        self.logger.log_section(
                            f"[Step {step_count}] [Executor] Executed Commands",
                            {
                                "Execution time": f"{command_execution_time:.2f} seconds" if command_execution_time else "N/A",
                                "==> Execution Results": execution_results
                            }
                        )
                        # Update memory
                        total_execution_time = time.time() - execution_start_time
                        self.memory.add_step_execution(step_count, analysis, execution_results, command_generation_time, command_execution_time)
                        
                    except Exception as e:
                        self.logger.error(f"Exception during step {step_count} execution:")
                        self.logger.error(traceback.format_exc())
                        raise e
                    
                    # [3] Verifier: reflect and decide whether to continue
                    self.logger.log_header(f"[Step {step_count}] [Verifier]", width=70)
                    reflection_start_time = time.time()
                    try:
                        # Generate reflection
                        reflection_result = self.verifier.reflect_and_decide(
                            self.agent_query, 
                            self.input_files, 
                            self.available_tools, 
                            self.toolbox_metadata, 
                            global_plan,
                            self.memory,
                        )
                        reflection_time = time.time() - reflection_start_time
                        # Log detailed reflection and verification
                        self.logger.log_section(
                            f"[Step {step_count}] [Verifier] Generated Reflection",
                            {
                                "Execution time": f"{reflection_time:.2f} seconds" if reflection_time else "N/A",
                                "==> Reflection Result": reflection_result
                            }
                        )
                        # Parse reflection result
                        analysis, continue_signal = self.verifier.parse_reflection(reflection_result)
                        # Log detailed reflection and verification
                        self.logger.log_section(
                            f"[Step {step_count}] [Verifier] Parsed Reflection",
                            {
                                "==> Reflection Analysis": analysis,
                                "==> Decision": 'CONTINUE' if continue_signal else 'STOP'
                            }
                        )
                        # Update memory
                        self.memory.add_step_reflection(step_count, not continue_signal, analysis, reflection_time)
                    except Exception as e:
                        self.logger.error(f"Exception during step {step_count} verification:")
                        self.logger.error(traceback.format_exc())
                        raise e
                
                    # Check if the agent should continue
                    if not continue_signal: 
                        final_status = 'success'
                        self.logger.log("==> Successfully completed the agent loop. ✅")
                        break
                    
                    if time.time() - query_start_time > self.max_time:
                        final_status = 'timeout'
                        self.logger.log("==> Execution timeout reached. ⚠️")
                        break
                    
                    # Check if the agent should stop
                    if step_count == self.max_steps:
                        final_status = 'max_steps_reached'
                        self.logger.log("==> Maximum steps reached. ⚠️")
                        
            except Exception as e:
                # Log any error in the main loop
                self.logger.error(f"Exception in main agent loop at step {step_count}:")
                self.logger.error(traceback.format_exc())
                final_status = 'error'
                # Optionally, you can re-raise or just return error result
                # raise
            
            # [4] Final Output - Generate multiple types if requested
            self.logger.log_header("Final Output Generation", separator="#")
            self.logger.log(f"Starting final output generation (types: {self.output_types})...")

            # Prepare memory for final output generation
            global_plan = self.memory.get_global_plan()

            # Note: Skip Verifier module for final output generation
            concise_steps_for_output = self.memory.get_concise_steps(modules_to_include=["Planner", "Executor"])
            self.logger.log_section(
                "[Memory] Concise Steps (Used for Final Output Generation)",
                {
                    "==> Concise Steps": concise_steps_for_output
                }
            )

            final_output_start_time = time.time()
            try:
                final_output = {}
                
                for output_type in self.output_types_list:
                    if output_type == "direct":
                        self.logger.log("Generating direct output...")
                        # Generate direct output
                        direct_response = self.generator.generate_direct_output(
                            self.agent_query, 
                            self.input_files,
                            global_plan,
                            concise_steps_for_output,
                        )
                        final_output["direct"] = {"response": direct_response}
                        
                    elif output_type == "final":
                        self.logger.log("Generating final output...")
                        # Generate final output without schema
                        final_response = self.generator.generate_final_output(
                            self.agent_query, 
                            self.input_files,
                            global_plan,
                            concise_steps_for_output,
                        )
                        final_output["final"] = {"response": final_response}

                    elif output_type == "schema":
                        self.logger.log("Generating schema-based output...")
                        # Generate schema-based output
                        schema_response = self.generator.generate_schema_output(
                            self.agent_query, 
                            self.input_files,
                            global_plan,
                            concise_steps_for_output,
                            self.output_schema,
                        )
                        final_output["schema"] = schema_response
                        
                    else:
                        # This should not happen due to validation above
                        error_msg = f"Unsupported output_types: {output_type}. Must be one of: {valid_output_types}"
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                
                final_output_time = time.time() - final_output_start_time

                # Log detailed final output generation
                self.logger.log_section(
                    "Final Output Generation",
                    {
                        "==> Generation time": f"{final_output_time:.2f} seconds" if final_output_time else "N/A",
                        "==> Final Output": final_output
                    }
                )
            except Exception as e:
                self.logger.error("Exception during final output generation:")
                self.logger.error(traceback.format_exc())
                final_output = {"error": str(e)}
                final_output_time = 0.0  # Default if error occurred
                
            # Build result dictionary
            total_time = time.time() - query_start_time
            result = self._build_result_dict(final_output, final_output_time, final_status, total_time, step_count)
            
            # Log execution summary
            self.logger.log_section(
                "Execution Summary",
                {   
                    "Steps Completed": step_count + (1 if final_status == 'success' else 0),
                    "Final Status": final_status,
                    "Total Execution Time": f"{total_time:.2f} seconds" if total_time else "N/A"
                }
            )
            
            # Save output if path provided
            if self.output_path:
                self._save_output(result, self.output_path)

            self.logger.log(f"==> Done! ✅")
            return result

        finally:
            if hasattr(self, 'cleanup'):
                self.cleanup()
            execution_time = time.time() - start_time
