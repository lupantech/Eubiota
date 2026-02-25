"""
StreamingAgent - Agent with real-time callback support for streaming updates

This class extends the base Agent to emit events during execution,
enabling real-time progress updates in the frontend.
"""

import os
import time
import json
import traceback
from typing import Dict, Any, List, Optional, Callable

from scientist.base_agent.agent import Agent
from scientist.base_agent.memory import Memory
from scientist.base_agent.planner import Planner
from scientist.base_agent.executor import Executor
from scientist.base_agent.verifier import Verifier
from scientist.base_agent.generator import Generator
from scientist.utils.tool import get_all_tools_metadata
from scientist.utils.logger import setup_logger


class StreamingAgent(Agent):
    """Agent with streaming callback support for real-time updates"""
    
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
        on_event: Callable[[Dict[str, Any]], None] = None,
        check_stop: Callable[[], bool] = None,
        check_finish_now: Callable[[], bool] = None
    ):
        # Initialize parent class
        super().__init__(
            role_prompt=role_prompt,
            suggested_tools=suggested_tools,
            llm_planner=llm_planner,
            llm_executor=llm_executor,
            llm_verifier=llm_verifier,
            llm_generator=llm_generator,
            prompts_path=prompts_path,
            workspace_path=workspace_path
        )
        
        # Callback for emitting events
        self.on_event = on_event or (lambda x: None)
        # Callback to check if stop was requested
        self.check_stop = check_stop or (lambda: False)
        # Callback to check if finish now was requested (skip to generator)
        self.check_finish_now = check_finish_now or (lambda: False)
    
    def _emit(self, event_type: str, phase: str = None, step: int = None, data: Any = None, message: str = None):
        """Emit an event to the callback"""
        event = {"type": event_type}
        if phase:
            event["phase"] = phase
        if step is not None:
            event["step"] = step
        if data is not None:
            event["data"] = data
        if message:
            event["message"] = message
        self.on_event(event)
    
    def run(
        self,
        agent_query: str = None,
        query_instruction: str = None,
        input_files: List[Dict[str, Any]] = [],
        output_schema: Dict[str, Any] = {},
        output_types: str = "direct",
        max_steps: int = 3,
        max_time: int = 300,
        logging_path: str = None,
        output_path: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run agent with real-time event emission"""
        
        # Setup logging
        self._setup_logger(logging_path)
        self.logger.log_header("Agent Initialization", separator="#")
        
        # Store parameters
        self.agent_query = agent_query
        self.query_instruction = query_instruction
        self.input_files = {f: f for f in input_files} if input_files else {}
        self.output_schema = output_schema
        self.output_types = output_types
        self.max_steps = max_steps
        self.max_time = max_time
        self.logging_path = logging_path
        self.output_path = output_path
        
        # Parse output types
        self.output_types_list = self._parse_output_types(output_types)
        
        # Reconstruct query
        self.agent_query = f"**User Request:**\n**Query:** {self.agent_query}"
        if self.query_instruction:
            self.agent_query += f"\n**Instructions:** {self.query_instruction}"
        
        # Check for stop before starting
        if self.check_stop():
            return self._build_stopped_result()
        
        # Generate global plan
        self._emit("phase", "global_plan", message="Generating global plan...")
        query_start_time = time.time()
        global_plan_start_time = time.time()
        
        global_plan = self.planner.generate_global_plan(
            self.agent_query,
            self.input_files,
            self.available_tools,
            self.toolbox_metadata
        )
        global_plan_time = time.time() - global_plan_start_time
        self.memory.add_global_plan(global_plan, global_plan_time)
        
        self._emit("phase", "global_plan", data={"plan": global_plan, "time": global_plan_time})
        
        # Main agent loop
        step_count = 0
        final_status = None
        
        while step_count < self.max_steps:
            # Check for stop signal
            if self.check_stop():
                final_status = "stopped"
                self._emit("stopped", message="Execution stopped by user")
                break
            
            step_count += 1
            
            # Planner phase
            self._emit("phase", "planner", step=step_count, message=f"Planning step {step_count}...")
            plan_start_time = time.time()
            
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
            
            tool_to_use, step_goal, step_context = self.planner.parse_next_step(current_plan)
            self.memory.add_step_plan(step_count, tool_to_use, step_goal, step_context, plan_time)
            
            # Emit planner data
            planner_data = {
                "tool_to_use": tool_to_use,
                "step_goal": step_goal,
                "step_context": step_context,
                "plan_time": plan_time
            }
            self._emit("phase", "planner", step=step_count, data=planner_data)
            
            # Check for stop
            if self.check_stop():
                final_status = "stopped"
                break
            
            # Executor phase
            self._emit("phase", "executor", step=step_count, message=f"Executing step {step_count}...")
            execution_start_time = time.time()
            
            command_generation_start = time.time()
            analysis, commands = self.executor.generate_commands(tool_to_use, step_goal, step_context)
            command_generation_time = time.time() - command_generation_start
            
            command_execution_start = time.time()
            execution_results = self.executor.execute_commands(tool_to_use, commands, self.workspace_path)
            command_execution_time = time.time() - command_execution_start
            
            self.memory.add_step_execution(step_count, analysis, execution_results, command_generation_time, command_execution_time)
            
            # Emit executor data
            executor_data = {
                "analysis": analysis,
                "execution_results": execution_results,
                "command_generation_time": command_generation_time,
                "command_execution_time": command_execution_time
            }
            self._emit("phase", "executor", step=step_count, data=executor_data)
            
            # Check for stop
            if self.check_stop():
                final_status = "stopped"
                break
            
            # Verifier phase
            self._emit("phase", "verifier", step=step_count, message=f"Verifying step {step_count}...")
            reflection_start_time = time.time()
            
            reflection_result = self.verifier.reflect_and_decide(
                self.agent_query,
                self.input_files,
                self.available_tools,
                self.toolbox_metadata,
                global_plan,
                self.memory,
            )
            reflection_time = time.time() - reflection_start_time
            
            analysis, continue_signal = self.verifier.parse_reflection(reflection_result)
            self.memory.add_step_reflection(step_count, not continue_signal, analysis, reflection_time)
            
            # Emit verifier data
            verifier_data = {
                "is_done": not continue_signal,
                "analysis": analysis,
                "reflection_time": reflection_time
            }
            self._emit("phase", "verifier", step=step_count, data=verifier_data)
            
            # Check completion conditions
            if not continue_signal:
                final_status = "success"
                break
            
            # Check if user requested immediate response generation
            if self.check_finish_now():
                final_status = "user_requested_finish"
                self._emit("phase", "finish_now", message="User requested immediate response, skipping to generator...")
                break
            
            if time.time() - query_start_time > self.max_time:
                final_status = "timeout"
                break
            
            if step_count == self.max_steps:
                final_status = "max_steps_reached"
        
        # Handle stopped case
        if final_status == "stopped":
            return self._build_stopped_result(step_count, query_start_time)
        
        # Generate final output
        self._emit("phase", "generator", message="Generating final output...")
        
        global_plan = self.memory.get_global_plan()
        concise_steps = self.memory.get_concise_steps(modules_to_include=["Planner", "Executor"])
        
        final_output_start = time.time()
        final_output = {}
        
        for output_type in self.output_types_list:
            if self.check_stop():
                return self._build_stopped_result(step_count, query_start_time)
            
            if output_type == "direct":
                direct_response = self.generator.generate_direct_output(
                    self.agent_query, self.input_files, global_plan, concise_steps
                )
                final_output["direct"] = {"response": direct_response}
            elif output_type == "final":
                final_response = self.generator.generate_final_output(
                    self.agent_query, self.input_files, global_plan, concise_steps
                )
                final_output["final"] = {"response": final_response}
            elif output_type == "schema":
                schema_response = self.generator.generate_schema_output(
                    self.agent_query, self.input_files, global_plan, concise_steps, self.output_schema
                )
                final_output["schema"] = schema_response
        
        final_output_time = time.time() - final_output_start
        total_time = time.time() - query_start_time
        
        result = self._build_result_dict(final_output, final_output_time, final_status, total_time, step_count)
        
        if self.output_path:
            self._save_output(result, self.output_path)
        
        return result
    
    def _build_stopped_result(self, step_count: int = 0, start_time: float = None) -> Dict[str, Any]:
        """Build result for stopped execution"""
        total_time = time.time() - start_time if start_time else 0
        return {
            "configs": {
                "role_prompt": self.role_prompt,
                "suggested_tools": self.suggested_tools,
                "workspace_path": self.workspace_path,
                "llm_engines": self._collect_llm_engines_info()
            },
            "inputs": {
                "agent_query": getattr(self, 'agent_query', ''),
                "output_schema": getattr(self, 'output_schema', {}),
                "max_steps": getattr(self, 'max_steps', 0),
                "max_time": getattr(self, 'max_time', 0)
            },
            "outputs": {
                "global_plan": self.memory.get_global_plan() if self.memory else "",
                "step_history": self.memory.step_history if self.memory else {},
                "final_output": {"stopped": {"message": "Execution stopped by user"}},
                "final_status": "stopped",
                "total_execution_time": round(total_time, 2)
            }
        }
