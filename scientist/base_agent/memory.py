from typing import List, Dict, Any

class Memory: 
    def __init__(self):
        self.global_plan = {}
        self.step_history = {}

    def clear(self):
        """Clear all memory to free up space."""
        self.global_plan = {}
        self.step_history = {}

    def add_global_plan(self, global_plan, plan_time: float = 0.0):
        self.global_plan = {"Global Plan": global_plan, "execution_time": round(plan_time, 2)}

    def add_step_plan(self, step_count: int, tool_to_use: str, step_goal: str, step_context: str, plan_time: float):
        # Initialize step_history if not exists
        current_step = f"Step {step_count}"
        if current_step not in self.step_history:
            self.step_history[current_step] = {}

        # Add planner information
        self.step_history[current_step][f"Planner"] = {
            "tool_to_use": tool_to_use,
            "step_goal": step_goal,
            "step_context": step_context,
            "execution_time": round(plan_time, 2)
        }
 
    def add_step_execution(self, step_count: int, analysis: str, execution_results: Dict[str, Any], command_generation_time: float, command_execution_time: float):
        # Initialize step_history if it does not exist
        current_step = f"Step {step_count}"
        if current_step not in self.step_history:
            self.step_history[current_step] = {}
        
        # Integrate generated commands into execution_result
        enhanced_results = []
        for i, result in enumerate(execution_results):
            # Create a copy to avoid modifying the original
            result_copy = dict(result)
            
            # Insert generated_command after tool_name, before result
            tool_name = result_copy.get("tool_name")
            if tool_name:
                # Rebuild the dict with proper ordering
                ordered_result = {"tool_name": tool_name}
                ordered_result["command_arguments"] = result_copy.get("command_arguments")
                # Add remaining fields
                for key, value in result_copy.items():
                    if key != "tool_name":
                        ordered_result[key] = value
                result_copy = ordered_result
            
            enhanced_results.append(result_copy)
        
        # Build concise execution info
        executor_info = {
            "generation_analysis": analysis,
            "generation_time": round(command_generation_time, 2),
            "execution_results": enhanced_results,
            "execution_time": round(command_execution_time, 2)
        }

        # Add executor information to step_history
        self.step_history[current_step]["Executor"] = executor_info
    
    def add_step_reflection(self, step_count: int, stop_signal: bool, analysis: str, reflection_time: float):
        # Initialize step_history if not exists
        current_step = f"Step {step_count}"
        if current_step not in self.step_history:
            self.step_history[current_step] = {}

        # Add reflection information
        self.step_history[current_step]["Verifier"] = {
            "analysis": analysis,
            "stop_signal": stop_signal,
            "execution_time": round(reflection_time, 2)
        }

    def get_global_plan(self):
        """
        Return global plan.
        """
        return self.global_plan

    def get_step_history(self):
        """
        Return step history in a detailed format (normally used for logging).
        """
        return self.step_history

    def get_concise_steps(self, modules_to_include: List[str] = ["Planner", "Executor", "Verifier"]):
        """
        Return step history in a concise format (normally used for planning and final output generation).
        Note: By default, include all modules in the step history.
        """
        concise_steps = {}
        for step_key, step_info in self.step_history.items():
            one_step = {}
            
            # Add Planner information if available
            if "Planner" in step_info and "Planner" in modules_to_include:
                one_step.update({
                    "Tool-to-use from Planner": step_info["Planner"]["tool_to_use"],
                    "Step Goal from Planner": step_info["Planner"]["step_goal"],
                    "Step Context from Planner": step_info["Planner"]["step_context"]
                })
        
            # Add Executor information if available
            if "Executor" in step_info and "Executor" in modules_to_include:
                execution_results = step_info["Executor"]["execution_results"] # NOTE
                # Remove all keys ("execution_time") in each entry of execution_result
                for entry in execution_results:
                    if "execution_time" in entry:
                        entry.pop("execution_time")
                one_step.update({
                    "Tool Execution from Executor": execution_results
                })

            # Add Verifier information if available
            if "Verifier" in step_info and "Verifier" in modules_to_include:
                one_step.update({
                    "Step History Analysis from Verifier": step_info["Verifier"]["analysis"],
                    "Stop Signal from Verifier": step_info["Verifier"]["stop_signal"]
                })
            
            # Use step_key directly (it's already in the correct format like "Step 1")
            concise_steps[step_key] = one_step

        return concise_steps
