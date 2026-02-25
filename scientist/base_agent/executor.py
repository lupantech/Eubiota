import json
import time
import os
import re
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple

from scientist.utils.utils import load_yaml
from scientist.utils.tool import validate_tool_commands, get_tool_metadata

class Executor:
    def __init__(self, llm: Any, role_prompt: str, prompts_path: str = None, logger=None):
        self.llm = llm
        self.role_prompt = role_prompt
        self.logger = logger
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.prompts = self._load_prompts(prompts_path)

        # Some local tool execution args
        self.concurrent_workers = 4 # 4 workers for parallel command execution
        self.max_attempts = 3 # 3 attempts per command generation
        self.max_execution_time = 300 # 300 seconds total for all commands (increased for high load)
        self.execution_timeout = 60 # 60 seconds per individual command execution (increased for embeddings)
    
    def _load_prompts(self, prompts_path: str = None) -> Dict[str, str]:
        """Load executor prompts from YAML file."""
        if prompts_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, 'prompts', 'executor.yaml')
        
        return load_yaml(prompts_path)
    
    def _extract_commands_and_analysis_from_response(self, response_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract analysis and JSON commands from LLM response.
        
        Returns:
            Tuple of (analysis_text, commands_list)
        """
        # Extract Analysis section
        analysis_pattern = r"Analysis:\s*(.*?)(?=Generated Arguments:|$)"
        analysis_match = re.search(analysis_pattern, response_text, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided"
        
        # Find ```json position  
        start_marker = "```json"
        start_pos = response_text.find(start_marker)
        if start_pos == -1:
            self.logger.error(f"No ```json found in response")
            return analysis, []
        
        # Find last ``` position (more reliable than non-greedy regex)
        end_marker = "```"
        end_pos = response_text.rfind(end_marker)
        if end_pos == -1 or end_pos <= start_pos:
            self.logger.error(f"No closing ``` found")
            return analysis, []
        
        # Extract and parse JSON content
        json_content = response_text[start_pos + len(start_marker):end_pos].strip()
        
        try:
            parsed_json = json.loads(json_content)
            
            # Handle both single dict and list of dicts
            if isinstance(parsed_json, dict):
                return analysis, [parsed_json]
            elif isinstance(parsed_json, list):
                return analysis, [item for item in parsed_json if isinstance(item, dict)]
            else:
                self.logger.error(f"Expected dict or list, got {type(parsed_json)}")
                return analysis, []
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {e}")
            return analysis, []

    def _generate_tool_commands(self, tool_name: str, step_goal: str, step_context: str, 
                               tool_info: str, tool_metadata: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate tool command arguments with retry mechanism."""
        prompt_template = self.prompts.get('command_generation', '')
        prompt = prompt_template.format(
            step_goal=step_goal,
            step_context=step_context,
            tool_to_use=tool_name,
            tool_info=tool_info
        )

        # Get required parameters
        input_kwargs = tool_metadata.get('input_kwargs', {})
        required_params = [param for param, config in input_kwargs.items() 
                          if not config.get('optional', False)]
        
        # Try to generate valid commands (max self.max_attempts attempts)
        previous_outputs = [] # List of command arguments outputs from previous attempts
        previous_errors = [] # List of errors from previous attempts
        for attempt in range(self.max_attempts):
            self.logger.info(f"Command generation attempt {attempt + 1}/{self.max_attempts}")
            
            # Build retry instructions with feedback
            current_prompt = prompt
            if attempt > 0:
                current_prompt += f"\n\nRETRY ATTEMPT {attempt + 1}:\n"
                current_prompt += f"Required parameters: {required_params}\n"
                
                if previous_outputs:
                    current_prompt += f"\nPrevious failed command arguments output:\n{previous_outputs[-1]}\n"
                
                if previous_errors:
                    current_prompt += f"\nError to fix:\n{previous_errors[-1]}"
            
            # Generate response
            response = self.llm.generate(
                content=current_prompt,
                system_prompt="You are an expert tool command generator. Output valid JSON in ```json``` blocks."
            )
            
            response_text = response.choices[0].message.content if hasattr(response, "choices") else str(response)
            previous_outputs.append(response_text)
            
            # Log full response for debugging
            self.logger.info(f"=== LLM RESPONSE (attempt {attempt + 1}) ===")
            self.logger.info(f"Response type: {type(response_text)}")
            self.logger.info(f"Response length: {len(response_text) if response_text else 0}")
            self.logger.info(f"==> Full response content:\n{response_text}")
            self.logger.info(f"=== END RESPONSE ===")
            
            # Extract and validate commands
            try:
                self.logger.info(f"Attempting to extract commands from response...")
                analysis, commands = self._extract_commands_and_analysis_from_response(response_text)
                self.logger.info(f"==> Successfully extracted analysis: {analysis}")
                self.logger.info(f"==> Successfully extracted {len(commands)} commands: {commands}")
                valid_commands, validation_errors = validate_tool_commands(
                    commands, input_kwargs, required_params
                )
                if valid_commands:
                    self.logger.info(f"Generated {len(valid_commands)} valid command(s)")
                    return analysis, valid_commands
                else:
                    error_msg = f"Command validation failed: {validation_errors}"
                    self.logger.error(f"{error_msg}")
                    previous_errors.append(error_msg)
            except Exception as e:
                error_msg = f"Command extraction failed: {str(e)}"
                self.logger.error(f"{error_msg}")
                self.logger.error(f"Raw response that failed parsing: {repr(response_text)}")
                previous_errors.append(error_msg)
        
        # All attempts failed
        raise RuntimeError(f"Failed to generate valid commands after 3 attempts. Last errors: {previous_errors}")

    def generate_commands(self, tool_to_use: str, step_goal: str, step_context: str):
        """Generate commands for a given tool and action."""
        self.logger.info(f"Starting command generation for tool: {tool_to_use}")
        
        # Load tool and get metadata
        try:
            tool_metadata = get_tool_metadata(tool_to_use, self.llm)
        except Exception as e:
            error_msg = f"Failed to load tool {tool_to_use}: {e}"
            self.logger.error(f"{error_msg}")
            return error_msg, None
        
        # Get formatted tool info
        tool_info = tool_metadata['formatted_info']
        
        # Generate commands
        try:
            analysis, commands = self._generate_tool_commands(
                tool_to_use, step_goal, step_context, 
                tool_info, tool_metadata['metadata']
            )
            return analysis, commands
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_msg = f"Failed to generate commands: {e}"
            self.logger.error(f"{error_msg}")
            self.logger.error(f"Traceback:\n{tb_str}")
            return error_msg, None

    def execute_commands(self, tool_to_use: str, commands: List[Dict[str, Any]], workspace_path: str):
        """Execute previously generated commands."""
        if commands is None:
            error_result = [{"tool_name": tool_to_use, "error": "No command data provided", "success": False}]
            error_logs = {
                "generated_commands": [],
                "execution_logs": error_result,
                "error": "No command data provided"
            }
            return error_result, error_logs
        
        # Set custom output directory for the tool instance
        try:
            tool_metadata = get_tool_metadata(tool_to_use, self.llm)
            tool_instance = tool_metadata['tool_instance']
            tool_instance.set_custom_output_dir(workspace_path)
        except Exception as e:
            error_msg = f"Failed to load tool {tool_to_use}: {e}"
            self.logger.error(f"{error_msg}")
            error_result = [{"tool_name": tool_to_use, "error": error_msg, "success": False}]
            error_logs = {
                "generated_commands": commands or [],
                "execution_logs": error_result,
                "error": error_msg
            }
            return error_result, error_logs
        
        self.logger.info(f"Starting command execution for tool: {tool_to_use}")
        
        # Execute commands in parallel
        execution_results = self._execute_commands_parallel(tool_instance, commands)
        
        return execution_results

    def _execute_commands_parallel(self, tool_instance, commands: List[Dict[str, Any]], 
                                  concurrent_workers: int = None):
        """Execute tool commands in parallel."""
        if concurrent_workers is None:
            concurrent_workers = self.concurrent_workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = []
            for idx, command in enumerate(commands):
                self.logger.info(f"Submitting command {idx}: {command}")
                future = executor.submit(self._execute_single_command, tool_instance, command)
                futures.append(future)
            
            command_results = []
            for future in concurrent.futures.as_completed(futures, timeout=self.max_execution_time):
                try:
                    result = future.result(timeout=self.execution_timeout)
                    command_results.append(result)
                except Exception as e:
                    command_results.append({
                        "command_arguments": {},
                        "result": None,
                        "success": False,
                        "error": str(e),
                        "execution_time": 0
                    })
        
        return command_results
    
    def _execute_single_command(self, tool_instance, command: Dict[str, Any]):
        """Execute a single tool command."""
        try:
            # Record start time before execution
            start_time = time.time()
            # Execute the tool with the provided parameters
            tool_result = tool_instance.run(**command)
            # Calculate execution time
            execution_time = round(time.time() - start_time, 2)
            
            # Format result
            if isinstance(tool_result, dict):
                result_data = tool_result
            else:
                result_data = {"result": str(tool_result)}
            
            # Build response
            response = {
                "command_arguments": command,
                "result": result_data,
                "success": True,
                "error": None,
                "execution_time": execution_time
            }
            
            return response
            
        except Exception as e:
            return {
                "command_arguments": command,
                "result": None,
                "success": False,
                "error": str(e),
                "execution_time": 0
            }
