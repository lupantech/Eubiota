import os
import re
import json
from typing import Dict, Any, Tuple

from scientist.utils.utils import load_yaml
from scientist.base_agent.formatters import ReflectionResult

class Verifier:
    def __init__(self, llm: Any, role_prompt: str, prompts_path: str = None, logger=None):
        self.llm = llm
        self.role_prompt = role_prompt
        self.logger = logger
        self.prompts = self._load_prompts(prompts_path)
    
    def _load_prompts(self, prompts_path: str = None) -> Dict[str, str]:
        """Load verifier prompts from YAML file."""
        if prompts_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, 'prompts', 'verifier.yaml')
        return load_yaml(prompts_path)
    
    def reflect_and_decide(self, agent_query: str, input_files: Dict[str, Any], 
                          available_tools: list, toolbox_metadata: dict,
                          global_plan: str, memory) -> str:
        """Reflect on the current state and decide whether to continue."""
        previous_steps = memory.get_concise_steps()
        current_files_display = [os.path.basename(f) for f in input_files.keys()] if input_files else []
        
        prompt = self.prompts.get('reflection_decision', '').format(
            agent_query=agent_query,
            current_files_display=current_files_display,
            available_tools=available_tools,
            toolbox_metadata=json.dumps(toolbox_metadata, indent=4),
            global_plan=global_plan,
            previous_steps=previous_steps
        )

        reflection_result = None
        
        # Try with response_format first (preferred method)
        try:
            self.logger.info("Attempting LLM generation with response_format=ReflectionResult")
            reflection_result = self.llm.generate(
                prompt, 
                system_prompt=self.role_prompt, 
                response_format=ReflectionResult
            )
            self.logger.info(f"Successfully generated reflection with response_format: {type(reflection_result)}")
                
        except TypeError as te:
            # LLM doesn't support response_format parameter
            self.logger.warning(f"LLM doesn't support response_format, falling back to standard generation: {te}")
            try:
                reflection_result = self.llm.generate(
                    prompt, 
                    system_prompt=self.role_prompt
                )
                self.logger.info(f"Successfully generated reflection without response_format: {type(reflection_result)}")
                    
            except Exception as e2:
                self.logger.error(f"Failed to generate reflection without response_format: {e2}")
                reflection_result = None
                
        except Exception as e:
            # Any other exception during generation
            self.logger.error(f"Unexpected error during LLM generation: {e}")
            reflection_result = None
            
        # Validate the result
        if reflection_result is not None:
            self.logger.info(f"Reflection result type: {type(reflection_result)}")
        else:
            self.logger.warning("No reflection result generated, will use fallback parsing")
            
        return reflection_result

    def parse_reflection(self, reflection_result: str | ReflectionResult) -> Tuple[str, bool]:
        """Parse verifier.yaml format string: Analysis: ... Conclusion: STOP/CONTINUE
        Note: ReflectionResult objects are handled directly in agent.py for efficiency.
        """
        # Handle None or unexpected types
        if reflection_result is None:
            self.logger.warning("reflection_result is None, treating as continue signal")
            return "No reflection analysis available", True
            
        # Log the type for debugging
        self.logger.debug(f"reflection_result type: {type(reflection_result)}")
            
        # Method 1: Direct type checking for efficiency
        if isinstance(reflection_result, ReflectionResult):
            self.logger.info(f"Successfully parsed ReflectionResult (method 1): {type(reflection_result)}")
            analysis = reflection_result.analysis    
            continue_signal = not reflection_result.stop_signal  # stop_signal=False means continue
            return analysis, continue_signal
        
        # Convert to string if it's not already a string or ReflectionResult
        if not isinstance(reflection_result, str):
            self.logger.info(f"Converting reflection_result from {type(reflection_result)} to string")
            reflection_result = str(reflection_result)
        
        # Method 2: Strict regular parsing for non-ReflectionResult objects
        # Parse verifier.yaml format: Analysis: ... Conclusion: STOP/CONTINUE
        # Extract Analysis content
        analysis_match = re.search(r'Analysis:\s*(.*?)(?=Conclusion:|$)', reflection_result, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else ""
        
        # Extract Conclusion
        conclusion_match = re.search(r'Conclusion:\s*(STOP|CONTINUE)', reflection_result, re.IGNORECASE)
        if conclusion_match:
            self.logger.info(f"Successfully parsed Conclusion with strict parsing (method 2): {type(conclusion_match)}")
            conclusion = conclusion_match.group(1).upper()
            continue_signal = conclusion == 'CONTINUE'
            return analysis, continue_signal
        # raise ValueError(f"No valid conclusion found in the response: {reflection_result}")

        # Method 3: Flexible parsing for non-ReflectionResult objects (octotools script)
        self.logger.info("Fallback: Flexible parsing for non-ReflectionResult objects...")
        pattern = r'conclusion\**:?\s*\**\s*(\w+)'
        matches = list(re.finditer(pattern, reflection_result, re.IGNORECASE | re.DOTALL))
        if matches:
            conclusion = matches[-1].group(1).upper()
            if conclusion in ['STOP', 'CONTINUE']:
                continue_signal = conclusion == 'CONTINUE'
                self.logger.info(f"Successfully parsed Conclusion with flexible parsing (method 3): {type(continue_signal)}")
                return reflection_result, continue_signal

        # Method 4: If no valid conclusion found, search for STOP or CONTINUE anywhere in the text
        self.logger.info("Fallback: Searching for STOP or CONTINUE in the response...")
        if 'stop' in reflection_result.lower():
            continue_signal = False
            self.logger.info(f"Successfully parsed Conclusion with string matching (method 4): {type(continue_signal)}")
            return reflection_result, continue_signal
        elif 'continue' in reflection_result.lower():
            continue_signal = True
            self.logger.info(f"Successfully parsed Conclusion with string matching (method 4): {type(continue_signal)}")
            return reflection_result, continue_signal
        else:
            self.logger.warning("No valid conclusion found in the response. Continuing...")
            return reflection_result, True
