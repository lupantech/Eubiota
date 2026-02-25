import os
import json
from typing import Dict, Any

from scientist.utils.utils import (
    load_yaml, 
    parse_json_response, 
    create_pydantic_model_from_schema, 
    is_structured_schema
)

class Generator:
    def __init__(self, llm: Any, role_prompt: str, prompts_path: str = None, logger=None):
        self.llm = llm
        self.role_prompt = role_prompt
        self.logger = logger
        self.prompts = self._load_prompts(prompts_path)
    
    def _load_prompts(self, prompts_path: str = None) -> Dict[str, str]:
        """Load generator prompts from YAML file."""
        if prompts_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, 'prompts', 'generator.yaml')
        return load_yaml(prompts_path)
    
    def generate_direct_output(self, agent_query: str, input_files: Dict[str, Any], global_plan, concise_steps: Dict[str, Any]) -> str:
        """Generate a concise direct output based on the query and execution history."""
        current_files_display = [os.path.basename(f) for f in input_files.keys()] if input_files else []
        
        prompt = self.prompts.get('direct_output_generation', '').format(
            agent_query=agent_query,
            current_files_display=current_files_display,
            global_plan=global_plan,
            execution_history=concise_steps
        )
        
        try:
            response = self.llm.generate(
                content=prompt,
                system_prompt=self.role_prompt,
            )
            if response is not None:
                self.logger.info(f"Successfully generated direct output: {type(response)}")
                return str(response)
            else:
                self.logger.warning("No response generated from LLM")
                return "No response generated from LLM"
        except Exception as e:
            self.logger.error(f"Error generating direct output: {e}")
            return "No response generated from LLM"

    def generate_final_output(self, agent_query: str, input_files: Dict[str, Any], global_plan, concise_steps: Dict[str, Any]) -> Any:
        """Generate the final output based on execution history using FinalOutput model."""
        current_files_display = [os.path.basename(f) for f in input_files.keys()] if input_files else []
        
        prompt = self.prompts.get('final_output_generation', '').format(
            agent_query=agent_query,
            global_plan=global_plan,
            current_files_display=current_files_display,
            execution_history=concise_steps
        )
        
        # Use FinalOutput model for default structure
        response = self.llm.generate(
            content=prompt,
            system_prompt=self.role_prompt
        )

        # Handle None response
        if response is not None:
            self.logger.info(f"Successfully generated final output: {type(response)}")
            return str(response)
        else:
            self.logger.error("LLM generate_final_output returned None")
            return "No final output generated"

    def generate_schema_output(self, agent_query: str, input_files: Dict[str, Any], 
                              global_plan, concise_steps: Dict[str, Any], output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate schema-based output using custom schema with response_format control."""
        if not output_schema:
            raise ValueError("output_schema is required for generate_schema_output")
            
        current_files_display = [os.path.basename(f) for f in input_files.keys()] if input_files else []
        
        prompt = self.prompts.get('schema_output_generation', '').format(
            agent_query=agent_query,
            global_plan=global_plan,
            current_files_display=current_files_display,
            execution_history=concise_steps,
            schema_fields=json.dumps(output_schema, indent=4)
        )
        
        # Check if schema follows new format: {field: {'description': str, 'type': str}}
        if is_structured_schema(output_schema):
            # Try with dynamic Pydantic model first
            try:
                DynamicModel = create_pydantic_model_from_schema(output_schema, "CustomOutputModel")
                response = self.llm.generate(
                    content=prompt,
                    system_prompt=self.role_prompt,
                    response_format=DynamicModel
                )
                
                # If response is the model instance, convert to dict (full support)
                if hasattr(response, 'model_dump'):
                    self.logger.info(f"Successfully generated schema output: {type(response)}")
                    return response.model_dump()
                else:
                    # Model doesn't support response_format, returned string instead
                    # Fall back to JSON parsing
                    self.logger.warning("LLM returned string instead of Pydantic model, attempting JSON parsing")
                    parsed_response = parse_json_response(response)
                    if isinstance(parsed_response, dict):
                        self.logger.info(f"Successfully parsed JSON from string response: {type(parsed_response)}")
                        return parsed_response
                    else:
                        self.logger.error(f"Failed to parse JSON from string response: {type(parsed_response)}")
                        raise ValueError("Failed to parse JSON from string response")
                      
            except Exception as e:
                self.logger.error(f"Error generating schema output: {e}")
                raise e
        else:
            self.logger.error("Invalid schema format")
            raise ValueError("Invalid schema format")

