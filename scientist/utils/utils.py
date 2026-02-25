import json
import yaml
import re
import importlib
import inspect
from typing import Dict, Any, Optional, List, Tuple, Type, Union
from pydantic import BaseModel, Field, create_model


def load_yaml(path: str) -> dict:
    """Load YAML file."""
    try:
        with open(path, 'r') as f:
            result = yaml.safe_load(f)
            return result
    except Exception as e:
        print(f"Error loading yaml from {path}: {e}")
        return {}

def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Robustly parse a JSON string from LLM output, handling markdown code block wrappers and various formats."""
    response = response.strip()
    
    # 1. Try extracting content between ```json ... ``` using regex
    json_block_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_block_pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except Exception:
            pass

    # 2. Try extracting content between ``` ... ``` using regex
    code_block_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except Exception:
            pass
    
    # 3. Try parsing the original string directly
    try:
        # print("Triggering Type 3")
        return json.loads(response)
    except Exception:
        pass
    
    # 4. Try extracting content between first and last ```
    if '```' in response:
        first_backticks = response.find('```')
        last_backticks = response.rfind('```')
        if first_backticks != last_backticks:  # Ensure we have both opening and closing
            json_str = response[first_backticks + 3:last_backticks].strip()
            try:
                # print("Triggering Type 4")
                return json.loads(json_str)
            except Exception:
                pass
    
    # 5. Use regex to extract content between first { and last }
    if '{' in response and '}' in response:
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        if first_brace < last_brace:  # Ensure proper order
            json_str = response[first_brace:last_brace + 1]
            try:
                # print("Triggering Type 5")
                return json.loads(json_str)
            except Exception as e:
                # Try to fix common escape issues
                try:
                    # Handle escaped braces by unescaping them
                    fixed_json = json_str.replace('\\{', '{').replace('\\}', '}')
                    # print("Triggering Type 5 (fixed)")
                    return json.loads(fixed_json)
                except Exception as e2:
                    pass
    return None

def create_pydantic_model_from_schema(schema: Dict[str, Dict[str, str]], model_name: str = "DynamicModel") -> Type[BaseModel]:
    """
    Create a Pydantic model dynamically from a schema dict.
    
    Args:
        schema: Dict with field_name -> {'description': str, 'type': str}
        model_name: Name for the generated model class
        
    Example:
        schema = {
            'answer': {'description': 'The final answer', 'type': 'string'},
            'score': {'description': 'Confidence score', 'type': 'number'}
        }
        Model = create_pydantic_model_from_schema(schema)

    Reference: https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation
    """
    fields = {}
    
    for field_name, field_info in schema.items():
        field_type = field_info.get('type', 'string')
        description = field_info.get('description', '')
        
        # Map string type names to Python types
        if field_type == 'string':
            python_type = str
        elif field_type == 'integer':
            python_type = int
        elif field_type == 'number' or field_type == 'float':
            python_type = float
        elif field_type == 'boolean':
            python_type = bool
        elif field_type == 'array':
            python_type = List[str]  # Default to List[str], could be made more sophisticated
        elif field_type == 'object':
            python_type = Dict[str, Any]
        else:
            # Default to string for unknown types
            python_type = str
            
        # Create field with description
        fields[field_name] = (python_type, Field(description=description))
    
    # Create the model dynamically
    return create_model(model_name, **fields)

def is_structured_schema(schema: Dict[str, Any]) -> bool:
    """
    Check if schema follows structured format:
    {field_name: {'description': str, 'type': str}}
    
    Args:
        schema: The schema dict to check
        
    Returns:
        True if schema follows the structured format, False otherwise
        
    Example:
        schema = {
            'answer': {'description': 'The final answer', 'type': 'string'},
            'score': {'description': 'Confidence score', 'type': 'number'}
        }
        is_structured_schema(schema)  # returns True
    """
    if not isinstance(schema, dict):
        return False
        
    for field_name, field_info in schema.items():
        if not isinstance(field_info, dict):
            return False
        if 'description' not in field_info or 'type' not in field_info:
            return False
        if not isinstance(field_info['description'], str) or not isinstance(field_info['type'], str):
            return False
            
    return True

def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {make_json_serializable(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(element) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        return str(obj)
    
def make_json_serializable_truncated(obj, max_length: int = 100000):
    if isinstance(obj, (int, float, bool, type(None))):
        if isinstance(obj, (int, float)) and len(str(obj)) > max_length:
            return str(obj)[:max_length - 3] + "..."
        return obj
    elif isinstance(obj, str):
        return obj if len(obj) <= max_length else obj[:max_length - 3] + "..."
    elif isinstance(obj, dict):
        return {make_json_serializable_truncated(key, max_length): make_json_serializable_truncated(value, max_length) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable_truncated(element, max_length) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable_truncated(obj.__dict__, max_length)
    else:
        result = str(obj)
        return result if len(result) <= max_length else result[:max_length - 3] + "..."
