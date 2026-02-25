import json
import importlib
import inspect
from typing import Any, Dict, List, Tuple, Optional

# Global tool instance cache to avoid re-instantiation
_TOOL_INSTANCE_CACHE = {}

def clear_tool_cache():
    """Clear the global tool instance cache. Useful for testing or when tools need to be reloaded."""
    global _TOOL_INSTANCE_CACHE
    _TOOL_INSTANCE_CACHE.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the tool instance cache."""
    return {
        "total_cached_instances": len(_TOOL_INSTANCE_CACHE),
        "cached_tools": list(_TOOL_INSTANCE_CACHE.keys())
    }

def load_tool_dynamically(tool_name: str, llm_instance: Any = None, engine_name: str = None, use_cache: bool = True) -> Any:
    """
    Dynamically load and instantiate a tool by name with caching support.
    Args:
        tool_name: str, the name of the tool to load
        llm_instance: Any, the LLM instance to pass to the tool (legacy support)
        engine_name: str, the engine name to pass to tools that support it (e.g., "claude-sonnet-4-0", "gpt-4o")
        use_cache: bool, whether to use cached instance (default: True)
    Returns:
        Any, the instantiated tool
    Example 1:
        tool_instance = load_tool_dynamically('Google_Search_Tool')
        tool_instance.run(**kwargs)
    Example 2:
        tool_instance = load_tool_dynamically('Base_Generator_Tool', engine_name='claude-sonnet-4-0')
        tool_instance.run(**kwargs)
    """
    try:
        # Create cache key based on tool name and model
        # Use model_string from llm_instance instead of id() for proper sharing
        # This allows:
        # - Tools with same model (e.g., all gpt-4o agents) share instances
        # - Tools with different models (gpt-4o vs claude) use different instances
        # - Embeddings-based tools still shared across agents using same model
        model_key = 'default'
        if engine_name:
            model_key = engine_name
        elif llm_instance and hasattr(llm_instance, 'model_string'):
            model_key = llm_instance.model_string
        elif llm_instance:
            model_key = 'shared'
        
        cache_key = f"{tool_name}_{model_key}"

        # Return cached instance if available and caching is enabled
        if use_cache and cache_key in _TOOL_INSTANCE_CACHE:
            return _TOOL_INSTANCE_CACHE[cache_key]

        # Use TOOL_REGISTRY for direct access
        from scientist.tools import TOOL_REGISTRY

        if tool_name not in TOOL_REGISTRY:
            raise RuntimeError(f"Tool '{tool_name}' not found in TOOL_REGISTRY. Available tools: {list(TOOL_REGISTRY.keys())}")

        tool_class = TOOL_REGISTRY[tool_name]

        # Check what parameters the tool needs for initialization
        sig = inspect.signature(tool_class.__init__)
        params = sig.parameters # e.g. keys: dict_keys(['self', 'engine_name', 'llm'])

        # Prepare initialization arguments based on signature detection
        init_kwargs = {}

        # Priority: engine_name parameter (new way) > llm instance (legacy way)
        if 'engine_name' in params and engine_name is not None:
            init_kwargs['engine_name'] = engine_name
        elif 'llm' in params and llm_instance is not None:
            init_kwargs['llm'] = llm_instance

        # For parameters that don't have defaults, provide None
        for param_name, param in params.items():
            if param_name not in ['self'] and param_name not in init_kwargs:
                if param.default == inspect.Parameter.empty:
                    # Required parameter without default, provide None
                    init_kwargs[param_name] = None

        # Instantiate the tool with detected parameters
        tool_instance = tool_class(**init_kwargs)

        # Cache the instance if caching is enabled
        if use_cache:
            _TOOL_INSTANCE_CACHE[cache_key] = tool_instance

        return tool_instance

    except Exception as e:
        raise RuntimeError(f"Failed to load tool {tool_name}: {e}")

def get_tool_metadata(tool_name: str, llm_instance: Any = None, engine_name: str = None) -> Dict[str, Any]:
    """Get comprehensive tool metadata including limitations and best practices."""

    # Load the tool instance (with engine_name support)
    tool_instance = load_tool_dynamically(tool_name, llm_instance, engine_name)
    if not tool_instance:
        raise RuntimeError(f"Failed to load tool {tool_name}")
    
    # Get tool metadata
    metadata = tool_instance.get_metadata()
    
    # Format input requirements for prompt
    input_info = []
    for param_name, param_config in metadata.get('input_kwargs', {}).items():
        param_type = param_config.get('type', 'string')
        is_optional = param_config.get('optional', False)
        description = param_config.get('description', 'No description available')
        required_text = "Optional" if is_optional else "Required"
        input_info.append(f"  - {param_name} ({param_type}, {required_text}): {description}")
    
    # Create formatted tool info for prompt
    tool_info = f"""
Description: {metadata.get('description', 'No description available')}

Input Parameters:
{chr(10).join(input_info)}

Output Schema: {json.dumps(metadata.get('output_schema', {}), indent=2)}

Limitations:
{metadata.get('limitations', 'No limitations specified')}

Best Practices:
{metadata.get('best_practices', 'No best practices specified')}

Documentation: {metadata.get('documentation', 'No documentation available')}
""".strip()

    return {
        'tool_instance': tool_instance,
        'metadata': metadata,
        'formatted_info': tool_info
    }

def validate_tool_commands(commands: List[Dict[str, Any]], input_kwargs: Dict[str, Any], 
                          required_params: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate that commands contain all required parameters with correct types.
    
    Returns:
        tuple: (valid_commands, validation_errors)
        - If all commands are valid: (commands, [])
        - If any command is invalid: ([], [error_messages])
    """
    validation_errors = []
    valid_commands = []
    
    for i, command in enumerate(commands):
        # Check if command is a dictionary
        if not isinstance(command, dict):
            validation_errors.append(f"Command {i+1} is not a dictionary: {type(command)}")
            continue
        
        # Check if all required parameters are present
        missing_params = []
        for param in required_params:
            if param not in command:
                missing_params.append(param)
            elif command[param] is None or command[param] == "":
                missing_params.append(f"{param} (empty/null)")
        
        if missing_params:
            validation_errors.append(f"Command {i+1} missing required parameters: {missing_params}")
            continue
        
        # Validate parameter types
        invalid_types = []
        for param_name, param_value in command.items():
            if param_name in input_kwargs:
                # This defaults to 'string' if the 'type' key is missing
                expected_type = input_kwargs[param_name].get('type', 'string')
                if not validate_parameter_type(param_value, expected_type):
                    invalid_types.append(
                        f"{param_name} (expected {expected_type}, got {type(param_value).__name__})"
                    )
        
        if invalid_types:
            validation_errors.append(f"Command {i+1} has invalid parameter types: {invalid_types}")
            continue
        
        # If all validations pass for this command
        valid_commands.append(command)
    
    # Return empty list if any validation failed (strict validation)
    if validation_errors:
        return [], validation_errors
    
    return valid_commands, []

def validate_parameter_type(value: Any, expected_type: str) -> bool:
    """Validate that a parameter value matches the expected type."""
    if expected_type == 'string':
        return isinstance(value, str)
    elif expected_type == 'boolean':
        return isinstance(value, bool)
    elif expected_type == 'integer':
        return isinstance(value, int)
    elif expected_type == 'number':
        return isinstance(value, (int, float))
    elif expected_type == 'array':
        return isinstance(value, list)
    elif expected_type == 'object':
        return isinstance(value, dict)
    else:
        # For unknown types, just check it's not None
        return value is not None

def get_all_tools_metadata(suggested_tools: Optional[List[str]], llm_instance: Any, tool_engines: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all suggested tools using the same logic as executor.

    Args:
        suggested_tools: List of tool names to load
        llm_instance: Default LLM instance to pass to tools (legacy support)
        tool_engines: Optional list of engine names for each tool (matches suggested_tools order)
                      - "Default": Use tool's default engine
                      - "None": Tool doesn't need engine
                      - specific model name: Use that engine (e.g., "claude-sonnet-4-0")

    Returns:
        Dict mapping tool names to their metadata
    """
    toolbox_metadata = {}

    for idx, tool_name in enumerate(suggested_tools or []):
        try:
            # Determine engine_name for this tool
            engine_name = None
            engine_config = "Default"  # Default value
            if tool_engines and idx < len(tool_engines):
                engine_config = tool_engines[idx]
                # Don't pass "Default" or "None" as engine_name - let tool use its default
                if engine_config not in ["Default", "None", None]:
                    engine_name = engine_config
                    print(f"[INFO] Initializing {tool_name} with engine: {engine_name}")

            tool_data = get_tool_metadata(tool_name, llm_instance, engine_name)
            metadata = tool_data['metadata']
            tool_instance = tool_data['tool_instance']

            # Get the actual engine name from the tool instance
            actual_engine = "N/A"
            if hasattr(tool_instance, '_engine_name'):
                actual_engine = tool_instance._engine_name
            elif hasattr(tool_instance, 'require_llm_engine') and not tool_instance.require_llm_engine:
                actual_engine = "No LLM Required"

            toolbox_metadata[tool_name] = {
                "description": metadata.get("description", ""),
                "input_kwargs": metadata.get("input_kwargs", {}),
                "limitations": metadata.get("limitations", ""),
                "best_practices": metadata.get("best_practices", ""),
                "documentation": metadata.get("documentation", ""),
                "engine": actual_engine,  # Add engine information
            }
        except Exception as e:
            print(f"[WARNING] Failed to load tool {tool_name}: {e}")
            import traceback
            traceback.print_exc()
            # Skip tools that fail to load
            continue

    return toolbox_metadata

def extract_concise_tool_metadata(tools_metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract only tool names and descriptions"""
    concise_tools = []
    for tool_name, tool_info in tools_metadata.items():
        concise_tools.append({
            "name": tool_name,
            "description": tool_info.get("description", "No description available")
        })
    return concise_tools
