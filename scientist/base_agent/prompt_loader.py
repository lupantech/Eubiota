"""
Prompt Loader Utility
Loads prompts from YAML files.
"""

import os
import yaml
from typing import Dict, Any


def load_prompts(module_name: str, prompts_dir: str = None) -> Dict[str, str]:
    """
    Load prompts from YAML file for a specific module.

    Args:
        module_name: Name of the module (planner, executor, verifier, generator)
        prompts_dir: Optional custom path to prompts directory

    Returns:
        Dictionary mapping prompt names to prompt templates
    """
    if prompts_dir is None:
        # Default to prompts directory in models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(current_dir, 'prompts')

    yaml_path = os.path.join(prompts_dir, f"{module_name}.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Prompts file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)

    return prompts


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with the provided keyword arguments.

    Args:
        template: The prompt template string
        **kwargs: Variables to substitute in the template

    Returns:
        Formatted prompt string
    """
    # Handle optional placeholders
    formatted = template
    for key, value in kwargs.items():
        placeholder = f"{{{key}}}"
        if placeholder in formatted:
            # If value is provided, replace it
            formatted = formatted.replace(placeholder, str(value))

    return formatted
