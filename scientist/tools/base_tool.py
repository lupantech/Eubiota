import abc
import os
from typing import List, Dict, Any, Optional

class Tool(abc.ABC):
    """
    Base Tool class for all scientist tools.
    All tools should inherit from this class and implement the run() method.

    Tools can optionally use an LLM engine by setting require_llm_engine = True
    and accepting an llm parameter in __init__.
    """

    require_llm_engine = False  # Default is False, tools that need LLM should set this to True

    def __init__(self, name: str, description: str, input_kwargs: dict, output_schema: dict,
                 limitations: str, best_practices: str, documentation_path: str = None,
                 llm=None):
        self.name = name
        self.description = description
        self.limitations = limitations
        self.best_practices = best_practices
        self.input_kwargs = input_kwargs
        self.output_schema = output_schema
        self.documentation_path = documentation_path
        self.output_dir = None  # Will be set by executor
        self.generated_files = []  # Track files generated during execution

        # LLM engine support (optional, can be None if tool doesn't need it)
        self.llm = llm

        if self.documentation_path:
            # Check if it's a URL or local file path
            if self.documentation_path.startswith(('http://', 'https://')):
                # For URLs, just store the URL as documentation reference
                self.documentation = f"Documentation available at: {self.documentation_path}"
            else:
                # For local files, try to read the content
                try:
                    with open(self.documentation_path, "r") as f:
                        self.documentation = f.read()
                except FileNotFoundError:
                    self.documentation = f"Documentation file not found: {self.documentation_path}"
        else:
            self.documentation = "No documentation available"

    def get_name(self):
        return self.name

    def get_description(self):
        return self.description

    def get_input_kwargs(self):
        return self.input_kwargs

    def get_output_schema(self):
        return self.output_schema

    def set_custom_output_dir(self, output_dir):
        self.output_dir = output_dir
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def set_llm_engine(self, llm):
        """
        Set the LLM engine for the tool.

        Parameters:
            llm: The LLM engine instance.
        """
        self.llm = llm

    def get_metadata(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_kwargs": self.input_kwargs,
            "output_schema": self.output_schema,
            "limitations": self.limitations,
            "best_practices": self.best_practices,
            "documentation": self.documentation,
            "require_llm_engine": self.require_llm_engine
        }

    @abc.abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        Returns:
            Dict containing at least:
            - result: The main result of the tool
            - generated_files: List of files generated (optional)
        """
        pass
    