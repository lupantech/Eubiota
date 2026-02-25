# Customization Guide

This guide details how to customize Science Agent, with a focus on adding new tools and configuring agent capabilities.

## 1. Adding New Tools

Science Agent uses a modular tool system. To add a new tool, follow these steps:

### Step 1: Create Tool Directory
Create a new directory for your tool in `scientist/tools/`. It is recommended to keep related files together.

```bash
scientist/tools/
└── Your_Tool_Name/
    ├── __init__.py
    ├── tool.py       # Implementation class
    └── README.md     # Optional documentation
```

### Step 2: Implement the Tool Class
Inherit from the `Tool` class defined in `scientist.tools.base_tool`. You need to implement the `__init__` method and the `run` method.

**Example Implementation (`scientist/tools/Your_Tool_Name/tool.py`):**

```python
from scientist.tools.base_tool import Tool

class YourCustomTool(Tool):
    def __init__(self, llm=None):
        super().__init__(
            name="Your_Custom_Tool",
            description="A clear description of what this tool does.",
            input_kwargs={
                "query": {"type": "string", "description": "The input query for the tool."}
            },
            output_schema={
                "result": {"type": "string", "description": "The output result."}
            },
            limitations="Describe any limitations here.",
            best_practices="Describe best practices here.",
            llm=llm
        )
        # Set to True if your tool needs to access self.llm
        self.require_llm_engine = False 

    def run(self, query: str, **kwargs):
        """
        Execute the tool logic.
        
        Args:
            query (str): The input query.
            **kwargs: Additional arguments.
            
        Returns:
            dict: A dictionary containing 'result' and optionally 'generated_files'.
        """
        # Your custom logic here
        result_str = f"Processed: {query}"
        
        return {
            "result": result_str,
            "generated_files": []
        }
```

### Step 3: Register in Configuration
To enable your tool, you need to add it to the training configuration file: `trainer/train_scientist/config.yaml`.

Find the `ENABLE_TOOLS` and `TOOL_ENGINE` lists and append your tool.

```yaml
env:
  # ... other settings ...
  
  # Add your tool class name to the list
  ENABLE_TOOLS: [
    "Google_Search_Tool", 
    # ... other tools ...
    "YourCustomTool" 
  ]
  
  # Specify the engine for your tool (use "None" if it's deterministic/code-based, or "gpt-4o" if it uses LLM)
  TOOL_ENGINE: [
    "None",
    # ... corresponding to other tools ...
    "None"
  ]
```

**Note on `TOOL_ENGINE`**:
*   If your tool implementation uses `self.llm`, specify the model name (e.g., `"gpt-4o"`, `"dashscope"`).
*   If your tool is purely code-based (like a calculator or database lookup), set it to `"None"`.