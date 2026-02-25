"""
Initializer for Scientist - Simplified version without tool name mapping
"""

import os
import sys
import importlib
import inspect
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class Initializer:
    """
    Simplified initializer for Scientist tools.
    Directly loads tools without complex name mapping.
    """

    def __init__(
        self,
        enabled_tools: List[str] = None,
        tool_engine: List[str] = None,
        model_string: str = None,
        verbose: bool = False,
        vllm_config_path: str = None,
        base_url: str = None,
        check_model: bool = True
    ):
        """
        Initialize tools for the agent.

        Args:
            enabled_tools: List of tool names to enable (e.g., ["Base_Generator_Tool"])
            tool_engine: List of engine names per tool (e.g., ["gpt-4o-mini"])
            model_string: Default model string
            verbose: Whether to print verbose output
            vllm_config_path: Path to vLLM config
            base_url: Base URL for LLM API
            check_model: Whether to check model availability
        """
        self.toolbox_metadata = {}
        self.available_tools = []
        self.enabled_tools = enabled_tools or []
        self.tool_engine = tool_engine or []
        self.model_string = model_string
        self.verbose = verbose
        self.vllm_config_path = vllm_config_path
        self.base_url = base_url
        self.check_model = check_model

        # Tool instance cache
        self.tool_instances_cache = {}

        # Lock for thread-safe dictionary updates
        self._metadata_lock = threading.Lock()

        if self.verbose:
            print("\n==> Initializing Scientist tools...")
            print(f"Enabled tools: {self.enabled_tools}")
            print(f"Tool engines: {self.tool_engine}")
            print(f"Default model: {self.model_string}")

        self._set_up_tools()

    def get_project_root(self):
        """Find the scientist directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != '/':
            # Look for scientist directory
            if os.path.basename(current_dir) == 'scientist':
                return current_dir
            current_dir = os.path.dirname(current_dir)

        # Fallback: use the parent of models
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _load_single_tool(self, idx: int, tool_name: str, tools_dir: str, scientist_dir: str) -> None:
        """
        Load a single tool and extract its metadata.
        Thread-safe method for parallel tool loading.
        """
        # Convert tool name to directory name: lowercase and remove '_tool' suffix
        # e.g., "Base_Generator_Tool" -> "base_generator"
        dir_name = tool_name.lower()
        if dir_name.endswith('_tool'):
            dir_name = dir_name[:-5]
        tool_dir = os.path.join(tools_dir, dir_name)

        if not os.path.exists(tool_dir):
            if self.verbose:
                print(f"Warning: Tool directory not found: {tool_dir}")
            return

        tool_file = os.path.join(tool_dir, 'tool.py')
        if not os.path.exists(tool_file):
            if self.verbose:
                print(f"Warning: tool.py not found in {tool_dir}")
            return

        try:
            # Import the tool module
            # Use relative import from scientist package
            module_name = f"scientist.tools.{dir_name}.tool"

            if self.verbose:
                print(f"\n==> Importing: {module_name}")

            module = importlib.import_module(module_name)

            # Find tool class
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith('Tool') and name != 'Tool':
                    if self.verbose:
                        print(f"Found tool class: {name}")

                    try:
                        # Get the engine for this tool
                        if idx < len(self.tool_engine):
                            engine = self.tool_engine[idx]
                        else:
                            engine = "Default"

                        # Check if tool requires LLM engine (class attribute)
                        requires_llm = getattr(obj, 'require_llm_engine', False)

                        # Instantiate tool with appropriate parameters
                        if not requires_llm:
                            # Tools without LLM engine don't need any parameters
                            tool_instance = obj()
                        elif engine == "Default":
                            # Use tool's default engine (no explicit engine_name)
                            tool_instance = obj()
                        elif engine == "self":
                            # Use the model_string from initializer config
                            tool_instance = obj(engine_name=self.model_string)
                        else:
                            # Use the specified engine
                            tool_instance = obj(engine_name=engine)

                        # Get tool name from instance
                        metadata_key = getattr(tool_instance, 'tool_name', tool_name)

                        # Thread-safe cache update
                        with self._metadata_lock:
                            # Cache the tool instance
                            self.tool_instances_cache[metadata_key] = tool_instance

                            if self.verbose:
                                print(f"Cached tool: {metadata_key}")

                            # Extract metadata
                            self.toolbox_metadata[metadata_key] = {
                                'tool_name': getattr(tool_instance, 'tool_name', tool_name),
                                'tool_description': getattr(tool_instance, 'tool_description', 'No description'),
                                'tool_version': getattr(tool_instance, 'tool_version', '1.0'),
                                'input_types': getattr(tool_instance, 'input_types', {}),
                                'output_type': getattr(tool_instance, 'output_type', 'str'),
                                'demo_commands': getattr(tool_instance, 'demo_commands', []),
                                'user_metadata': getattr(tool_instance, 'user_metadata', {}),
                                'require_llm_engine': getattr(obj, 'require_llm_engine', False),
                            }

                            if self.verbose:
                                print(f"Metadata: {self.toolbox_metadata[metadata_key]}")

                    except Exception as e:
                        if self.verbose:
                            print(f"Error instantiating {name}: {str(e)}")

                    break  # Only load the first matching tool class

        except Exception as e:
            if self.verbose:
                print(f"Error loading tool {tool_name}: {str(e)}")
                import traceback
                traceback.print_exc()

    def load_tools_and_get_metadata(self) -> Dict[str, Any]:
        """
        Load tools and extract metadata using parallel processing.
        Simplified version: directly derives directory from tool name.
        """
        if self.verbose:
            print("Loading tools and getting metadata (parallel mode)...")

        self.toolbox_metadata = {}
        scientist_dir = self.get_project_root()
        tools_dir = os.path.join(scientist_dir, 'tools')

        if self.verbose:
            print(f"Scientist directory: {scientist_dir}")
            print(f"Tools directory: {tools_dir}")

        # Add to Python path
        sys.path.insert(0, scientist_dir)
        sys.path.insert(0, os.path.dirname(scientist_dir))

        if not os.path.exists(tools_dir):
            if self.verbose:
                print(f"Warning: Tools directory does not exist: {tools_dir}")
            return self.toolbox_metadata

        # Pre-import the tools package to avoid import deadlock in parallel loading
        # This ensures all __init__.py files are already loaded before threads start
        try:
            import scientist.tools
            if self.verbose:
                print("\n==> Pre-imported scientist.tools package to avoid import deadlocks")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to pre-import scientist.tools: {e}")

        # Determine number of worker threads (use CPU count)
        max_workers = os.cpu_count() or 4
        if self.verbose:
            print(f"==> Using {max_workers} parallel workers for tool loading")

        # Use ThreadPoolExecutor for parallel tool loading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tool loading tasks
            futures = []
            for idx, tool_name in enumerate(self.enabled_tools):
                future = executor.submit(
                    self._load_single_tool,
                    idx,
                    tool_name,
                    tools_dir,
                    scientist_dir
                )
                futures.append(future)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    if self.verbose:
                        print(f"Error in parallel tool loading: {str(e)}")

        if self.verbose:
            print(f"\n==> Total tools loaded: {len(self.toolbox_metadata)}")

        return self.toolbox_metadata

    def _set_up_tools(self) -> None:
        """Set up tools by loading metadata and marking as available."""
        if self.verbose:
            print("\n==> Setting up tools...")

        # Load tools and metadata
        self.load_tools_and_get_metadata()

        # All loaded tools are available (no demo command checking)
        self.available_tools = list(self.toolbox_metadata.keys())

        if self.verbose:
            print(f"✅ Tools setup complete")
            print(f"✅ Available tools: {self.available_tools}")


if __name__ == "__main__":
    # Test - All available tools (now including Pubmed and Database RAG)
    enabled_tools = [
        "Base_Generator_Tool",
        "Wikipedia_Search_Tool",
        "KEGG_Gene_Search_Tool",
        "PubMed_Search_Tool",
        "Google_Search_Tool",
        "Python_Coder_Tool",
        "URL_Context_Search_Tool"
    ]
    tool_engine = ["gpt-4o-mini"] * len(enabled_tools)
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string="gpt-4o-mini",
        verbose=True
    )

    print("\nAvailable tools:")
    print(initializer.available_tools)

    print("\nToolbox metadata:")
    for tool_name, metadata in initializer.toolbox_metadata.items():
        print(f"  {tool_name}: {metadata['tool_description']}")
