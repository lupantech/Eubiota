import os
import re
import sys
import pickle
import multiprocessing
from io import StringIO
import contextlib
from typing import List, Tuple, Optional, Dict, Any

from scientist.tools.base_tool import Tool
from scientist.engine.factory import create_llm_engine

# Tool name constant
TOOL_NAME = "Python_Coder_Tool"

LIMITATIONS = f"""
1. {TOOL_NAME} is a general-purpose Python code generator/executor (not limited to arithmetic).
2. {TOOL_NAME} may use Python standard library modules (e.g., `json`) and common data-structure operations (lists/dicts/sets, parsing, aggregation).
3. Execution is time-limited (default 10 seconds) and output/variable capture is truncated to avoid runaway logs.
4. For workflow runs, tools may be given a run-specific workspace directory (`output_dir`) where they can write artifacts; prefer writing intermediate outputs there.
5. This tool is not intended for heavy computation or very large datasets; keep processing bounded and efficient.
6. This tool uses Python exec() to run code. Make sure generated code can be run with exec().
"""

BEST_PRACTICES = f"""
1. Provide clear I/O instructions: where the input comes from (inline JSON vs a file path) and where to save outputs (e.g., write `output.json` into the workspace).
2. For JSON tasks: specify the schema (keys you expect), desired transformation, and required output schema.
3. Ask for a concise printed summary and write full results to a JSON file when large.
4. Keep tasks bounded so they complete within the execution timeout.
5. Never assume this tool has access to context from previous steps. Always provide context in the tool input query.
6. When reading unknown JSON, **inspect the top-level type first** (`dict` vs `list`) and branch accordingly.
7. Always include file extension when reading files (i.e. "file.json" instead of "file")
8. If previous attempts produce errors, analyze the error and correct the generated code. Include corrections when prompting the tool in future iterations.
9. When absolute file paths are provided in the query, use them EXACTLY as given. Do NOT convert absolute paths to relative paths.
"""

# Helper function to execute code in a subprocess
def _execute_code_in_process(code: str, result_queue: multiprocessing.Queue, output_dir: str = None):
    """
    Execute code in a subprocess and put results in the queue.
    This function runs in a separate process.
    """
    import sys
    import os
    from io import StringIO

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    local_vars = {}
    captured_output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        exec(code, local_vars, local_vars)
        sys.stdout = old_stdout

        # Filter serializable variables
        safe_vars = {}
        for k, v in local_vars.items():
            if not k.startswith('__') and not isinstance(v, type(sys)):
                try:
                    # Check if variable is serializable
                    pickle.dumps(v)
                    safe_vars[k] = str(v)[:20000]
                except:
                    safe_vars[k] = f"<{type(v).__name__}>"

        result_queue.put({
            "printed_output": captured_output.getvalue()[:20000],
            "variables": safe_vars,
            "error": None
        })
    except Exception as e:
        sys.stdout = old_stdout
        result_queue.put({
            "printed_output": captured_output.getvalue()[:20000],
            "variables": {},
            "error": str(e)
        })


def execute_with_timeout(code: str, timeout_seconds: int = 10, output_dir: str = None) -> Dict[str, Any]:
    """
    Execute code in a subprocess with timeout support.
    Works in both main thread and worker threads.

    Uses multiprocessing to run code in a separate process that can be
    forcefully terminated if it exceeds the timeout. This is the only
    reliable way to interrupt infinite loops or blocking operations.

    Parameters:
        code: The Python code to execute
        timeout_seconds: Maximum execution time in seconds
        output_dir: Optional working directory for the code

    Returns:
        dict with printed_output, variables, and optionally error
    """
    # Use 'spawn' context to avoid issues with forking in multithreaded environments
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()

    process = ctx.Process(
        target=_execute_code_in_process,
        args=(code, result_queue, output_dir)
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        # Timeout - forcefully terminate the process
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()  # Use SIGKILL if terminate (SIGTERM) wasn't enough
            process.join()
        return {"error": f"Execution timed out after {timeout_seconds} seconds"}

    # Get result from queue
    try:
        result = result_queue.get_nowait()
        return result
    except Exception:
        return {"error": "Failed to get result from subprocess"}


class Python_Coder_Tool(Tool):
    """
    Python Code Generator Tool - generates and executes Python code for various tasks.
    Python_Coder_Tool requires an LLM engine to generate code.
    """

    require_llm_engine = True

    def __init__(self, engine_name="gpt-4o", output_dir=None):
        """
        Initialize Python Code Generator Tool.

        Parameters:
            engine_name: Name of the engine to create (e.g., "claude-sonnet-4-0", "gpt-4o", "dashscope")
            output_dir: Optional workspace directory for file I/O operations
        """
        super().__init__(
            name=TOOL_NAME,
            description="A tool that generates and executes Python code snippets for data processing, calculations, and file operations. The generated code runs with access to Python standard library.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "A clear, specific description of the operation to perform, including any necessary inputs and expected outputs."
                }
            },
            output_schema={
                "printed_output": {
                    "type": "string",
                    "description": "The console output from executing the generated code."
                },
                "variables": {
                    "type": "object",
                    "description": "The local variables defined during code execution."
                },
                "execution_code": {
                    "type": "string",
                    "description": "The actual Python code that was executed."
                },
                "error": {
                    "type": "string",
                    "description": "Error message if execution failed.",
                    "optional": True
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES,
            documentation_path=None,
            llm=None  # Will be lazy loaded
        )

        # Store engine name for lazy loading
        self._engine_name = engine_name
        self._llm_engine = None
        self.output_dir = output_dir

    def _ensure_engine(self):
        """Lazy initialization of LLM engine."""
        if self._llm_engine is None:
            self._llm_engine = create_llm_engine(
                model_string=self._engine_name,
                is_multimodal=False,
                temperature=0.0
            )
            self.llm = self._llm_engine

    @staticmethod
    def preprocess_code(code):
        """
        Preprocesses the generated code snippet by extracting it from the response.
        Returns only the first Python code block found.

        Parameters:
            code (str): The response containing the code snippet.

        Returns:
            str: The extracted code snippet from the first Python block.

        Raises:
            ValueError: If no Python code block is found.
        """
        # Look for the first occurrence of a Python code block
        match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
        if not match:
            raise ValueError("No Python code block found in the response")
        return match.group(1).strip()

    def truncate_string(self, text, max_length):
        """
        Truncates a string using middle truncation if it exceeds max_length.

        Parameters:
            text (str): The text to truncate
            max_length (int): Maximum allowed length

        Returns:
            str: Truncated text with middle omission if needed
        """
        if len(text) <= max_length:
            return text

        # Keep first and last portions
        head_size = max_length // 2 - 50  # Leave room for truncation message
        tail_size = max_length // 2 - 50

        return (
            text[:head_size] +
            " ... (truncated: middle content omitted) ... " +
            text[-tail_size:]
        )

    def safe_repr(self, obj, max_length=20000):
        """
        Safely represent a variable with truncation for large objects.

        Parameters:
            obj: The object to represent
            max_length (int): Maximum length for representation

        Returns:
            str: Safe string representation of the object
        """
        try:
            # Handle special cases that can be extremely verbose
            import types

            # Skip function objects, modules, classes
            if isinstance(obj, (types.FunctionType, types.ModuleType, type)):
                return f"<{type(obj).__name__}: {getattr(obj, '__name__', 'unnamed')}>"

            # Handle itertools and other iterator objects
            if hasattr(obj, '__iter__') and hasattr(obj, '__next__'):
                return f"<iterator: {type(obj).__name__}>"

            # Convert to string and truncate if needed
            obj_str = str(obj)
            return self.truncate_string(obj_str, max_length)

        except Exception as e:
            return f"<repr error: {type(obj).__name__}>"

    @contextlib.contextmanager
    def capture_output(self):
        """
        Context manager to capture the standard output.

        Yields:
            StringIO: The captured output.
        """
        new_out = StringIO()
        old_out = sys.stdout
        sys.stdout = new_out
        try:
            yield sys.stdout
        finally:
            sys.stdout = old_out

    @contextlib.contextmanager
    def _maybe_chdir_to_output_dir(self):
        """
        Temporarily set CWD to `self.output_dir` (workspace) so relative file I/O
        naturally lands in the workspace.
        """
        old_cwd = os.getcwd()
        try:
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                os.chdir(self.output_dir)
            yield
        finally:
            try:
                os.chdir(old_cwd)
            except Exception:
                # If restoring fails, there's not much we can do safely here.
                pass

    def _extract_file_candidates(self, text: str) -> List[str]:
        """
        Extract candidate file paths from a natural-language query.
        Heuristic: look for quoted strings ending with common data extensions.
        """
        if not text:
            return []
        # Match quoted paths: '...json' or "...json" (also csv/tsv/txt)
        pattern = r"""['"]([^'"]+\.(?:json|jsonl|csv|tsv|txt))['"]"""
        return list(dict.fromkeys(re.findall(pattern, text, flags=re.IGNORECASE)))

    def _read_file_preview(self, path: str, max_bytes: int = 4096, max_lines: int = 80) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (preview_text, error). Preview is truncated to max_bytes/max_lines.
        """
        try:
            if not os.path.exists(path):
                return None, f"File not found: {path}"
            if not os.path.isfile(path):
                return None, f"Not a file: {path}"

            lines: List[str] = []
            read_bytes = 0
            with open(path, "rb") as f:
                for _ in range(max_lines):
                    chunk = f.readline()
                    if not chunk:
                        break
                    read_bytes += len(chunk)
                    # Decode best-effort
                    try:
                        lines.append(chunk.decode("utf-8", errors="replace"))
                    except Exception:
                        lines.append(str(chunk))
                    if read_bytes >= max_bytes:
                        break

            preview = "".join(lines)
            if read_bytes >= max_bytes:
                preview += "\n... (preview truncated)\n"
            return preview, None
        except Exception as e:
            return None, str(e)

    def _build_input_previews_block(self, query: str) -> str:
        """
        Build a prompt block that includes small previews of referenced input files
        so the LLM can infer schema without hard-coding assumptions.
        """
        candidates = self._extract_file_candidates(query)
        if not candidates:
            return ""

        # Resolve relative paths against workspace (output_dir) if set; otherwise CWD.
        base_dir = self.output_dir or os.getcwd()
        blocks: List[str] = []

        for raw in candidates[:3]:  # keep bounded
            resolved = raw
            if not os.path.isabs(resolved):
                resolved = os.path.join(base_dir, raw)

            preview, err = self._read_file_preview(resolved)
            if err:
                blocks.append(f"- {raw} -> {resolved} (preview error: {err})")
                continue

            # Include file size when available
            try:
                size = os.path.getsize(resolved)
                size_str = f"{size} bytes"
            except Exception:
                size_str = "unknown size"

            blocks.append(
                "----\n"
                f"FILE: {resolved} ({size_str})\n"
                "PREVIEW:\n"
                f"{preview}\n"
                "----"
            )

        if not blocks:
            return ""

        return (
            "\n\n"
            "Input file previews (for schema inference; do not assume structure beyond what you see):\n"
            + "\n".join(blocks)
            + "\n"
        )

    def execute_code_snippet(self, code, max_head_tail=20000, max_var_length=20000, max_vars=20):
        """
        Executes the given Python code snippet with truncation for large outputs.
        Uses multiprocessing for reliable timeout in both main and worker threads.

        Parameters:
            code (str): The Python code snippet to be executed.
            max_head_tail (int): Maximum length for printed output before truncation
            max_var_length (int): Maximum length for each variable representation
            max_vars (int): Maximum number of variables to include in output

        Returns:
            dict: A dictionary containing the printed output and local variables.
        """
        # Check for dangerous functions and remove them
        dangerous_functions = ['exit', 'quit', 'sys.exit']
        for func in dangerous_functions:
            if func in code:
                print(f"Warning: Removing unsafe '{func}' call from code")
                # Use regex to remove function calls with any arguments
                code = re.sub(rf'{func}\s*\([^)]*\)', 'pass', code)

        try:
            execution_code = self.preprocess_code(code)

            # Execute in subprocess with timeout (works in both main and worker threads)
            result = execute_with_timeout(
                execution_code,
                timeout_seconds=10,
                output_dir=self.output_dir
            )

            if result.get("error"):
                return {"error": result["error"], "execution_code": execution_code}

            # Truncate printed output using middle truncation
            printed_output = self.truncate_string(result.get("printed_output", ""), max_head_tail)

            # Filter and limit variables
            raw_vars = result.get("variables", {})
            used_vars = {}
            var_count = 0
            for k, v in raw_vars.items():
                if var_count >= max_vars:
                    used_vars["__truncated__"] = f"... ({len(raw_vars) - var_count} more variables omitted)"
                    break
                used_vars[k] = self.truncate_string(str(v), max_var_length)
                var_count += 1

            return {"printed_output": printed_output, "variables": used_vars, "execution_code": execution_code}

        except Exception as e:
            print(f"Error executing code: {e}")
            return {"error": str(e)}

    def _execute(self, query):
        """
        Execute the tool with given query.

        Parameters:
            query (str): A query describing the desired operation.

        Returns:
            Dict containing:
            - printed_output: The console output
            - variables: Local variables from execution
            - execution_code: The executed Python code
            - error: Error message if execution failed (optional)
        """
        if not query:
            return {"error": "Query parameter is required"}

        task_description = """
        Given a query, generate a Python code snippet that performs the specified operation on the provided data. Please think step by step. Ensure to break down the process into clear, logical steps. Make sure to print the final result in the generated code snippet with a descriptive message explaining what the output represents. This tool uses Python exec() to run code.

        Note: your working directory will be set to the tool workspace (output_dir) when executing, so relative file reads/writes will go there. If the final result is a .json file, print it out in a structured format (retain ALL details requested) and save it as a file.

        JSON shape guidance (common workflow handoff patterns):
        - If you load a JSON file and the top-level object is a dict, do NOT iterate it directly (iterating a dict yields string keys).
        - If the dict contains keys like "from_previous" or "previous_phase_outputs", those usually contain the payload you actually want.
        - If the top-level object is a list, iterate the list elements.
        - In batch-result lists, each element is often a dict like:
          {"item": {...}, "index": <int>, "outputs": {...}, "status": "..."}
          In that case, you typically want element["outputs"].

        File I/O guidance:
        - When absolute file paths are provided in the query, use them EXACTLY as given. Do NOT convert absolute paths to relative paths.
        - If the query provides an absolute output path (e.g., output_json_path/workspace_output_json.json), write exactly there. Include file extensions.
        - Otherwise, write relative outputs into the current working directory (the workspace).

        Import any libraries needed for the code to run, including math, json, csv, etc.

        The final output should be presented in the following format:

        ```python
        <code snippet>
        ```
        """
        task_description = task_description.strip()
        previews_block = self._build_input_previews_block(query)
        full_prompt = f"Task:\n{task_description}{previews_block}\n\nQuery:\n{query}"

        self._ensure_engine()
        response = self._llm_engine.generate(full_prompt)
        result_or_error = self.execute_code_snippet(response)

        return result_or_error
    
    def run(self, **kwargs):
        query = kwargs.get("query")

        return self._execute(query)


if __name__ == "__main__":
    """
    Test the Python Coder Tool:
    python scientist/tools/python_coder/tool.py
    """
    import os
    from scientist.tools.utilis import print_json, save_result

    # Initialize tool with engine name (uses lazy loading)
    tool = Python_Coder_Tool(engine_name="gpt-4o")

    examples = [
        {'query': 'Find the sum of prime numbers up to 50'},
        {'query': 'Given the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], calculate the sum of squares of odd numbers'},
    ]

    for example in examples:
        print(f"\n###Query: {example['query']}")
        result = tool.run(**example)
        print("\n###Execution Result:")
        print_json(result)

        # Save result
        query_name = example['query'].replace(' ', '_')[:50]
        save_result(result, query_name, os.path.join(os.path.dirname(__file__), 'test_logs'))
