"""
Base Generator Tool - Adapted for unified Tool interface
Supports LLM engine configuration via engine_name
"""

import os
from scientist.tools.base_tool import Tool
from scientist.engine.factory import create_llm_engine

# Tool name constant
TOOL_NAME = "Base_Generator_Tool"

LIMITATION = f"""
The {TOOL_NAME} may provide hallucinated or incorrect responses.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Use it for general queries or tasks that don't require specialized knowledge or specific tools in the toolbox.
2. Provide clear, specific query.
3. Use it to answer the original query through step by step reasoning for tasks without complex or multi-step reasoning.
4. For complex queries, break them down into subtasks and use the tool multiple times.
5. Use it as a starting point for complex tasks, then refine with specialized tools.
6. Verify important information from its responses.
"""


class Base_Generator_Tool(Tool):
    """
    A generalized tool that takes query from the user as prompt,
    and answers the question step by step to the best of its ability.

    Requires LLM engine for operation.
    """

    require_llm_engine = True

    def __init__(self, engine_name="gpt-4o"):
        """
        Initialize Base Generator Tool.

        Args:
            engine_name: Name of the engine to create (e.g., "claude-sonnet-4-0", "gpt-4o", "dashscope", "Default")
        """
        super().__init__(
            name=TOOL_NAME,
            description="A generalized tool that takes query from the user as prompt, and answers the question step by step to the best of its ability.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The prompt that includes query from the user to guide the agent to generate response."
                }
            },
            output_schema={
                "response": {
                    "type": "string",
                    "description": "The generated output from the LLM"
                }
            },
            limitations=LIMITATION,
            best_practices=BEST_PRACTICE,
            llm=None  # Will be lazy loaded
        )

        # Store engine name for lazy loading
        self._engine_name = engine_name
        self._llm_engine = None

    def _ensure_engine(self):
        """Lazy initialization of LLM engine."""
        if self._llm_engine is None:
            self._llm_engine = create_llm_engine(
                model_string=self._engine_name,
                is_multimodal=False,
                temperature=0.0
            )
            self.llm = self._llm_engine

    def run(self, **kwargs):
        """
        Generate the output using the LLM.

        Parameters:
            query (str): The search query/prompt for the LLM.

        Returns:
            dict: A dictionary containing the response.
        """
        query = kwargs.get("query", "")

        try:
            self._ensure_engine()
            # Generate response using the LLM engine
            response = self._llm_engine(query, max_tokens=2048)
            return {"response": response}
        except Exception as e:
            return {"response": f"Error generating response: {str(e)}"}


if __name__ == "__main__":
    """
    Test the Base Generator Tool:
    python scientist/tools/base_generator/tool.py
    """
    from scientist.tools.utilis import print_json, save_result
    
    print("Testing Base Generator Tool...")

    tool = Base_Generator_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(metadata)
    print("\n" + "="*50 + "\n")

    # Sample queries for testing
    queries = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 50)

        try:
            result = tool.run(query=query)
            print("Result:")
            print_json(result)
            save_result(result, query, os.path.join(os.path.dirname(__file__), "test_logs"))
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Execution failed: {e}")
            print("\n" + "="*50 + "\n")

    print("Done!")
