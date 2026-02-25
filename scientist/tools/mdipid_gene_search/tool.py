import os
import json
from scientist.tools.base_tool import Tool
from scientist.tools.utilis import remove_keys_nested

ALL_POSSIBLE_KEYS = [
    "gene_id",
    "general information of microbial protein",
    "microbiota & their related proteins modulation on drug response (MMDR)"
]

DEFAULT_KEYS_TO_OUTPUT = ALL_POSSIBLE_KEYS

TOOL_NAME = "MDIPID_Gene_Search_Tool"

LIMITATIONS = f"""
1. {TOOL_NAME} searches the MDIPID Gene database using a predefined gene ID list.
2. {TOOL_NAME} always returns LLM-generated natural language answers based on retrieved data from the database.
3. {TOOL_NAME} requires a specific question (info_to_search) to be provided for each query.
4. {TOOL_NAME} only searches for exact gene name matches (case-sensitive).
5. The quality of the answer depends on the specificity and clarity of the question provided.
"""

BEST_PRACTICES = f"""
1. Use {TOOL_NAME} when you need to find information about specific genes (e.g., 'alr', 'fucI', 'recA').
2. Always provide a clear and specific question in the info_to_search parameter.
3. Good question examples:
   - "What drugs are associated with this gene?"
   - "What is the function of this gene?"
   - "What diseases are related to this gene?"
   - "What is the microbial protein encoded by this gene?"
4. Avoid vague questions like "Tell me about this gene" - be specific about what information you need.
5. The tool will retrieve relevant data from MDIPID and generate a concise answer using LLM.
6. Ensure you provide the exact gene symbol as it appears in the database (case-sensitive).
"""

QUERY_FORMAT = """
Gene Context:
{gene_info}

Gene Name: {query}
Information to Search: {info_to_search}

Please answer the question based on the gene information context provided above in a concise and informative manner.
"""

class MDIPID_Gene_Search_Tool(Tool):
    """
    MDIPID Gene Search Tool with optional LLM-based answer generation.

    Requires LLM engine when info_to_search is provided.
    """

    require_llm_engine = True

    def __init__(self, engine_name="gpt-4o"):
        """
        Initialize MDIPID Gene Search Tool.

        Args:
            engine_name: Name of the LLM engine to create (e.g., "gpt-4o", "gpt-4o-mini")
        """
        super().__init__(
            name=TOOL_NAME,
            description="A specialized tool for searching the MDIPID Gene database and generating LLM-based answers about gene information, including microbial protein details and drug response modulation (MMDR). Always returns concise natural language answers based on retrieved data.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The exact gene name or ID to search (e.g., 'alr', 'fucI', 'recA'). Case-sensitive."
                },
                "info_to_search": {
                    "type": "string",
                    "description": "The specific question or information to search regarding the gene. This is a required parameter. Examples: 'What drugs are associated with this gene?', 'What is the function of this gene?', 'What diseases are related to this gene?'"
                },
                "keys_to_output": {
                    "type": "array",
                    "description": f"Optional. The keys to filter from the database before sending to LLM. The default keys are {DEFAULT_KEYS_TO_OUTPUT}. You can specify specific keys to focus the answer on certain aspects.",
                    "enum": ALL_POSSIBLE_KEYS,
                    "optional": True
                }
            },
            output_schema={
                "status": {
                    "type": "string",
                    "description": "Execution status ('Success' or error message)."
                },
                "results": {
                    "type": "list",
                    "description": "A list of matched genes with their details."
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES,
            llm=None  # Will be lazy loaded
        )

        # Store engine name for lazy loading
        self._engine_name = engine_name
        self._llm_engine = None

        self.number_of_results = 3

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, "../../.."))
        self.cache_dir = os.path.join(project_root, "data", "mdipid_gene_database")

        self.keys_to_output = DEFAULT_KEYS_TO_OUTPUT

        # Use global embedding store if available, otherwise load directly
        from scientist.utils.embedding_store import GlobalEmbeddingStore
        store = GlobalEmbeddingStore.get_instance()
        if store.is_initialized:
            self.gene_id_list = store.get_gene_id_list()
            self.gene_data = store.get_gene_data()
        else:
            # Fallback: Load gene ID list directly
            gene_id_list_path = os.path.join(self.cache_dir, "gene_id_list.json")
            if os.path.exists(gene_id_list_path):
                with open(gene_id_list_path, 'r', encoding='utf-8') as f:
                    self.gene_id_list = json.load(f)
            else:
                print(f"Warning: Gene list file not found at {gene_id_list_path}")
                self.gene_id_list = []

            # Load all gene data
            self.gene_data = {}
            gene_raw_dir = os.path.join(self.cache_dir, "raw")
            for gene_id in self.gene_id_list:
                gene_file_path = os.path.join(gene_raw_dir, f"{gene_id}.json")
                if os.path.exists(gene_file_path):
                    with open(gene_file_path, 'r', encoding='utf-8') as f:
                        self.gene_data[gene_id] = json.load(f)

    def _ensure_engine(self):
        """Lazy initialization of LLM engine."""
        if self._llm_engine is None:
            from scientist.engine.factory import create_llm_engine
            self._llm_engine = create_llm_engine(
                model_string=self._engine_name,
                is_multimodal=False,
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            self.llm = self._llm_engine

    def _pre_process_gene_data(self, data):
        """Process the gene data to remove unnecessary keys"""
        keys_to_remove = ["FASTA"]
        return remove_keys_nested(data, keys_to_remove)

    def _execute(self, query, keys_to_output=None):
        results = []
        if not isinstance(keys_to_output, list) or len(keys_to_output) == 0:
            keys_to_output = self.keys_to_output

        # Check for exact match
        if query in self.gene_id_list:
            gene_data = self.gene_data.get(query, {})

            if gene_data:
                # Pre-process to remove unnecessary keys
                gene_data = self._pre_process_gene_data(gene_data)

                # Filter by requested keys
                filtered_data = {k: v for k, v in gene_data.items() if k in keys_to_output}

                results.append({"data": filtered_data, "similarity": 1.0, "matching_type": "exact match"})
        else:
            # No match found
            results.append({"data": {}, "similarity": 0.0, "matching_type": "no match"})

        return results

    def _answer_info_to_search(self, query, info_to_search, results):
        """
        Process retrieved gene data and generate LLM answers.

        Args:
            query: The gene name query
            info_to_search: The information/question to search (required)
            results: List of retrieved gene data

        Returns:
            List of results with LLM-generated answers
        """
        # Ensure LLM engine is loaded
        self._ensure_engine()

        processed_results = []
        for result in results:
            gene_data = result.get("data", {})
            similarity = result.get("similarity", 0)
            matching_type = result.get("matching_type", "no match")

            # If no gene data found, skip LLM generation
            if not gene_data:
                processed_results.append({
                    "query": query,
                    "info_to_search": info_to_search,
                    "similarity": similarity,
                    "matching_type": matching_type,
                    "response": "No gene data found. The gene may not exist in the MDIPID database."
                })
                continue

            # Format the context with the gene data
            context_str = json.dumps(gene_data, indent=2)
            
            # if context is too long, truncate it
            if len(context_str) > 20000: 
                    context_str = context_str[:20000] + "...(truncated for input)"


            # Create the prompt using QUERY_FORMAT
            prompt = QUERY_FORMAT.format(
                gene_info=context_str,
                query=query,
                info_to_search=info_to_search
            )

            # Generate the answer using LLM
            try:
                llm_response = self.llm.generate(prompt)
            except Exception as e:
                llm_response = f"Error generating answer: {str(e)}"

            processed_results.append({
                "query": query,
                "info_to_search": info_to_search,
                "similarity": similarity,
                "matching_type": matching_type,
                "response": llm_response
            })

        return processed_results

    def run(self, **kwargs):
        query = kwargs.get("query")
        info_to_search = kwargs.get("info_to_search")
        keys_to_output = kwargs.get("keys_to_output", None)

        # Validate required parameters
        if not query:
            return {"status": "Error: 'query' parameter is required", "results": []}

        if not info_to_search:
            return {"status": "Error: 'info_to_search' parameter is required. Please provide a specific question about the gene.", "results": []}

        try:
            # Retrieve gene data
            results = self._execute(query, keys_to_output)

            if not results:
                return {"status": "Success", "results": [], "message": f"No gene found matching '{query}'"}

            # Process results with LLM answer generation
            results = self._answer_info_to_search(query, info_to_search, results)

            return {"status": "Success", "results": results}
        except Exception as e:
            return {"status": f"Error: {str(e)}", "results": []}

if __name__ == "__main__":
    """
    Test the MDIPID Gene Search Tool:
    python scientist/tools/mdipid_gene_search/tool.py
    """
    import time
    from scientist.tools.utilis import print_json, save_result

    print("="*80)
    print("MDIPID Gene Search Tool - Tests")
    print("="*80)
    print("Note: info_to_search is now a REQUIRED parameter")
    print("="*80)

    tool = MDIPID_Gene_Search_Tool(engine_name="gpt-4o")
    print(f"Total genes loaded: {len(tool.gene_id_list)}")

    examples = [
        {
            "query": "alr",
            "info_to_search": "What drugs are associated with the alr gene?"
        },
        {
            "query": "fucI",
            "info_to_search": "What is the function of the fucI gene?"
        },
        {
            "query": "recA",
            "info_to_search": "What diseases are associated with recA?"
        },
        {
            "query": "bla",
            "info_to_search": "What is the microbial protein about the gene named bla?",
            "keys_to_output": ["gene_id", "general information of microbial protein"]
        },
        {
            "query": "fake_gene",
            "info_to_search": "What is this gene about?"
        }
    ]

    for example in examples:
        print(f"\nQuerying: {example['query']}")
        print(f"Question: {example['info_to_search']}")
        start_time = time.time()
        result = tool.run(**example)
        print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
        print_json(result)

        # save the result to a json file
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logs")
        save_result(result, example['query'], output_dir)
        print("-"*80)

    print("\n" + "="*80)
    print("All Tests Completed!")
    print("="*80) 