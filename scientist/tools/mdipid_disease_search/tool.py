import os
import json
import openai
import numpy as np
import pickle as pkl

from scientist.tools.base_tool import Tool
from scientist.tools.utilis import remove_keys_nested, cosine_similarity

from dotenv import load_dotenv
load_dotenv()

ALL_POSSIBLE_KEYS = [
    "disease_id", 
    "disease_name", 
    "general_information", 
    "General information of associated drug and substance", 
    "Microbiota-disease associations (MBDA)"
]

DEFAULT_KEYS_TO_OUTPUT = [
    "disease_name", 
    "general_information", 
    "General information of associated drug and substance", 
    "Microbiota-disease associations (MBDA)"
]

TOOL_NAME = "MDIPID_Disease_Search_Tool"

LIMITATIONS = f"""
1. {TOOL_NAME} searches the MDIPID database for disease-drug and disease-microbiota associations.
2. {TOOL_NAME} always returns LLM-generated natural language answers based on retrieved data from the database.
3. {TOOL_NAME} requires a specific question (info_to_search) to be provided for each query.
4. The quality of the answer depends on the specificity and clarity of the question provided.
"""

BEST_PRACTICES = f"""
1. Use {TOOL_NAME} when you need to find specific drugs, substances, or microbiota associated with a disease.
2. Always provide a clear and specific question in the info_to_search parameter.
3. Good question examples:
   - "What drugs could be used to treat this disease?"
   - "What microbiota are associated with this disease?"
   - "What is the relationship between this disease and gut microbiome?"
4. Avoid vague questions like "Tell me about this disease" - be specific about what information you need.
5. The tool will retrieve relevant data from MDIPID and generate a concise answer using LLM.
"""

QUERY_FORMAT = """
Disease Context:
{disease_info}

Disease Name: {query}
Information to Search: {info_to_search}

Please answer the question based on the disease information context provided above in a concise and informative manner.
"""

class MDIPID_Disease_Search_Tool(Tool):
    """
    MDIPID Disease Search Tool with optional LLM-based answer generation.

    Requires LLM engine when info_to_search is provided.
    """

    require_llm_engine = True

    def __init__(self, engine_name="gpt-4o"):
        """
        Initialize MDIPID Disease Search Tool.

        Args:
            engine_name: Name of the LLM engine to create (e.g., "gpt-4o", "gpt-4o-mini")
        """
        super().__init__(
            name=TOOL_NAME,
            description="A specialized tool for searching the MDIPID database and generating LLM-based answers about disease information, associated drugs, and microbiota-disease associations. Always returns concise natural language answers based on retrieved data.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The disease name or keyword to search (e.g., 'Allergy', 'Diabetes', 'Colorectal cancer')."
                },
                "info_to_search": {
                    "type": "string",
                    "description": "The specific question or information to search regarding the disease. This is a required parameter. Examples: 'What drugs could be used to treat this disease?', 'What microbiota are associated with this disease?', 'What is the relationship between this disease and the gut microbiome?'"
                },
                "keys_to_output": {
                    "type": "array",
                    "description": f"Optional. The keys to filter from the database before sending to LLM. The default keys are {DEFAULT_KEYS_TO_OUTPUT}. You can specify keys like 'Microbiota-disease associations (MBDA)' to focus the answer on specific aspects.",
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
                    "description": "A list of matched diseases with their associated drugs and microbiota info."
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

        self.embedding_model = "text-embedding-3-large"

        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        project_root = os.path.abspath(os.path.join(current_file_dir, "../../.."))
        self.cache_dir = os.path.join(project_root, "data", "mdipid_disease_database")

        self.keys_to_output = DEFAULT_KEYS_TO_OUTPUT

        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Use global embedding store if available, otherwise load directly
        from scientist.utils.embedding_store import GlobalEmbeddingStore
        store = GlobalEmbeddingStore.get_instance()
        if store.is_initialized:
            self.embeddings = store.get_embeddings("mdipid_disease_database")
        else:
            # Fallback: load directly if store not initialized
            embedding_path = os.path.join(self.cache_dir, "embeddings", f"{self.embedding_model}.pkl")
            if os.path.exists(embedding_path):
                with open(embedding_path, "rb") as f:
                    self.embeddings = pkl.load(f)
            else:
                print(f"Warning: Embedding file not found at {embedding_path}")
                self.embeddings = {}

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

    def _cosine_similarity(self, a, b):
        return cosine_similarity(a, b)

    def embed_text(self, text):
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _pre_process_disease_data(self, data):
        """Process the disease data to remove unnecessary keys"""
        keys_to_remove = ["FASTA"]
        return remove_keys_nested(data, keys_to_remove)

    def _execute(self, query, keys_to_output=None):
        sims = []
        results = []

        if not isinstance(keys_to_output, list) or len(keys_to_output) == 0:
            keys_to_output = self.keys_to_output

        # [1] First pass: check for exact and partial matches
        exact_partial_matches = []
        for disease_id, data in self.embeddings.items():
            disease_name = data.get("embed_text", "")

            if query.lower().strip() == disease_name.lower():  # exact match
                exact_partial_matches.append((1.0, disease_id, "exact match"))
            elif query.lower().strip() in disease_name.lower():  # partial match
                exact_partial_matches.append((1.0, disease_id, "partial match"))

        # If exact or partial matches found, use those; otherwise use similarity search
        if exact_partial_matches:
            sims = exact_partial_matches
        else:
            # [2] Second pass: use embedding similarity search
            query_embedding = self.embed_text(query)

            for disease_id, data in self.embeddings.items():
                embedding = data["embedding"]
                similarity = round(self._cosine_similarity(query_embedding, embedding), 4)
                sims.append((similarity, disease_id, "similar match"))

        # Sort the diseases by similarity
        sims = sorted(sims, key=lambda x: x[0], reverse=True)[:self.number_of_results]

        # For each disease, parse the disease file and format the output
        for sim, disease_id, matching_type in sims:
            json_path = os.path.join(self.cache_dir, "raw", f"{disease_id}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)

                # Pre-process to remove unnecessary keys
                full_data = self._pre_process_disease_data(full_data)

                # Filter by requested keys
                filtered_data = {k: v for k, v in full_data.items() if k in keys_to_output}

                results.append({"data": filtered_data, "similarity": sim})

        # NOTE: if there are any exact matches (similarity 1.0), only return those
        exact_matches = [result for result in results if result.get("similarity", 0) >= 0.9999]
        if exact_matches:
            results = exact_matches

        return results

    def _answer_info_to_search(self, query, info_to_search, results):
        """
        Process retrieved disease data and generate LLM answers.

        Args:
            query: The disease name query
            info_to_search: The information/question to search (required)
            results: List of retrieved disease data

        Returns:
            List of results with LLM-generated answers
        """
        # Ensure LLM engine is loaded
        self._ensure_engine()

        processed_results = []
        for result in results:
            disease_data = result.get("data", {})
            similarity = result.get("similarity", 0)

            # Format the context with the disease data
            context_str = json.dumps(disease_data, indent=2)

            # if context is too long, truncate it
            if len(context_str) > 20000: 
                    context_str = context_str[:20000] + "...(truncated for input)"

            # Create the prompt using QUERY_FORMAT
            prompt = QUERY_FORMAT.format(
                disease_info=context_str,
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
            return {"status": "Error: 'info_to_search' parameter is required. Please provide a specific question about the disease.", "results": []}

        try:
            # Retrieve disease data
            results = self._execute(query, keys_to_output)

            # Process results with LLM answer generation
            results = self._answer_info_to_search(query, info_to_search, results)

            return {"status": "Success", "results": results}
        except Exception as e:
            return {"status": f"Error: {str(e)}", "results": []}

if __name__ == "__main__":
    """
    Test the MDIPID Disease Search Tool:
    python scientist/tools/mdipid_disease_search/tool.py
    """
    import time
    from scientist.tools.utilis import print_json, save_result

    print("="*80)
    print("MDIPID Disease Search Tool - Tests")
    print("="*80)
    print("Note: info_to_search is now a REQUIRED parameter")
    print("="*80)

    tool = MDIPID_Disease_Search_Tool(engine_name="gpt-4o")

    examples = [
        {
            "query": "Allergy",
            "info_to_search": "What drugs could be used to treat Allergy?"
        },
        {
            "query": "Type 2 diabetes",
            "info_to_search": "What microbes are associated with Type 2 diabetes?"        },
        {
            "query": "Inflammatory bowel disease",
            "info_to_search": "What is the microbial association with inflammatory bowel disease?",
            "keys_to_output": ["disease_name", "Microbiota-disease associations (MBDA)"]
        },
        {
            "query": "Asthma",
            "info_to_search": "What are the main microbial changes associated with Asthma?"
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