# this is the 15 version (Hybrid RAG)
# It retrieves microbiota data, generates an answer using LLM (to be user-friendly),
# AND returns a truncated version of the raw data (to provide evidence while saving context).
# it is different from version 2 is that if there are exact name matches, it only returns those results
# created by qixin, verified by pan
"""
Drug-MB Interaction dataset: Accuracy: 65.85% (27/41)
- Planner: gpt-4o
- Other modules and tools: gpt-4o
- Toolset: 14 tools used
- Temperature: 0.0

Updated Best Practices as follows:
2. [Targeted Filtering] **CRITICAL**: To prevent context overflow, ALWAYS select specific `keys_to_output` relevant to your question. Do not request all keys unless necessary.
   - Question about **the effects of Drugs/Chemicals on this microbiota**? -> Use `['drug or other exogenous substances impact on microbiota (DEIM)']`
   - Question about **the association between this Microbiota and Diseases**? -> Use `['microbiota-disease associations (MBDA)']`
   - Question about **general Protein info** (protein name, gene name, UniProt AC, EC classification, tissue distribution, function)? -> Use `['general information of microbial protein']`
   - Question about **how Microbial Proteins modulate Drug Response** (e.g., enzymes metabolizing drugs, protein-drug interactions)? -> Use `['microbiota & their related proteins modulation on drug response (MMDR)']`
   - Or you may select other mixture of keys based on your question
"""
import os
import json
import openai
import pickle as pkl
import copy

from scientist.tools.base_tool import Tool
from scientist.tools.utilis import remove_keys_nested, cosine_similarity

from dotenv import load_dotenv
load_dotenv()

TOOL_NAME = "MDIPID_Microbe_Search_Tool"

ALL_POSSIBLE_KEYS = [
    "MIC_id",
    "general_information",
    "general information of microbial protein",
    "microbiota & their related proteins modulation on drug response (MMDR)",
    "drug or other exogenous substances impact on microbiota (DEIM)",
    "microbiota-disease associations (MBDA)"
]

DEFAULT_KEYS_TO_OUTPUT = [
    "MIC_id",
    "general_information",
    "microbiota-disease associations (MBDA)"
]

LIMITATIONS = f"""
1. [Scope] {TOOL_NAME} searches the MDIPID Microbiota database for information about specific microbes. It can ONLY search by **Microbe Name**
2. [Hybrid Output] Returns two distinct parts:
   - 'llm_answer': A natural language summary based on the database record.
   - 'evidence_data': A TRUNCATED snippet of the raw JSON for verification.
3. [Data Truncation] To save context window, 'evidence_data' is strictly truncated. **NEVER** assume 'evidence_data' is the complete list. If 'llm_answer' says there are 20 diseases but 'evidence_data' only shows 5, trust 'llm_answer'.
4. [Input Requirement] You MUST provide a specific `info_to_search` question. Vague inputs like "tell me about this" result in poor quality answers. The `query` parameter MUST be a specific taxon name (e.g., 'Bacteroides vulgatus'). Do NOT input drugs, diseases, or conditions as the query.
"""


BEST_PRACTICES = f"""
1. [Input Requirement] **`query` MUST be ONE specific Microbe Name.**
   - WRONG: query="deficiency of Ual-specificity phosphatases" (This is a condition, not a microbe)
   - WRONG: query=""Pedobacter sp., Clostridium indolis, Ruminococcus torques, Chlamydia sp." (This are too many names, search one by one)
   - CORRECT: query="unclassified Muribaculaceae" (Then ask about the condition in `info_to_search`)
   
2. [Targeted Filtering] **CRITICAL**: To prevent context overflow, ALWAYS select specific `keys_to_output` relevant to your question. Do not request all keys unless necessary.
   - Question about **the effects of Drugs/Chemicals on this microbiota**? -> Use `['drug or other exogenous substances impact on microbiota (DEIM)']`
   - Question about **the association between this Microbiota and Diseases**? -> Use `['microbiota-disease associations (MBDA)']`
   - Question about **general Protein info** (protein name, gene name, UniProt AC, EC classification, tissue distribution, function)? -> Use `['general information of microbial protein']`
   - Question about **how Microbial Proteins modulate Drug Response** (e.g., enzymes metabolizing drugs, protein-drug interactions)? -> Use `['microbiota & their related proteins modulation on drug response (MMDR)']`
   - Or you may select other mixture of keys based on your question

3. [Question Phrasing] The `info_to_search` parameter drives the internal reasoning. Phrasing it as a clear, standalone question yields the best results.
   - Bad: "drugs"
   - Good: "List all approved drugs that regulate the abundance of this microbiota."

4. [Handling Conflicts] If the 'evidence_data' seems to contradict or miss items mentioned in 'llm_answer', it is due to truncation. **Prioritize the information in 'llm_answer'** as it saw the untruncated data.
"""

QUERY_FORMAT = """
You are an expert biologist assistant.
Below is the retrieved information about a microbiota from the MDIPID database. 
Note: The data provided below is the relevant context retrieved for the user's query.

Microbiota Context:
{microbiota_info}

---
User Query:
Microbiota Name: {query}
Specific Question: {info_to_search}

Task:
Please answer the Specific Question concisely based ONLY on the Microbiota Context provided above. 
If the information is not present in the context, state that clearly.
"""

class MDIPID_Microbe_Search_Tool(Tool):
    """
    MDIPID Microbe Search Tool (Hybrid Version).
    Retrieves data, generates an answer, and returns truncated evidence.
    """

    require_llm_engine = True

    def __init__(self, engine_name="gpt-4o"):
        super().__init__(
            name=TOOL_NAME,
            description="A specialized tool for searching the MDIPID database. It returns a concise answer to your question AND provides truncated structured data as evidence. Useful for taxonomy, related proteins, drug interactions (MMDR/DEIM), and disease associations (MBDA).",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The microbe name or keyword to search (e.g., 'Phocaeicola vulgatus')."
                },
                "info_to_search": {
                    "type": "string",
                    "description": "The specific question to answer. REQUIRED. (e.g., 'What diseases is this associated with?')"
                },
                "keys_to_output": {
                    "type": "array",
                    "description": f"Optional keys to filter database. Default: {DEFAULT_KEYS_TO_OUTPUT}.",
                    "enum": ALL_POSSIBLE_KEYS,
                    "optional": True
                }
            },
            output_schema={
                "status": {
                    "type": "string",
                    "description": "Execution status."
                },
                "results": {
                    "type": "list",
                    "description": "List containing 'response' (LLM answer) and 'evidence_preview' (truncated JSON)."
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES,
            llm=None
        )

        self._engine_name = engine_name
        self._llm_engine = None
        
        # Search settings
        self.number_of_results = 3
        self.embedding_model = "text-embedding-3-large"
        
        # Truncation settings
        self.max_list_items = 5      # Only keep top k items in lists for display
        self.max_output_chars = 2000 # Hard limit for the final JSON string return

        # Path setup
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, "../../.."))
        self.cache_dir = os.path.join(project_root, "data", "mdipid_microbiota_database")
        
        self.keys_to_output = DEFAULT_KEYS_TO_OUTPUT
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Use global embedding store if available, otherwise load directly
        from scientist.utils.embedding_store import GlobalEmbeddingStore
        store = GlobalEmbeddingStore.get_instance()
        if store.is_initialized:
            self.embeddings = store.get_embeddings("mdipid_microbiota_database")
        else:
            # Fallback: Load Embeddings directly
            embedding_path = os.path.join(self.cache_dir, "embeddings", f"{self.embedding_model}.pkl")
            if os.path.exists(embedding_path):
                with open(embedding_path, "rb") as f:
                    self.embeddings = pkl.load(f)
            else:
                print(f"Warning: Embedding file not found at {embedding_path}.")
                self.embeddings = {}

    def _ensure_engine(self):
        """Lazy load LLM engine."""
        if self._llm_engine is None:
            from scientist.engine.factory import create_llm_engine
            self._llm_engine = create_llm_engine(
                model_string=self._engine_name,
                is_multimodal=False,
                temperature=0.0
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

    def _smart_truncate_data(self, data):
        """
        Smartly truncates the data dictionary for DISPLAY/OUTPUT purposes.
        1. Limits list lengths (e.g., only first 5 diseases).
        2. Enforces a hard character limit on the final string.
        """
        # Deep copy to avoid modifying the original data used for LLM generation
        trunc_data = copy.deepcopy(data)
        
        # Step 1: List Truncation
        for k, v in trunc_data.items():
            if isinstance(v, list) and len(v) > self.max_list_items:
                original_len = len(v)
                trunc_data[k] = v[:self.max_list_items]
                trunc_data[k].append(f"... ({original_len - self.max_list_items} more items hidden)")
        
        # Step 2: String Conversion & Hard Limit
        json_str = json.dumps(trunc_data, indent=2, ensure_ascii=False)
        if len(json_str) > self.max_output_chars:
            return json_str[:self.max_output_chars] + "\n... [Truncated...]"
        
        return json_str

    def _execute_search(self, query, keys_to_output=None):
        """Core search logic: Exact -> Partial -> Embedding."""
        sims = []
        
        if not isinstance(keys_to_output, list) or len(keys_to_output) == 0:
            keys_to_output = self.keys_to_output

        # Pass 1: Name Match
        exact_partial_matches = []
        for mic_id, data in self.embeddings.items():
            mic_name = data.get("embed_text", "") # Version B used 'embed_text', A used 'embedd_names', checking B standard
            if not mic_name and "embedd_names" in data: mic_name = data["embedd_names"] # Fallback

            if query.lower().strip() == mic_name.lower():
                exact_partial_matches.append((1.0, mic_id, "exact"))
            elif query.lower().strip() in mic_name.lower():
                exact_partial_matches.append((0.95, mic_id, "partial"))

        if exact_partial_matches:
            # Sort exact matches first
            sims = sorted(exact_partial_matches, key=lambda x: x[0], reverse=True)[:self.number_of_results]
        else:
            # Pass 2: Embedding Similarity
            query_embedding = self.embed_text(query)
            for mic_id, data in self.embeddings.items():
                embedding = data["embedding"]
                similarity = round(self._cosine_similarity(query_embedding, embedding), 4)
                sims.append((similarity, mic_id, "semantic"))
            
            sims = sorted(sims, key=lambda x: x[0], reverse=True)[:self.number_of_results]

        # Retrieve Data
        results = []
        for sim, mic_id, match_type in sims:
            json_path = os.path.join(self.cache_dir, "raw", f"{mic_id}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        full_data = json.load(f)
                    
                    # Clean up
                    full_data = remove_keys_nested(full_data, ["FASTA"]) # From Version B
                    
                    # Filter Keys
                    filtered_data = {k: v for k, v in full_data.items() if k in keys_to_output}
                    
                    results.append({
                        "id": mic_id,
                        "data": filtered_data,
                        "similarity": sim,
                        "match_type": match_type
                    })
                except Exception as e:
                    print(f"Error reading {mic_id}: {e}")
                    continue
        
        # Prioritize Exact Matches
        exact_matches = [result for result in results if result.get("similarity", 0) >= 0.9999]
        if exact_matches:
            results = exact_matches
        return results

    def run(self, **kwargs):
        query = kwargs.get("query")
        info_to_search = kwargs.get("info_to_search")
        keys_to_output = kwargs.get("keys_to_output", None)

        if not query:
            return {"status": "Error: 'query' is required", "results": []}
        
        if not info_to_search:
            return {"status": "Error: 'info_to_search' is required. Please ask a specific question.", "results": []}

        try:
            # 1. Retrieve Data (Full relevant context)
            search_results = self._execute_search(query, keys_to_output)
            
            if not search_results:
                return {"status": "Success", "results": [{"response": f"No microbiota found for query: {query}", "evidence_preview": "{}"}]}

            self._ensure_engine()
            final_results = []

            for res in search_results:
                full_data = res["data"]
                
                # 2. Internal RAG Generation
                # We use the FULL (or lightly filtered) data for the LLM to ensure accuracy.
                # Only if the data is insanely large (e.g. > 20k chars) do we truncate for the input prompt.
                context_str = json.dumps(full_data, indent=2)
                if len(context_str) > 20000: 
                    context_str = context_str[:20000] + "...(truncated for input)"

                prompt = QUERY_FORMAT.format(
                    microbiota_info=context_str,
                    query=query,
                    info_to_search=info_to_search
                )
                
                try:
                    llm_response = self.llm.generate(prompt)
                except Exception as e:
                    llm_response = f"Error generating answer: {str(e)}"

                # 3. Create Output (Truncated Evidence)
                # This is what goes back to the Agent's context history.
                evidence_preview = self._smart_truncate_data(full_data)

                final_results.append({
                    "microbe_id": res["id"],
                    "match_type": res["match_type"],
                    "llm_answer": llm_response,
                    "evidence_data": evidence_preview  # Controlled length (~1000 chars)
                })

            return {"status": "Success", "results": final_results}

        except Exception as e:
            return {"status": f"Error: {str(e)}", "results": []}

if __name__ == "__main__":
    """
    Test the MDIPID Microbe Search Tool:
    python scientist/tools/mdipid_microbe_search/tool.py
    """
    import time
    from scientist.tools.utilis import print_json, save_result

    print("="*80)
    print("MDIPID Microbe Search Tool - Tests")
    print("="*80)
    print("Note: info_to_search is a REQUIRED parameter")
    print("="*80)

    tool = MDIPID_Microbe_Search_Tool(engine_name="gpt-4o")
    print(f"Total microbes loaded: {len(tool.embeddings)}")

    examples = [
        {
            "query": "Veillonella sp.",
            "info_to_search": "Does Vitamin D supplementation decrease the abundance of this microbe?", # <--- Ask about drug association
            "keys_to_output": ["drug or other exogenous substances impact on microbiota (DEIM)"]
        },
        # {
        #     "query": "Phocaeicola vulgatus",
        #     "info_to_search": "What diseases are associated with this microbe?"
        # },
        # {
        #     "query": "Bacteroides fragilis",
        #     "info_to_search": "What drugs are known to affect this microbe?",
        #     "keys_to_output": ["drug or other exogenous substances impact on microbiota (DEIM)"]
        # },
        # {
        #     "query": "Escherichia coli",
        #     "info_to_search": "What proteins are related to this microbe?",
        #     "keys_to_output": ["general information of microbial protein", "microbes & their related proteins modulation on drug response (MMDR)"]
        # },
        # {
        #     "query": "fake_microbe_xyz",
        #     "info_to_search": "What is this microbe about?"
        # }
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

    print("Done!")
