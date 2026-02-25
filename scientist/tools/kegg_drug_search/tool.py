import os
import json
import openai
import numpy as np
import pickle as pkl
import copy  

from scientist.tools.base_tool import Tool
from scientist.tools.utilis import remove_duplicates_nested

from dotenv import load_dotenv
load_dotenv()

ALL_POSSIBLE_KEYS = ["ENTRY", "NAME", "PRODUCT", "FORMULA", "EXACT_MASS", "MOL_WEIGHT", "REMARK", "EFFICACY", "COMMENT", "BRITE", "DBLINKS", "ATOM", "BOND"]

DEFAULT_KEYS_TO_OUTPUT = ["ENTRY", "NAME", "PRODUCT", "FORMULA", "EFFICACY", "BRITE"]

TOOL_NAME = "KEGG_Drug_Search_Tool"

LIMITATIONS = f"""
1. {TOOL_NAME} can only return the biomedical and medicine domain knowledge given a drug name or keyword.
2. {TOOL_NAME} will not return summarized answer, but the json format of queried information.
"""

BEST_PRACTICES = f"""
1. Use {TOOL_NAME} when you want **structured drug facts from the local KEGG drug cache** (e.g., entry ID, formula, efficacy, BRITE category). This tool returns JSON-like fields, not a narrative summary.
2. Query with the **most specific drug identifier you have**:
   - Preferred: exact generic name (e.g., "meropenem"), brand name, or a known synonym.
   - If unsure: try multiple queries (generic name, brand name, common synonyms, spelling variants).
3. Keep queries short and “name-like”:
   - **semantic queries are allowed** (e.g., "antibiotics targeting Pseudomonas aeruginosa..."). In that case, interpret results as approximate and validate the returned `NAME`.
4. Interpret results correctly:
   - Results are the **top matches by embedding similarity**.
   - If an **exact name match** is detected, results are restricted to exact matches.
   - Always confirm the returned `NAME` matches your intended drug before using downstream.
5. If comparing multiple candidate drugs, run one query per drug and aggregate externally rather than putting multiple drugs into a single query.
"""

class KEGG_Drug_Search_Tool(Tool):
    def __init__(self):
        super().__init__(
            name=TOOL_NAME,
            description="A specialized tool for searching the KEGG database to find information about a drug (e.g., category, efficacy, etc.) given a drug name.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The drug name or keyword to search in KEGG database (e.g., 'Cephalexin', 'Latozinemab', 'Diethylamine')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of candidate drugs to return (use a larger number for broad/semantic queries). Default: 3.",
                    "optional": True
                },
                "keys_to_output": {
                    "type": "array",
                    "description": f"The keys to output. The default keys to output are {DEFAULT_KEYS_TO_OUTPUT}. You can choose to output more keys by providing a list of keys. (e.g., {ALL_POSSIBLE_KEYS})",
                    "enum": ALL_POSSIBLE_KEYS,
                    "optional": True
                }
            },
            output_schema={
                "status": {
                    "type": "string",
                    "description": "The status of the tool execution. If the tool executes successfully, the status will be 'Success'. Otherwise, the status will be error message."
                },
                "results": {
                    "type": "list",
                    "description": "a list of related drugs, sorted by similarity"
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES
        )
        self.number_of_results = 3
        
        
        self.max_list_items = 10       
        self.max_output_chars = 5000  

        self.embedding_model = "text-embedding-3-large"

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Adjusted path to point to data directory relative to scientist/tools/kegg_drug_search
        project_root = os.path.abspath(os.path.join(current_file_dir, "../../.."))
        self.cache_dir = os.path.join(project_root, "data", "kegg_drug_database")

        # self.keys_to_output = ["NAME", "EFFICACY"]
        self.keys_to_output = DEFAULT_KEYS_TO_OUTPUT # NOTE default keys to output

        # Create OpenAI client instance
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Use global embedding store if available, otherwise load directly
        from scientist.utils.embedding_store import GlobalEmbeddingStore
        store = GlobalEmbeddingStore.get_instance()
        if store.is_initialized:
            self.embeddings = store.get_embeddings("kegg_drug_database")
        else:
            # Fallback: load directly if store not initialized
            with open(os.path.join(self.cache_dir, "embeddings", f"{self.embedding_model}.pkl"), "rb") as f:
                self.embeddings = pkl.load(f)

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))   

    def embed_text(self, text):
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _truncate_data(self, data):
        """
        Helper function to smartly truncate data to save context tokens.
        """
        trunc_data = copy.deepcopy(data)
        
        for k, v in trunc_data.items():
            if isinstance(v, list) and len(v) > self.max_list_items:
                original_len = len(v)
                trunc_data[k] = v[:self.max_list_items]
                trunc_data[k].append(f"... ({original_len - self.max_list_items} more items truncated)")
            
            elif isinstance(v, str) and len(v) > 1000:
                trunc_data[k] = v[:1000] + "... (text truncated)"

 
        try:
            json_str = json.dumps(trunc_data, indent=2, ensure_ascii=False)
            if len(json_str) > self.max_output_chars:
                return json_str[:self.max_output_chars] + "\n... [Output truncated due to length limit]"
        except Exception:
            pass 

        return trunc_data

    def _execute(self, query, keys_to_output=None, max_results=None):
        sims = []
        results = []

        # Use provided keys_to_output or fall back to default
        if not isinstance(keys_to_output, list) or len(keys_to_output) == 0:
            keys_to_output = self.keys_to_output

        # Use provided max_results (clamped) or fall back to default
        if isinstance(max_results, int):
            max_results = max(1, min(20, max_results))
        else:
            max_results = self.number_of_results
        
        # embed the query
        query_embedding = self.embed_text(query)    

        # for each drug, compute the cosine similarity between the query and the drug
        for drug_id, data in self.embeddings.items():
            drug_names = data["embedd_names"] # e.g. Nicotinamide adenine dinucleotide; Nadide

            if query.lower().strip() in drug_names.lower():   # exact match
                similarity = 1.0
            else:    
                embedding = data["embedding"] # a 3072-dimensional embedding vector
                embedding_text = data["embed_text"]
                similarity = round(self.cosine_similarity(query_embedding, embedding), 4)
            sims.append((similarity, drug_id))

        # sort the drugs by the similarity
        sims = sorted(sims, key=lambda x: x[0], reverse=True)[:max_results]
        
        # for each drug, parse the drug file and format the output
        for sim, drug_id in sims:
            try:
                with open(os.path.join(self.cache_dir, "raw", f"{drug_id}.json")) as f:
                    data_raw = json.load(f)
                data = {k: v for k, v in data_raw.items() if k in keys_to_output}
                
                data = remove_duplicates_nested(data)
                data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in data.items()}
                
                # add truncation step
                final_data = self._truncate_data(data)
                
                results.append({"data": final_data, "similarity": sim})
            except Exception as e:
                print(f"Error processing {drug_id}: {e}")
                continue

        exact_matches = [result for result in results if result.get("similarity", 0) >= 0.9999]
        if exact_matches:
            results = exact_matches

        return results

    def run(self, **kwargs):
        query = kwargs["query"]
        keys_to_output = kwargs.get("keys_to_output", None)
        max_results = kwargs.get("max_results", None)
        results = self._execute(query, keys_to_output, max_results=max_results)
        return {"status": "Success", "results": results}
    

if __name__ == "__main__":
    """
    Test the KEGG Drug Search Tool:
    python scientist/tools/kegg_drug_search/tool.py
    """
    import os
    import time
    from scientist.tools.utilis import print_json, save_result

    tool = KEGG_Drug_Search_Tool()
    
    examples = [
        {"query": "Cephalexin"},
        {"query": "Goserelin acetate", "keys_to_output": ["NAME", "ATOM", "BOND"]}, 
        {"query": "Aspirin", "keys_to_output": ["ENTRY", "NAME", "FORMULA", "EFFICACY"]},
    ]

    print("Running KEGG Drug Search Tool (Context Safe Version)...")
    for example in examples:
        start_time = time.time()
        result = tool.run(**example)
        print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
        print_json(result)

        # save the result to a json file
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logs")
        save_result(result, example['query'], output_dir)
        print("-" * 60)

    print("Done!")
