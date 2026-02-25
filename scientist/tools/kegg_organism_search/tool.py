import os
import json
import openai
import numpy as np
import pickle as pkl

from scientist.tools.base_tool import Tool
from scientist.tools.utilis import remove_duplicates_nested

ALL_POSSIBLE_KEYS = ["ENTRY", "ORG_CODE", "NAME", "CATEGORY", "ANNOTATION", "TAXONOMY", "BRITE", "DATA_SOURCE", "KEYWORDS", "DISEASE", "COMMENT", "CHROMOSOME", "STATISTICS", "CREATED", "REFERENCE"]

DEFAULT_KEYS_TO_OUTPUT = ["ENTRY", "NAME", "CATEGORY", "BRITE", "DISEASE"]

from dotenv import load_dotenv
load_dotenv()

TOOL_NAME = "KEGG_Organism_Search_Tool"

LIMITATIONS = f"""
1. {TOOL_NAME} can only return the biomedical and medicine domain knowledge given an organism name or keyword.
2. {TOOL_NAME} will not return summarized answer, but the json format of queried information.
"""

BEST_PRACTICES = f"""
1. Choose {TOOL_NAME} when you need specific name or keyword about an organism.
"""

class KEGG_Organism_Search_Tool(Tool):
    def __init__(self):
        super().__init__(
            name=TOOL_NAME,
            description="A specialized tool for searching the KEGG database to find information about an organism (e.g., category, description, etc.) given an organism name.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The organism name or keyword to search in KEGG database (e.g., 'Haemophilus influenzae', 'Fruit fly')"
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
                    "description": "a list of related organisms, sorted by similarity. 1.0 means exact match, 0.0 means no match, and other values are the similarity score between the query and the organism."
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES
        )

        self.number_of_results = 5

        # self.embedding_model = "text-embedding-3-small"
        self.embedding_model = "text-embedding-3-large"

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Adjusted path to point to data directory relative to scientist/tools/kegg_organism_search
        project_root = os.path.abspath(os.path.join(current_file_dir, "../../.."))
        self.cache_dir = os.path.join(project_root, "data", "kegg_organism_database")

        # self.keys_to_output = [
        #     "ENTRY", "ORG_CODE", "NAME", "CATEGORY", "ANNOTATION", 
        #     "TAXONOMY", "BRITE", "DATA_SOURCE", "KEYWORDS", "DISEASE", 
        #     "COMMENT", "CHROMOSOME", "STATISTICS", "CREATED", "REFERENCE"]
        
        # self.keys_to_output = ["ENTRY", "ORG_CODE", "NAME", "CATEGORY", "DISEASE"]
        self.keys_to_output = DEFAULT_KEYS_TO_OUTPUT # NOTE default keys to output

        # Create OpenAI client instance
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Use global embedding store if available, otherwise load directly
        from scientist.utils.embedding_store import GlobalEmbeddingStore
        store = GlobalEmbeddingStore.get_instance()
        if store.is_initialized:
            self.embeddings = store.get_embeddings("kegg_organism_database")
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

    def _execute(self, query, keys_to_output=None):
        sims = []
        results = []

        # Use provided keys_to_output or fall back to default
        if not isinstance(keys_to_output, list) or len(keys_to_output) == 0:
            keys_to_output = self.keys_to_output

        # embed the query
        query_embedding = self.embed_text(query)    

        # for each organism, compute the cosine similarity between the query and the organism
        for organism_id, data in self.embeddings.items():
            organism_names = data["embedd_names"] # e.g. "SUPERGRP Non-Hodgkin lymphoma [DS:H02418]; B-cell acute lymphoblastic leukemia

            # calculate the similarity
            if query.lower().strip() in organism_names.lower():   # exact match
                similarity = 1.0
            else:
                embedding = data["embedding"] # a 3072-dimensional embedding vector
                embedding_text = data["embed_text"]
                similarity = round(self.cosine_similarity(query_embedding, embedding), 4)
            
            # append the similarity and organism id to the list
            sims.append((similarity, organism_id))

        # sort the organisms by the similarity
        sims = sorted(sims, key=lambda x: x[0], reverse=True)[:self.number_of_results]

        # for each organism, parse the organism file and format the output
        for sim, organism_id in sims:
            with open(os.path.join(self.cache_dir, "raw", f"{organism_id}.json")) as f:
                data = json.load(f)
            data = {k: v for k, v in data.items() if k in keys_to_output}
            # NOTE remove duplicates from nested lists while preserving order
            data = remove_duplicates_nested(data)
            # convert one-element list to a string
            data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in data.items()}
            results.append({"data": data, "similarity": sim})

        # NOTE if there are any exact matches (similarity 1.0), only return those
        # use epsilon comparison to handle floating point precision
        exact_matches = [result for result in results if result.get("similarity", 0) >= 0.9999]
        if exact_matches:
            results = exact_matches

        return results

    def run(self, **kwargs):
        query = kwargs["query"]
        keys_to_output = kwargs.get("keys_to_output", None)
        results = self._execute(query, keys_to_output)
        return {"status": "Success", "results": results}
    

if __name__ == "__main__":
    """
    Test the KEGG Organism Search Tool:
    python scientist/tools/kegg_organism_search/tool.py
    """
    import os
    import time
    from scientist.tools.utilis import print_json, save_result

    tool = KEGG_Organism_Search_Tool()

    examples = [
        {"query": "Haemophilus influenzae"},
        {"query": "Fruit fly", "keys_to_output": ["ENTRY", "NAME", "CATEGORY", "TAXONOMY"]},
    ]

    for example in examples:
        start_time = time.time()
        result = tool.run(**example)
        print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
        print_json(result)

        # save the result to a json file
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logs")
        save_result(result, example['query'], output_dir)
        print("")

    print("Done!")
