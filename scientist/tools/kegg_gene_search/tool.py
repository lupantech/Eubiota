import os
import json
import requests
from time import sleep

from scientist.tools.base_tool import Tool
from scientist.tools.utilis import remove_duplicates_nested

from dotenv import load_dotenv
load_dotenv()

# Tool name constant
TOOL_NAME = "KEGG_Gene_Search_Tool"

ALL_POSSIBLE_KEYS = ["symbol", "entry", "url", "name", "pathway", "brite", "module", "reaction", "dblinks", "genes", "disease", "journal", "sequence", "reference", "authors", "title"]

DEFAULT_KEYS_TO_OUTPUT = ["symbol", "entry", "url", "name", "pathway", "brite", "module", "reaction", "disease"]

LIMITATIONS = f"""
1. {TOOL_NAME} can only return the biomedical and medicine domain knowledge given a gene name.
2. {TOOL_NAME} will not return summarized answer, but the json format of queried information.
"""

BEST_PRACTICES = f"""
1. Choose {TOOL_NAME} when you need specific information about a gene.
2. Use specific gene names, such as dnaA, fusA, ruvA, ruvX for better results.
"""

class KEGG_Gene_Search_Tool(Tool):
    """
    KEGG Gene Search Tool - searches the KEGG database for gene information.
    Kegg_Gene_Search_Tool does not require an LLM engine.
    """

    require_llm_engine = False

    def __init__(self):
        """
        Initialize KEGG Gene Search Tool.
        Kegg_Gene_Search_Tool does not require an LLM engine.
        """
        super().__init__(
            name=TOOL_NAME,
            description="A specialized tool for searching the KEGG database to find biological pathways, genes, compounds, and other molecular information given a gene name.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The gene name to search in KEGG database (e.g., dnaA, fusA, ruvA, ruvX)"
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
                    "type": "dict",
                    "description": "The specific information about the gene query, including its symbol, name, Ko identifier, pathways, brite, and other information if available."
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES,
            llm=None
            # NOTE: Disable documentation for now as we've customized the tool function
            # documentation_path=f"{os.path.dirname(os.path.abspath(__file__)).split('library')[0]}/docs/KEGG.md"
        )
        self.base_url = "https://rest.kegg.jp"
        self.max_retries = 5
        # Use absolute path for cache directory relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.cache_dir = os.path.join(project_root, "data", "kegg_gene_database")

    @staticmethod
    def _parse_kegg_get_response(text):
        result = {}
        current_key = None
        for line in text.splitlines():
            if not line.strip() or line.strip() == "///":
                continue
            prefix = line[:12].strip() # e.g. "ENTRY", "NAME", "SYMBOL", "PATHWAY", "BRITE", "GENES", ""
            suffix = line[12:].strip() # e.g. "K02313", "dnaA; chromosomal replication initiator protein"
            if prefix:
                # this is the first line of the block
                key, value = prefix, suffix
                if key not in result:
                    result[key] = []
                result[key].append(value)
                current_key = key
            else:
                # this is the rest lines of the block
                value = suffix
                if current_key:
                    result[current_key].append(value)

        # Format the result
        for k, v in result.items():
            if len(v) == 1:
                result[k] = v[0] # convert one-element list to a string
        result = {k.lower(): v for k, v in result.items()} # convert all keys to lowercase
        return result

    @staticmethod
    def _find_matched_ko_id(query, find_response):
        """
        Find the matching KO ID for a query in the KEGG find response.
        Args:
            query (str): The gene symbol to search for (e.g. 'dnaA', 'accB')
            find_response (str): Raw response from KEGG find API, containing lines like:
                               'ko:K02313       dnaA; chromosomal replication initiator protein'
                               'ko:K02160       accB, bccP; acetyl-CoA carboxylase biotin carboxyl carrier protein'
        Returns:
            list: List of tuples containing (KO_ID, gene_symbols) for matching entries
                 e.g. [('K02313', ['dnaA'])], [('K02160', ['accB', 'bccP'])]
        """
        matching_entries = []
        for line in find_response.split("\n"):
            if line:
                # Split by tab and check if we have exactly 2 parts
                parts = line.split("\t")
                if len(parts) != 2:
                    # Skip malformed lines
                    continue
                ko_id, name = parts
                # Check if ko_id contains ':' before splitting
                if ":" not in ko_id:
                    continue
                ko_id = ko_id.strip().split(":")[1].strip() # e.g. K02313
                # Check if name contains ';' before splitting
                if ";" in name:
                    gene_id = [e.strip() for e in name.split(";")[0].strip().split(",")] # e.g. dnaA
                else:
                    # If no semicolon, use the whole name
                    gene_id = [e.strip() for e in name.strip().split(",")]
                if query in gene_id:
                    matching_entries.append((ko_id, gene_id))
        return matching_entries

    def _simplify_json_response(self, json_response, keys_to_output):
        # remove the KO suffix if present
        if json_response.get("entry"):
            # remove the KO suffix if present
            entry = json_response["entry"].strip()
            if entry.endswith("KO"):
                json_response["entry"] = entry[:-2].strip()

        # add url to the json_response
        if json_response.get("entry"):
            # e.g. K02313 -> https://www.genome.jp/entry/K02313
            json_response["url"] = f"https://www.genome.jp/entry/{json_response['entry']}"
        else:
            json_response["url"] = None

        # Filter the json_response based on keys_to_output
        cleaned_json_response = {k: json_response.get(k, None) for k in keys_to_output if k in json_response}

        # NOTE: remove duplicates from nested lists while preserving order
        cleaned_json_response = remove_duplicates_nested(cleaned_json_response)

        # NOTE: return the first 10 genes to save the space
        if "genes" in cleaned_json_response:
            cleaned_json_response["genes"] = cleaned_json_response["genes"][:10]

        return cleaned_json_response

    def _save_cache(self, query, status, results):
        # Replace the space with underscore, e.g. "xyn10D/fae1" -> "xyn10D_fae1"
        if "/" in query:
            query = query.replace("/", "_")

        if self.cache_dir is None:
            print(f"[KEGGGeneSearch Tool] Cache is disabled. Skipping the cache saving...")
            return

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = f"{self.cache_dir}/{query}.json"
        try:
            with open(cache_path, "w+") as cache_file:
                json.dump({"status": status, "results": results}, cache_file, indent=4)
            # print(f"[KEGGGeneSearch Tool] Save the cache successfully for {query}")
        except Exception as e:
            print(f"[KEGGGeneSearch Tool] Error: {str(e)}")

    def _load_cache(self, query):
        # Replace the space with underscore, e.g. "xyn10D/fae1" -> "xyn10D_fae1"
        if "/" in query:
            query = query.replace("/", "_")

        cache_path = f"{self.cache_dir}/{query}.json"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)
                    print(f"[KEGGGeneSearch Tool] Load the cache successfully for {query}")
                    return cache_data["status"], cache_data["results"]
            except Exception as e:
                print(f"[KEGGGeneSearch Tool] Error: {str(e)}")
                print("Load the cache failed. Searching the KEGG database instead...")
        return None, None

    def _execute(self, query, keys_to_output=None, database="ko"):
        base_url = self.base_url
        find_url = f"{base_url}/find/{database}/{query}"

        # Use provided keys_to_output or fall back to default
        if not isinstance(keys_to_output, list) or len(keys_to_output) == 0:
            keys_to_output = DEFAULT_KEYS_TO_OUTPUT

        # At first, try to load the cache
        status, results = self._load_cache(query)
        if status is not None:
            # Filter cached results based on keys_to_output
            filtered_results = []
            for result in results:
                if "result" in result:
                    filtered_result = {k: result["result"].get(k, None) for k in keys_to_output if k in result["result"]}
                    filtered_results.append({"match_type": result.get("match_type", "unknown"), "result": filtered_result})
                else:
                    filtered_results.append(result)
            return status, filtered_results

        # Find the target query in the KEGG database
        try:
            response = requests.get(find_url)
            attempt = 1
            # HTTP 200 means successful request. Retry if we get any other status code (e.g. 404 Not Found, 500 Server Error)
            while response.status_code != 200 and attempt <= self.max_retries:
                sleep(0.5*attempt)
                response = requests.get(find_url)
                attempt += 1
            find_response = response.text
            matching_entries = self._find_matched_ko_id(query, find_response)
        except Exception as e:
            print(f"[KEGGGeneSearch Tool] Error: {str(e)}")
            status = f"Failed to search KEGG database: {str(e)}"
            self._save_cache(query, status, [])
            return status, []

        # No results found
        if len(matching_entries) == 0:
            print(f"[KEGGGeneSearch Tool] No results found for {query}")
            status = "Success to search KEGG database, but no results found"
            self._save_cache(query, status, [])
            return status, []
        
        # Get the detailed information of the target query
        results = []
        for ko_id, gene_id in matching_entries:
            try:
                get_url = f"{base_url}/get/{ko_id}"
                response = requests.get(get_url)
                attempt = 1
                # HTTP 200 means successful request. Retry if we get any other status code (e.g. 404 Not Found, 500 Server Error)
                while response.status_code != 200 and attempt <= self.max_retries:
                    sleep(0.5*attempt)
                    response = requests.get(get_url)
                    attempt += 1
                get_response = response.text
                json_response = self._parse_kegg_get_response(get_response)
                simplified_json_response = self._simplify_json_response(json_response, keys_to_output)
            except Exception as e:
                print(f"[KEGGGeneSearch Tool] Error: {str(e)}")
                continue
            match_type = "exact" if len(gene_id) == 1 else "loose"
            results.append({"match_type": match_type, "result": simplified_json_response})

        # Save the result to the cache
        self._save_cache(query, "Success", results)
        return "Success", results

    def run(self, **kwargs):
        # Get the arguments
        query = kwargs["query"]
        keys_to_output = kwargs.get("keys_to_output", None)

        # Execute the tool
        status, results = self._execute(query, keys_to_output)

        # Format the result
        result = {
            "status": status,
            "results": results
        }
        return result

if __name__ == "__main__":
    """
    Test the KEGG Gene Search Tool:
    python scientist/tools/kegg_gene_search/tool.py
    """
    import os
    import time
    from scientist.tools.utilis import print_json, save_result

    tool = KEGG_Gene_Search_Tool()

    examples = [
        {'query': 'dnaA'}, # https://www.genome.jp/entry/K02313
        {'query': 'fusA'}, # https://www.genome.jp/entry/K02355, https://www.genome.jp/entry/K02355
        {'query': 'ruvA'}, # https://www.genzome.jp/entry/K03550
        {'query': 'ruvX'}, # https://www.genome.jp/entry/K07447
        {'query': 'accB'}, # https://www.genome.jp/entry/K02160
        {'query': 'acpP'}, # https://www.genome.jp/entry/K02078
        {'query': 'aat'}, # https://www.genome.jp/entry/K00684
        {'query': 'scfB'}, # No results found
        {'query': 'xyn10D/fae1'}, # No results found
        {'query': 'CACNA2D1'}, # https://www.genome.jp/entry/K00684
        {'query': 'ABO'}, # https://www.genome.jp/entry/K00709
        {'query': 'PSCA'}, # No results found
        {'query': 'TERT'}, # https://www.genome.jp/entry/K11126
        {'query': 'SULT1B1'}, # No results found
        {'query': 'dnaA', 'keys_to_output': ['symbol', 'entry', 'name', 'pathway']}, # Example with custom keys_to_output
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
