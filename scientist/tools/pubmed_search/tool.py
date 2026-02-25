# this is the the after merging version of PubMed Search Tool
# the best practices are well defined
from Bio import Entrez, Medline
from io import StringIO
import os
import re
import time
from urllib.error import HTTPError, URLError
import threading

from scientist.tools.base_tool import Tool

from dotenv import load_dotenv
load_dotenv()

# Tool name constant
TOOL_NAME = "PubMed_Search_Tool"

# Limit concurrent Entrez requests to avoid "Too many open files"
# urllib opens a new connection for each request, so limit concurrency strictly.
ENTREZ_SEMAPHORE = threading.BoundedSemaphore(3) # 3 for a free api key

LIMITATIONS = f"""
1. {TOOL_NAME} is limited to the PubMed database (https://pubmed.ncbi.nlm.nih.gov/).
2. Extremely specific combinations (e.g., long lists of species/strains/genes plus model plus treatment, chains of AND statements) will not return results. Split these up into individual queries.
3. Results may include non-peer-reviewed or less relevant items; always verify relevance and quality.
4. {TOOL_NAME} returns a summary synthesized from the supplied abstracts and their citations in the format of [number] PMID.
"""

BEST_PRACTICES = f"""
1. Choosing {TOOL_NAME} suits for searching general information about biomedical topics.
2. Do NOT enumerate many species/strains/genes. ALWAYS keep queries below 3 topics. 1-2 topics will work the best.
3. Bag-of-words query is better than sentence query.
4. Use AND operator to combine multiple keywords. For example,
GOOD: "microbiome AND inflammation", "gut microbiome"
BAD: "microbiome gut inflammation"
GOOD: "microbiome AND DSS colitis"
BAD: "microbiome gut inflammation DSS colitis"
5. Check the abstract of the publication to determine if it is relevant to the query after the search.
6. For complex questions (e.g., condition + model + intervention, taxa/gene family), search each axis separately (e.g., "influenza AND ferret", "probiotic therapy", "Lactobacillus AND influenza")  and combine downstream. DO NOT use queries like "lactobacillus acidophilus lactobacillus kefiranofaciens lactobacillus gasseri AND influenza"; this should be split into "lactobacillus acidophilus AND influenza", "lactobacillus kefiranofaciens AND influenza", and "lactobacillus gasseri AND influenza".
7. If a query returns 0 results, drop modifiers (e.g., "single-cell", "patient-derived xenograft"), remove species/strain lists, and retry broader terms before trying other tools.
8. After retrieving results, scan abstracts to confirm biological context (organism, tissue, assay) and relevance.
"""

SUMMARIZE_PROMPT_TEMPLATE = """
You are an AI expert analyst specializing in biomedical literature. Your task is to provide a concise, evidence‑grounded answer to the user's query based **only** on the supplied publications.

## User Query
{keywords}

## Available Publications
{formatted_publications}

## Instructions
Follow these steps carefully:

1.  **Relevance Analysis:** For each publication, critically evaluate if its core findings, methodology, or conclusions are directly relevant to the user's query. A mere mention of a keyword is not sufficient for relevance.

2.  **Information Synthesis:** From the **relevant publications only**, extract and synthesize the key information that directly answers the query. The summary must be cohesive and based exclusively on the provided text. Do not invent information or draw conclusions not explicitly supported by the sources.

3. **Cite Precisely:** After each fact or statement, cite the source in brackets—e.g., "[1]" or "[2]". If multiple sources support one point, combine them like "[1,2]".

## Output
A brief, well‑structured summary of the publications that directly answers the query, with in‑text citations.
"""

class PubMed_Search_Tool(Tool):
    """
    PubMed Search Tool - searches the PubMed database for biomedical literature.
    Pubmed_Search_Tool requires an LLM engine for summarizing publications.
    """

    require_llm_engine = True

    def __init__(self, engine_name="gpt-4o"):
        """
        Initialize PubMed Search Tool.

        Args:
            engine_name: Name of the engine to create (e.g., "gpt-4o-mini", "dashscope", "Default")
        """
        super().__init__(
            name=TOOL_NAME,
            description="A comprehensive tool for searching the PubMed database, the primary database for biomedical and life sciences literature.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The search query for biomedical and life sciences publications"
                },
                "journals": {
                    "type": "array", 
                    "description": "List of journal names to search within. Can be a single journal name or multiple journals. Use the journal's common name (e.g., 'Nature', 'Science', 'Cell', 'Microbiome'). If not specified, the search spans all journals. Maximum suggested number of journals: 10. Note: The journal names are case-sensitive.",
                    "optional": True
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of results to return. Default: 20",
                    "optional": True
                }
            },
            output_schema={
                "type": "object",
                "description": "Dictionary with journal names as keys, each containing summary and citations",
                "properties": {
                    "journal_name": {
                        "type": "object",
                        "properties": {
                            "term_used": {
                                "type": "string",
                                "description": "The final search term used to retrieve the publications"
                            },
                            "summary": {
                                "type": "string",
                                "description": "The summary of the publications that directly answers the query"
                            },
                            "citations": {
                                "type": "array",
                                "description": "The citations of the publications, in the format of [number] PMID"
                            }
                        }
                    }
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES,
            llm=None  # Will be lazy loaded
            # NOTE: Disable documentation for now as we've customized the tool function
            # documentation_path=f"{os.path.dirname(os.path.abspath(__file__)).split('library')[0]}/docs/PubMed.md"
        )

        # Store engine name for lazy loading
        self._engine_name = engine_name
        self._llm_engine = None

        self.max_results = 20
        self.entrez_email = "test@gmail.com" # NOTE: Use Bingxuan's Free account email for now. TODO: Upgrade to a subscription email
        self.entrez_tool = "PubMedQueryTool"
        self.max_journals = 10 # Maximum number of journals to search

        self.entrez_api_key = os.getenv("NCBI_API_KEY") # API Key for E-utils

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

    def _with_retry(self, call, max_attempts=5, delay=1.01):
        last_err = None
        for attempt in range(max_attempts):
            try:
                return call()
            except Exception as e:
                last_err = e
                # Retry on any exception (HTTP 429, 5xx, Network Error, etc.)
                sleep_time = delay * (attempt + 1)
                print(f"PubMed API error: {str(e)} (attempt {attempt+1}/{max_attempts}), sleeping {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                continue
        if last_err:
            raise last_err
        raise RuntimeError("Retry operation failed with no captured exception")

    def _search_single_journal(self, keywords, journal, max_results):
        """Search for publications in a single journal"""
        try:
            # Set the Entrez account
            Entrez.email = self.entrez_email
            Entrez.tool = self.entrez_tool
            if self.entrez_api_key:
                Entrez.api_key = self.entrez_api_key

            # Build search query
            search_keywords = keywords
            if journal:
                search_keywords = f"{keywords} AND {journal}[Journal]"

            def _do_esearch(query):
                with ENTREZ_SEMAPHORE:
                    with Entrez.esearch(db="pubmed", term=query, retmax=max_results) as search_handle:
                        return Entrez.read(search_handle)
        
            # Get the search results
            search_results = self._with_retry(lambda: _do_esearch(search_keywords))

            # Get the ID list
            id_list = search_results["IdList"]

            # If no results and query contains spaces, retry with " AND " instead of " "
            if not id_list and " " in keywords and " AND " not in keywords:
                retry_keywords = keywords.replace(" ", " AND ")
                if journal:
                    retry_keywords = f"{retry_keywords} AND {journal}[Journal]"
                
                # Get the search results
                print(f"Retry search with added AND operator: keywords: {keywords} -> {retry_keywords}")
                search_results = self._with_retry(lambda: _do_esearch(retry_keywords))

                # Get the ID list
                id_list = search_results["IdList"]

                # Update the search keywords
                search_keywords = retry_keywords

            if not id_list:
                return {
                    "term_used": search_keywords,
                    "summary": f"No publications found. Please try again with different keywords or journal.",
                    "citations": []
                }

            def _do_efetch():
                with ENTREZ_SEMAPHORE:
                    with Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text") as fetch_handle:
                        return fetch_handle.read()
            records_text = self._with_retry(_do_efetch)
            records = list(Medline.parse(StringIO(records_text)))
            
            # Format the publications
            publications = []
            for record in records:
                pub_info = {
                    "Title": record.get("TI", "N/A"),
                    "Authors": record.get("AU", []),
                    "Abstract": record.get("AB", "N/A"),
                    "Journal": record.get("JT", "N/A"),
                    "Publication Date": record.get("DP", "N/A"),
                    "PMID": record.get("PMID", "N/A"),
                    "Source": record.get("SO", "N/A"),
                }
                publications.append(pub_info)

            # Filter out the publications that do not have an abstract
            publications = [
                pub for pub in publications if pub["Abstract"] != "N/A"]

            # Filter out the publications that have the same title
            unique_publications = []
            seen_titles = set()
            for pub in publications:
                title = pub["Title"].lower().strip()
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_publications.append(pub)
            unique_publications.sort(
                key=lambda x: x["Publication Date"], reverse=True)
            publications = unique_publications[:max_results]

            # Format the citations
            citations = []
            for i, pub in enumerate(publications):
                citations.append(f"[{i+1}] PMID: {pub['PMID']}")

            # Format the publications for the summary
            formatted_publications = ""
            for i, pub in enumerate(publications):
                text = f"""
            Publication Abstract ([{i+1}] PMID: {pub['PMID']}):
            {pub['Abstract']}
            """
                formatted_publications += text

            # Ensure LLM engine is initialized
            self._ensure_engine()

            # Summarize the publications
            summary_prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
                keywords=keywords, 
                formatted_publications=formatted_publications
            )
            summary = self._llm_engine(summary_prompt, max_tokens=2048)

            # Return the results
            return {
                "term_used": search_keywords,
                "summary": summary,
                "citations": citations
            }
        except Exception as e:
            return {
                "term_used": search_keywords,
                "summary": f"Error occurred during PubMed search: {str(e)}",
                "citations": []
            }

    def _initialize_journal_list(self, journals):
        """Initialize journal list with error handling, allow empty list for all-journal searches"""
        try:
            # Try to parse journals parameter
            if journals is None or not journals:
                # No journals specified; search across all journals
                return []
            
            if isinstance(journals, list):
                # Validate that all items are strings
                journal_list = [str(j).strip() for j in journals if j and str(j).strip()]
                if journal_list:
                    journal_list = journal_list[:self.max_journals]
                    return journal_list
                else:
                    # no journals found; search across all journals
                    return []
            elif isinstance(journals, str):
                # Single journal as string, convert to list
                journal = journals.strip()
                if journal:
                    return [journal]
                else:
                    return []
            else:
                # Invalid type, fall back to searching across all journals
                print(f"Warning: Invalid journal type '{type(journals)}', searching across all journals")
                return []
                
        except Exception as e:
            # Any parsing error, fall back to all journals
            print(f"Warning: Failed to parse journals (error: {e}), searching across all journals")
            return []

    def _clean_keywords(self, keywords):
        """Clean the keywords"""
        # Replace commas with AND
        keywords = keywords.replace(',', ' AND ')
        
        # Remove symbols (but keep hyphens)
        keywords = re.sub(r'[^\w\s-]', ' ', keywords)

        # Remove multiple spaces
        keywords = re.sub(r'\s+', ' ', keywords)

        # Remove extra spaces
        keywords = keywords.strip()

        return keywords

    def _execute(self, keywords, journals, max_results):
        # Validate the keywords
        if not keywords or not isinstance(keywords, str) or keywords.strip() == "":
            print(f"Invalid keywords: {keywords}")
            return {
                "All Journals": {
                    "term_used": keywords,
                    "summary": "Invalid keywords",
                    "citations": []
                }
            }

        # Clean the keywords
        keywords = self._clean_keywords(keywords)
        
        results = {}

        # Initialize the journal list
        journal_list = self._initialize_journal_list(journals)

        # Search across all journals if none were provided or parsed
        if not journal_list:
            results["All Journals"] = self._search_single_journal(keywords, None, max_results)
            return results

        # Search the publications in the specified journals
        for journal in journal_list:
            results[journal] = self._search_single_journal(keywords, journal, max_results)

        return results

    def run(self, **kwargs):
        # Get the parameters
        keywords = kwargs["query"] if "query" in kwargs else None
        journals = kwargs["journals"] if "journals" in kwargs else None
        max_results = kwargs["max_results"] if "max_results" in kwargs else self.max_results

        # Execute the tool
        results = self._execute(keywords, journals, max_results)

        return results


if __name__ == "__main__":
    """
    Test the PubMed Search Tool:
    python scientist/tools/pubmed_search/tool.py
    """
    from scientist.tools.base_tool import Tool
    import os
    import time
    from scientist.tools.utilis import print_json, save_result
    from scientist.engine.factory import create_llm_engine

    model_string = "gpt-4o"

    tool = PubMed_Search_Tool(engine_name=model_string)

    examples = [
        # {'query': 'microbiome gut inflammation', 'max_results': 20},
        # {'query': 'microbiome gut inflammation', 'max_results': 10, 'journals': 'JAMA'}, # Empty
        # {'query': 'microbiome, gut, inflammation', 'max_results': 10, 'journals': 'JAMA'}, # Empty
        # {'query': 'protein', 'max_results': 10, 'journals': 'JAMA'},
        # {'query': 'microbiome gut inflammation', 'max_results': 10, 'journals': 'Science'},
        # {'query': 'microbiome gut inflammation', 'max_results': 10, 'journals': 'Microbiome'},
        # {'query': 'microbiome gut inflammation', 'max_results': 10, 'journals': 'Cell'},
        # {'query': 'CRISPR gene editing applications 2024', 'journals': 'Nature'},
        # {'query': 'CRISPR gene editing applications 2024', 'journals': 'Science'},
        # {'query': 'CRISPR gene editing applications 2024', 'journals': 'Nature Genetics'},
        # {'query': 'CRISPR gene editing applications', 'journals': 'Nature OR Science'},
        # {'query': 'CRISPR gene editing applications', 'journals': ['Nature', 'Science']},
        {'query': 'CRISPR gene editing applications', 'journals': ['Nature', 'Science', 'Cell']},
        # {'query': 'CRISPR gene editing applications'},
        {'query': 'Bacteroides finegoldii inflammation DSS colitis'},
        {'query': 'Bacteroides AND finegoldii AND inflammation AND DSS AND colitis'},
        {'query': 'Dorea formicigenerans inflammation DSS colitis'}
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
