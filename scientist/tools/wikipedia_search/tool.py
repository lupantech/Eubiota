"""
Wikipedia RAG Tool - Adapted for unified Tool interface
Supports LLM engine configuration via engine_name
Uses URL_Context_Search_Tool for true RAG capabilities
"""

import os
import sys
import wikipedia
from pydantic import BaseModel

from scientist.tools.base_tool import Tool
from scientist.engine.factory import create_llm_engine
from scientist.tools.url_context_search import URL_Context_Search_Tool

# Tool name mapping
TOOL_NAME = "Wikipedia_Search_Tool"

LIMITATIONS = f"""
{TOOL_NAME} has the following limitations:
1. It is designed specifically for retrieving grounded information from Wikipedia pages only.
2. Filtering of relevant pages depends on LLM model performance and may not always select optimal pages.
3. The returned information accuracy depends on Wikipedia content quality.
4. Requires OpenAI API for embeddings in the RAG process.
5. RAG processing may take longer for pages with extensive content.
"""

BEST_PRACTICES = f"""
For optimal results with {TOOL_NAME}:
1. Use specific, targeted queries rather than broad or ambiguous questions.
2. {TOOL_NAME} automatically filters for relevant pages using LLM-based selection and retrieves information using RAG technology.
3. If initial results are insufficient, examine the "other_pages" section for additional potentially relevant content.
4. Use {TOOL_NAME} as part of a multi-step research process rather than a single source of truth.
5. The tool uses chunking and embedding to find the most relevant information from Wikipedia pages.
"""


class Select_Relevant_Queries(BaseModel):
    matched_queries: list[str]
    matched_query_ids: list[int]


def select_relevant_queries(original_query: str, query_candidates: list[str], llm_engine):
    """Select relevant Wikipedia pages using LLM."""
    query_candidates_str = "\n".join([f"{i}. {query}" for i, query in enumerate(query_candidates)])

    prompt = f"""You are an expert AI assistant. Your task is to identify and select the most relevant queries from a list of Wikipedia search results that are most likely to address the user's original question.

## Input

Original Query: `{original_query}`
Query Candidates from Wikipedia Search:
{query_candidates_str}

## Instructions

1. Carefully read the original query and the list of query candidates.
2. Select the query candidates that are most relevant to the original query — i.e., those most likely to contain the information needed to answer the question.
3. Return the most relevant queries. If you think multiple queries are helpful, you can return up to 3 queries.
4. Return your output in the following format:

```
- Matched Queries: <list of matched queries>
- Matched Query IDs: <list of matched query ids>. Please make sure the ids are integers. And do not return empty list.
```

## Examples

Original Query: What is the capital of France?
Query Candidates from Wikipedia Search:
0. Closed-ended question
1. France
2. What Is a Nation?
3. Capital city
4. London
5. WhatsApp
6. French Revolution
7. Communes of France
8. Capital punishment
9. Louis XIV

Output:
- Matched Queries: France
- Matched Query IDs: [1]
"""

    try:
        response = llm_engine(prompt, response_format=Select_Relevant_Queries)

        matched_queries = response.matched_queries
        matched_query_ids = [int(i) for i in response.matched_query_ids]
        return matched_queries, matched_query_ids
    except Exception as e:
        print(f"Error selecting relevant queries: {e}")
        return [], []


class Wikipedia_Search_Tool(Tool):
    """
    Wikipedia search tool with RAG capabilities.
    Searches Wikipedia and returns relevant pages with their information.

    Requires LLM engine for relevance filtering.
    """

    require_llm_engine = True

    def __init__(self, engine_name="gpt-4o"):
        """
        Initialize Wikipedia RAG Tool.

        Args:
            engine_name: Name of the engine to create (e.g., "gpt-4o-mini", "dashscope", "Default")
        """
        super().__init__(
            name=TOOL_NAME,
            description="A tool that searches Wikipedia and returns relevant pages with their page titles, URLs, abstract, and retrieved information based on a given query.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The search query for Wikipedia."
                }
            },
            output_schema={
                "query": {
                    "type": "string",
                    "description": "The original search query"
                },
                "relevant_pages": {
                    "type": "array",
                    "description": "Array of relevant page objects with title, url, abstract, and retrieved_information"
                },
                "other_pages": {
                    "type": "array",
                    "description": "Array of other potentially relevant pages"
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES,
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
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            self.llm = self._llm_engine

    def _get_wikipedia_url(self, query):
        """Get the Wikipedia URL for a given query."""
        query = query.replace(" ", "_")
        return f"https://en.wikipedia.org/wiki/{query}"

    def search_wikipedia(self, query, max_length=100, max_pages=10):
        """
        Searches Wikipedia based on the given query and returns multiple pages with their text and URLs.

        Parameters:
            query (str): The search query for Wikipedia.
            max_length (int): Maximum length of abstract
            max_pages (int): Maximum number of pages to retrieve

        Returns:
            list: List of dictionaries containing page info (title, text, url)
        """
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                return [{"title": None, "url": None, "abstract": None, "error": f"No results found for query: {query}"}]

            pages_data = []
            pages_to_process = search_results[:max_pages] if max_pages else search_results

            for title in pages_to_process:
                try:
                    page = wikipedia.page(title)
                    text = page.content
                    url = page.url

                    if max_length != -1:
                        text = text[:max_length] + f"... [truncated]" if len(text) > max_length else text

                    pages_data.append({
                        "title": title,
                        "url": url,
                        "abstract": text
                    })
                except Exception as e:
                    pages_data.append({
                        "title": title,
                        "url": self._get_wikipedia_url(title),
                        "abstract": "Please use the URL to get the full text further if needed.",
                    })

            return pages_data
        except Exception as e:
            return [{"title": None, "url": None, "abstract": None, "error": f"Error searching Wikipedia: {str(e)}"}]

    def run(self, **kwargs):
        """
        Searches Wikipedia based on the provided query and returns all matching pages.

        Parameters:
            query (str): The search query for Wikipedia.

        Returns:
            dict: A dictionary containing the search results and all matching pages with their content.
        """
        query = kwargs.get("query", "")

        # Check if OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "[Wikipedia RAG Search] Error: OPENAI_API_KEY environment variable is not set."}

        # Ensure LLM engine is initialized
        self._ensure_engine()

        # First get relevant queries from the search results
        search_results = self.search_wikipedia(query)

        # Get the titles of the pages
        titles = [page["title"] for page in search_results if page["title"] is not None]
        if not titles:
            return {"query": query, "relevant_pages": [], "other_pages": search_results}

        # Select the most relevant pages
        matched_queries, matched_query_ids = select_relevant_queries(query, titles, self._llm_engine)

        # Only process the most relevant pages
        pages_data = [search_results[i] for i in matched_query_ids if i < len(search_results)]
        other_pages = [search_results[i] for i in range(len(search_results)) if i not in matched_query_ids]

        # For each relevant page, use URL_Context_Search_Tool for true RAG retrieval
        try:
            web_rag_tool = URL_Context_Search_Tool(engine_name=self._engine_name)
        except Exception as e:
            print(f"Error creating URL Context Search tool: {e}")
            # Fall back to just using abstracts if URL_Context_Search_Tool fails
            for page in pages_data:
                page["retrieved_information"] = page.get("abstract", "No information available")
            return {"query": query, "relevant_pages": pages_data, "other_pages": other_pages}

        # Use RAG to retrieve detailed information from each relevant page
        for page in pages_data:
            url = page["url"]
            if url is None:
                page["retrieved_information"] = "No information available"
                continue

            try:
                # Use Web RAG to get detailed, relevant information
                rag_result = web_rag_tool.run(query=query, url=url)
                page["retrieved_information"] = rag_result.get("result", page.get("abstract", "No information available"))
            except Exception as e:
                print(f"Error retrieving RAG information from {url}: {e}")
                # Fall back to abstract if RAG fails
                page["retrieved_information"] = page.get("abstract", "No information available")

        return {
            "query": query,
            "relevant_pages": pages_data,
            "other_pages": other_pages
        }


if __name__ == "__main__":
    """
    Test the Wikipedia Search Tool:
    python scientist/tools/wikipedia_search/tool.py
    """
    from scientist.tools.utilis import print_json, save_result

    print("Testing Wikipedia Search Tool...")

    tool = Wikipedia_Search_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(metadata)
    print("\n" + "="*50 + "\n")

    # Sample query for searching Wikipedia
    query = "When was the first moon landing?"

    # Execute the tool with the sample query
    try:
        result = tool.run(query=query)
        print("Execution Result:")
        print_json(result)
        save_result(result, query, os.path.join(os.path.dirname(__file__), "test_logs"))
    except Exception as e:
        print(f"Execution failed: {e}")

    print("\nDone!")
