import os
import json
import requests
import re
from typing import List

from google import genai
from google.genai import types

from scientist.tools.base_tool import Tool

from dotenv import load_dotenv
load_dotenv()

# Tool name constant
TOOL_NAME = "Google_Search_Tool"

LIMITATIONS = f"""
1. {TOOL_NAME} is only suitable for general information search.
2. {TOOL_NAME} contains less domain specific information.
3. {TOOL_NAME} is not suitable for searching and analyzing videos at YouTube or other video platforms.
"""

BEST_PRACTICES = f"""
For optimal results with {TOOL_NAME}:
1. Choose {TOOL_NAME} when you want to search general information about a topic.
2. Choose {TOOL_NAME} for question-type queries, such as "What is the capital of France?"
3. **CRITICAL**: The query MUST be specific, clear, and self-contained. Avoid vague references like "this", "that", "the above", "mentioned earlier".
4. **CRITICAL**: Include all necessary context directly in the query. Use specific names, identifiers, keywords, or terms.
5. {TOOL_NAME} will return summarized information with citations.
6. {TOOL_NAME} is more suitable for definitions, world knowledge, and general information search.
7. For entities or concepts from previous steps: explicitly include the full name or description in the query rather than using pronouns or vague references.
8. Never assume this tool has access to context from previous steps. Always provide context in the tool input query.

Examples:
- GOOD: "What is the molecular weight of Aspirin (acetylsalicylic acid)?"
- BAD: "What is the molecular weight of the compound mentioned above?"
- GOOD: "What are the side effects of Metformin for type 2 diabetes treatment?"
- BAD: "What are the side effects of this medication?"
- GOOD: "When was the Eiffel Tower built in Paris, France?"
- BAD: "When was it built?"
"""

class Google_Search_Tool(Tool):
    """
    Google Search Tool - performs web searches using Google's Gemini AI.
    Google_Search_Tool does not require an LLM engine (uses Google's API directly).
    """

    require_llm_engine = False

    def __init__(self):
        """
        Initialize Google Search Tool.
        Google_Search_Tool does not require an LLM engine (uses Google's API directly).
        """
        super().__init__(
            name=TOOL_NAME,
            description="A web search tool powered by Google's Gemini AI that provides real-time information from the internet with citation support.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The search query to find information on the web."
                },
                "add_citations": {
                    "type": "boolean",
                    "description": "Whether to add citations to the results. If True, the results will be formatted with citations. By default, it is True.",
                    "optional": True
                }
            },
            output_schema={
                "response": {
                    "type": "string", # NOTE: this is a string, not an array
                    "description": "The search results of the query."
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES,
            documentation_path="https://ai.google.dev/gemini-api/docs/google-search",
            llm=None
        )
        self.max_retries = 5
        self.search_model = "gemini-2.5-flash" # gemini-3-flash-preview is very slow and often stucks

        # Initialize client once during tool initialization
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)

        # Initialize grounding tool and config once
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        self.config = types.GenerateContentConfig(
            tools=[self.grounding_tool]
        )

    @staticmethod
    def get_real_url(url):
        """
        Convert a redirect URL to the final real URL in a stable manner.

        This function handles redirects by:
        1.  Setting a browser-like User-Agent to avoid being blocked or throttled.
        2.  Using a reasonable timeout to prevent getting stuck indefinitely.
        3.  Following HTTP redirects automatically (default requests behavior).
        4.  Catching specific request-related exceptions for cleaner error handling.
        """
        try:
            # Headers to mimic a real browser visit
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # allow_redirects=True is the default, but we state it for clarity.
            # The request will automatically follow the 3xx redirect chain.
            response = requests.get(
                url, 
                headers=headers, 
                timeout=3, # Increased timeout for more reliability
                allow_redirects=True 
            )
            
            # After all redirects, response.url contains the final URL.
            return response.url
            
        except Exception as e:
            # Catching specific exceptions from the requests library is better practice.
            # print(f"An error occurred: {e}")
            return url

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """
        Extract all URLs from Markdown‑style citations [number](url) in the given text.

        Args:
            text: A string containing Markdown citations.

        Returns:
            A list of URL strings.
        """
        pattern = re.compile(r'\[\d+\]\((https?://[^\s)]+)\)')
        urls = pattern.findall(text)
        return urls

    def reformat_response(self, response: str) -> str:
        """
        Reformat the response to a readable format.
        """
        urls = self.extract_urls(response)
        for url in urls:
            direct_url = self.get_real_url(url)
            response = response.replace(url, direct_url)
        return response

    @staticmethod
    def add_citations(response):
        text = response.text
        supports = response.candidates[0].grounding_metadata.grounding_supports
        chunks = response.candidates[0].grounding_metadata.grounding_chunks

        # Sort supports by end_index in descending order to avoid shifting issues when inserting.
        sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

        for support in sorted_supports:
            end_index = support.segment.end_index
            if support.grounding_chunk_indices:
                # Create citation string like [1](link1)[2](link2)
                citation_links = []
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        uri = chunks[i].web.uri
                        citation_links.append(f"[{i + 1}]({uri})")

                citation_string = ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]

        return text

    def _execute(self, query: str, add_citations_flag: bool):
        """
        https://ai.google.dev/gemini-api/docs/google-search
        """
        response = None
        response_text = None

        # Use pre-initialized client and config
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.search_model,
                    contents=query,
                    config=self.config,
                )
                response_text = response.text
                # If we get here, the API call was successful, so break out of the retry loop
                break
            except Exception as e:
                print(f"Google Search attempt {attempt + 1} failed: {str(e)}. Retrying...")
                if attempt == self.max_retries - 1:  # Last attempt
                    print(f"Google Search failed after {self.max_retries} attempts. Last error: {str(e)}")
                    return f"Google Search tried {self.max_retries} times but failed. Last error: {str(e)}"
                # Continue to next attempt

        # Check if we have a valid response before proceeding
        if response is None or response_text is None:
            return "Google Search failed to get a valid response"

        # Add citations if needed
        try:
            response_text = self.add_citations(response) if add_citations_flag else response_text
        except Exception as e:
            pass
            # print(f"Error adding citations: {str(e)}")
            # Continue with the original response_text if citations fail

        # Format the response
        try:
            response_text = self.reformat_response(response_text)
        except Exception as e:
            pass
            # print(f"Error reformatting response: {str(e)}")
            # Continue with the current response_text if reformatting fails

        return response_text


    def run(self, **kwargs):
        # Configure the arguments
        query = kwargs["query"]
        add_citations_flag = kwargs.get("add_citations", True) # default is True

        # Perform the search
        response = self._execute(query, add_citations_flag)
        from datetime import datetime
        import os
        
        # Ensure output directory exists
        if self.output_dir is None:
            # Set default output directory for testing
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(script_dir, "test_logs")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_file_name = self.output_dir + "/google_search_result_" + timestamp + ".txt"
        # with open(output_file_name, "w") as f:
        #     f.write('query: ' + query + '\n' + 'Google Search Result: ' + response)

        # Format the result
        result = {
            "response": response,

        }
        return result
    

if __name__ == "__main__":
    """
    Test the Google Search Tool:
    python scientist/tools/google_search/tool.py
    """
    import os
    import time
    from scientist.tools.utilis import print_json, save_result

    tool = Google_Search_Tool()

    examples = [
        # {'query': 'What is the capital of France?', 'add_citations': True},
        # {'query': 'Who won the euro 2024?', 'add_citations': False},
        # {'query': 'Who won the euro 2024?', 'add_citations': True},
        {'query': 'Physics and Society article arXiv August 11, 2016', 'add_citations': True},
        {'query': 'Physics and Society article arXiv August 11, 2016', 'add_citations': False},
        {'query': 'triisopropyl borate C3h symmetry', 'add_citations': False},
        {'query': 'triisopropyl borate C3h symmetry', 'add_citations': True},
        {'query': 'highest number of bird species on camera simultaneously in the video https://www.youtube.com/watch?v=L1vXCYZAYYM', 'add_citations': True},
        {'query': "Find the link to GitHub issue #26843 titled 'BUG: performance regression of polynomial evaluation'", 'add_citations': True},
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
