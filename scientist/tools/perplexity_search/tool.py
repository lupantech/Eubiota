import os
import requests

from scientist.tools.base_tool import Tool

from dotenv import load_dotenv
load_dotenv()

TOOL_NAME = "Perplexity_Search_Tool"

LIMITATIONS = """
1. This tool only suitable for search queries that can be answered by a narritive paragraph.
2. This tool is not suitable to provide domain specific information or knowledge.
3. Keywords, Bag-of-words types of query may not be able to provide the best results.
"""

BEST_PRACTICES = """
1. Choose this tool when you want to search general information about a topic.
2. Be specific about the question you want to ask.
3. The query should be a single question, not a statement.
4. Keep the query concise and to the point. The tool will return a summarized information.
5. Never assume this tool has access to context from previous steps. Always provide context in the tool input query.
"""

class Perplexity_Search_Tool(Tool):
    def __init__(self):
        super().__init__(
            name=TOOL_NAME,
            description="An AI-powered web search tool that provides conversational answers to questions by searching and synthesizing information from multiple web sources.",
            input_kwargs={
                "query": {
                    "type": "string",
                    "description": "The question or query to search and get a conversational answer"
                },
                "max_tokens": {
                    "type": "integer", 
                    "description": "The maximum number of tokens to return"
                }
            },
            output_schema={
                "result": {
                    "type": "string",
                    "description": "The conversational answer to the query"
                },
                "citations": {
                    "type": "dict",
                    "description": "The citations of the result"
                }
            },
            limitations=LIMITATIONS,
            best_practices=BEST_PRACTICES,
            documentation_path="https://docs.perplexity.ai/getting-started/overview"
        )
        self.max_tokens = 2000
        self.model = "sonar"

    def _execute(self, query: str, max_tokens: int):
        """
        Search the query and return the result.
        """
        try:
            api_key = os.getenv("PERPLEXITY_API_KEY")
        except Exception as e:
            raise Exception(f"Perplexity API key not found. Please set the PERPLEXITY_API_KEY environment variable.")

        # Set up the API endpoint and headers
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Define the request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            'max_tokens': max_tokens
        }

        # Make the API call
        response = requests.post(url, headers=headers, json=payload)
        formatted_response = response.json()["choices"][0]['message']['content']

        # Get the citations
        try:
            citations = response.json()["citations"]
            citations = {f"[{i+1}]": url for i, url in enumerate(citations)}
        except KeyError:
            citations = {}

        return formatted_response, citations

    def run(self, **kwargs):
        query = kwargs["query"]
        max_tokens = kwargs.get("max_tokens", self.max_tokens)  # default to 2000 tokens

        # Search the query
        result, citations = self._execute(query, max_tokens)

        # Format the result
        result = {
            "result": result,
            "citations": citations
        }

        return result


if __name__ == "__main__":
    """
    Test the Perplexity Search Tool:
    python scientist/tools/perplexity_search/tool.py
    """
    import os
    import time
    from scientist.tools.utilis import print_json, save_result

    tool = Perplexity_Search_Tool()

    examples = [
        {'query': 'What is the capital of France?', 'max_tokens': 2000},
        {'query': 'Who won the euro 2024?', 'max_tokens': 2000},
        {'query': 'triisopropyl borate C3h symmetry', 'max_tokens': 500},
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
