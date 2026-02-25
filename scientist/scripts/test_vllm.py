from trainer.engine.vllm import ChatVLLM

def test_text_generation():
    print("--- Testing Text Generation ---")
    try:
        llm = ChatVLLM(
            model_string="YOUR_LOCAL_MODEL_NAME",
            base_url="http://localhost:YOUR_PORT/v1/",
            use_cache=False,
            system_prompt="You are a helpful AI assistant."
        )

        test_prompt = """\\nTask: Analyze the given query to determine necessary skills and tools.\\n\\nInputs:\\n- Available tools: [\'Generalist_Solution_Generator_Tool\', \'Pubmed_Search_Tool\', \'Python_Code_Generator_Tool\', \'Wikipedia_Knowledge_Searcher_Tool\']\\n- Metadata for tools: {\'Generalist_Solution_Generator_Tool\': {\'tool_name\': \'Generalist_Solution_Generator_Tool\', \'tool_description\': \'A generalized tool that takes query from the user, and answers the question step by step to the best of its ability. It can also accept an image.\', \'tool_version\': \'1.0.0\', \'input_types\': {\'query\': \\"str - The query that includes query from the user to guide the agent to generate response (Examples: \'Describe this image in detail\').\\"}, \'output_type\': \'str - The generated response to the original query\', \'demo_commands\': [{\'command\': \'execution = tool.execute(query=\\"Summarize the following text in a few lines\\")\', \'description\': \'Generate a short summary given the query from the user.\'}], \'user_metadata\': {\'limitation\': \'The Generalist_Solution_Generator_Tool may provide hallucinated or incorrect responses.\', \'best_practice\': \\"Use the Generalist_Solution_Generator_Tool for general queries or tasks that don\'t require specialized knowledge or specific tools in the toolbox. For optimal results:\\\\n\\\\n1) Provide clear, specific query.\\\\n2) Use it to answer the original query through step by step reasoning for tasks without complex or multi-step reasoning.\\\\n3) For complex queries, break them down into subtasks and use the tool multiple times.\\\\n4) Use it as a starting point for complex tasks, then refine with specialized tools.\\\\n5) Verify important information from its responses.\\\\n\\"}, \'require_llm_engine\': True}, \'Pubmed_Search_Tool\': {\'tool_name\': \'Pubmed_Search_Tool\', \'tool_description\': \'A tool that searches PubMed Central to retrieve relevant article abstracts based on a given list of text queries. Use this ONLY if you cannot use the other more specific ontology tools.\', \'tool_version\': \'1.0.0\', \'input_types\': {\'queries\': \'list[str] - list of queries terms for searching PubMed.\'}, \'output_type\': \'list - List of items matching the search query. Each item consists of the title, abstract, keywords, and URL of the article. If no results found, a string message is returned.\', \'demo_commands\': [{\'command\': \'execution = tool.execute(query=[\\"scoliosis\\", \\"injury\\"])\', \'description\': \\"Search for PubMed articles mentioning \'scoliosis\' OR \'injury\'.\\"}, {\'command\': \'execution = tool.execute(query=[\\"COVID\\", \\"vaccine\\", \\"occupational health\\"])\', \'description\': \\"Search for PubMed articles mentioning \'COVID\' OR \'vaccine\' OR \'occupational health\'.\\"}], \'user_metadata\': {\'limitations\': \'Try to use shorter and more general search queries.\'}, \'require_llm_engine\': False}, \'Python_Code_Generator_Tool\': {\'tool_name\': \'Python_Code_Generator_Tool\', \'tool_description\': \'A tool that generates and executes simple Python code snippets for basic arithmetical calculations and math-related problems. The generated code runs in a highly restricted environment with only basic mathematical operations available.\', \'tool_version\': \'1.0.0\', \'input_types\': {\'query\': \'str - A clear, specific description of the arithmetic calculation or math problem to be solved, including any necessary numerical inputs.\'}, \'output_type\': \'dict - A dictionary containing the generated code, calculation result, and any error messages.\', \'demo_commands\': [{\'command\': \'execution = tool.execute(query=\\"Find the sum of prime numbers up to 50\\")\', \'description\': \'Generate a Python code snippet to find the sum of prime numbers up to 50.\'}, {\'command\': \'query=\\"Given the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], calculate the sum of squares of odd numbers\\"\\\\nexecution = tool.execute(query=query)\', \'description\': \'Generate a Python function for a specific mathematical operation on a given list of numbers.\'}], \'user_metadata\': {\'limitations\': [\'Restricted to basic Python arithmetic operations and built-in mathematical functions.\', \'Cannot use any external libraries or modules, including those in the Python standard library.\', \'Limited to simple mathematical calculations and problems.\', \'Cannot perform any string processing, data structure manipulation, or complex algorithms.\', \'No access to any system resources, file operations, or network requests.\', \\"Cannot use \'import\' statements.\\", \'All calculations must be self-contained within a single function or script.\', \'Input must be provided directly in the query string.\', \'Output is limited to numerical results or simple lists/tuples of numbers.\'], \'best_practices\': [\'Provide clear and specific queries that describe the desired mathematical calculation.\', \'Include all necessary numerical inputs directly in the query string.\', \'Keep tasks focused on basic arithmetic, algebraic calculations, or simple mathematical algorithms.\', \'Ensure all required numerical data is included in the query.\', \'Verify that the query only involves mathematical operations and does not require any data processing or complex algorithms.\', \'Review generated code to ensure it only uses basic Python arithmetic operations and built-in math functions.\']}, \'require_llm_engine\': True}, \'Wikipedia_Knowledge_Searcher_Tool\': {\'tool_name\': \'Wikipedia_Knowledge_Searcher_Tool\', \'tool_description\': \'A tool that searches Wikipedia and returns web text based on a given query.\', \'tool_version\': \'1.0.0\', \'input_types\': {\'query\': \'str - The search query for Wikipedia.\'}, \'output_type\': \'dict - A dictionary containing the search results, extracted text, and any error messages.\', \'demo_commands\': [{\'command\': \'execution = tool.execute(query=\\"Python programming language\\")\', \'description\': \'Search Wikipedia for information about Python programming language.\'}, {\'command\': \'execution = tool.execute(query=\\"Artificial Intelligence\\")\', \'description\': \'Search Wikipedia for information about Artificial Intelligence\'}, {\'command\': \'execution = tool.execute(query=\\"Theory of Relativity\\")\', \'description\': \'Search Wikipedia for the full article about the Theory of Relativity.\'}], \'user_metadata\': None, \'require_llm_engine\': False}}\\n- Query: A point $(x,y)$ is randomly and uniformly chosen inside the square with vertices (0,0), (0,2), (2,2), and (2,0).  What is the probability that $x+y < 3$? When ready, output the final answer enclosed in <answer> and </answer> tags. If question is a multi-option problem, answer in the format of option(e.g., A, B, C, ... ), please output the option directly. Do not generate any content after the </answer> tag.\\n\\nInstructions:\\n1. Identify the main objectives in the query.\\n2. List the necessary skills and tools.\\n3. For each skill and tool, explain how it helps address the query.\\n4. Note any additional considerations.\\n\\nFormat your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.\\n\\nBe biref and precise with insight. \\n"}]"""
        response = llm.generate(test_prompt, temperature=0.7, max_tokens=2048) # please also test for [response] because there might be multimodal issue for generate function. 

        print("\n--- Test Prompt ---")
        print(test_prompt)
        print("\n--- Generated Response ---")
        print(response)
        print("\n--- Test Passed ---\n")

    except Exception as e:
        print(f"\n--- Test Failed ---")
        print(f"Error during text generation test: {e}")
        print("--- Test Failed ---\n")

def test_connection_failure():
    print("--- Testing Connection Failure (Optional) ---")
    try:
        llm = ChatVLLM(
            base_url="http://localhost:12345/v1",
            use_cache=False
        )
        print("[WARNING] Connection to invalid URL succeeded unexpectedly.")
    except ValueError as e:
        print(f"[INFO] Expected connection failure caught: {e}")
        print("[INFO] Connection failure test passed.")
    except Exception as e:
        print(f"[INFO] Unexpected error during connection failure test: {e}")
    print("--- Connection Failure Test Done ---\n")


if __name__ == "__main__":
    print("Starting vLLM Connection and Generation Tests...\n")

    test_text_generation()
    
    # test_connection_failure() 

    print("All selected tests completed.")