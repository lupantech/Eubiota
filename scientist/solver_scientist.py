"""
Scientist Solver - Main entry point for using the base_agent system
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional

# Add parent directory to path so we can import scientist module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scientist.base_agent.agent import Agent
from scientist.engine.factory import create_llm_engine


def construct_solver(
    role_prompt: str = "You are an expert AI assistant specialized in scientific research and problem-solving.",
    suggested_tools: List[str] = None,
    enabled_tools: List[str] = None,  # Alias for suggested_tools
    llm_engine_name: str = "gpt-4o-mini",
    module_engine: List[str] = None,  # [planner, executor, verifier, generator] engines
    tool_engine: List[str] = None,  # Tool engines for each tool (e.g., ["claude-sonnet-4-0", "gpt-4o", ...])
    verbose: bool = False,
    logging_path: str = None,
    workspace_path: str = None,
    prompts_path: str = None,
    base_url: str = None,
    temperature: float = 0.0,
    check_model: bool = True,  # For compatibility, currently not used
    is_validation: bool = False
):
    """
    Construct a Scientist Agent solver.

    Args:
        role_prompt: System prompt for the agent
        suggested_tools: List of tool names to use
        enabled_tools: Alias for suggested_tools (for compatibility)
        llm_engine_name: LLM engine name for all modules (if module_engine not specified)
        module_engine: List of engine names for [planner, executor, verifier, generator]
                       If None, all modules use llm_engine_name
        tool_engine: Tool engine configuration - list of engine names for each tool
                     Order matches suggested_tools order (e.g., ["claude-sonnet-4-0", "gpt-4o", ...])
        verbose: Whether to print verbose output
        logging_path: Path for log file
        workspace_path: Workspace path for tool execution
        prompts_path: Path to prompts directory
        base_url: Base URL for LLM API
        temperature: LLM temperature
        check_model: Whether to check model availability (for compatibility, currently not used)

    Returns:
        Configured Agent instance
    """
    # Handle enabled_tools alias
    if enabled_tools is not None and suggested_tools is None:
        suggested_tools = enabled_tools

    # Set default tools if none specified
    if suggested_tools is None:
        suggested_tools = [
            "Base_Generator_Tool",
            "Wikipedia_Search_Tool",
            "KEGG_Gene_Search_Tool",
            "PubMed_Search_Tool",
            "Google_Search_Tool",
            "Perplexity_Search_Tool"
        ]

    # Set default module engines if not specified
    if module_engine is None:
        module_engine = [llm_engine_name] * 4  # All modules use same engine

    # Validate module_engine length
    if len(module_engine) != 4:
        raise ValueError(f"module_engine must have 4 elements [planner, executor, verifier, generator], got {len(module_engine)}")

    # Create LLM engines for each module
    llm_planner = create_llm_engine(
        model_string=module_engine[0],
        is_multimodal=False,
        base_url=base_url,
        temperature=temperature
    )

    llm_executor = create_llm_engine(
        model_string=module_engine[1],
        is_multimodal=False,
        base_url=base_url,
        temperature=temperature
    )

    llm_verifier = create_llm_engine(
        model_string=module_engine[2],
        is_multimodal=False,
        base_url=base_url,
        temperature=temperature
    )

    # Generator uses verifier engine (4th module engine)
    llm_generator = create_llm_engine(
        model_string=module_engine[3],
        is_multimodal=False,
        base_url=base_url,
        temperature=temperature
    )

    # Create Agent with reference interface
    agent = Agent(
        role_prompt=role_prompt,
        suggested_tools=suggested_tools,
        tool_engine=tool_engine,
        llm_planner=llm_planner,
        llm_executor=llm_executor,
        llm_verifier=llm_verifier,
        prompts_path=prompts_path,
        workspace_path=workspace_path,
        is_validation=is_validation
    )

    # Update generator's LLM (Agent.__init__ sets generator.llm to verifier's llm)
    agent.generator.llm = llm_generator

    # Setup logger if path provided
    if logging_path:
        agent._setup_logger(logging_path)

    return agent


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Scientist Solver - Comprehensive Tool Test with Gene Research")
    print("=" * 70)
    print()

    # All available tools
    all_tools = [
    "Base_Generator_Tool",
    "Wikipedia_Search_Tool",
    "KEGG_Gene_Search_Tool",
    "PubMed_Search_Tool",
    "Google_Search_Tool",
    "URL_Context_Search_Tool",
    "KEGG_Organism_Search_Tool",
    "KEGG_Drug_Search_Tool",
    "KEGG_Disease_Search_Tool",
    "KEGG_Organism_Search_Tool",
    "Perplexity_Search_Tool",
    "MDIPID_Disease_Search_Tool",
    "MDIPID_Microbe_Search_Tool",
    "MDIPID_Gene_Search_Tool",
    ]

    print("Creating solver with logger at test.log...")
    print()

    # Create solver
    agent = construct_solver(
        role_prompt="You are an expert AI research assistant specialized in biomedical sciences, genetics, and drug discovery.",
        suggested_tools=all_tools,
        llm_engine_name="gpt-4o-mini",
        verbose=True,
        logging_path=os.path.abspath("test.log")
    )

    # Complex gene research query about diabetes drug targets
    gene_query = """
    Research Question: Is the GLP1R (Glucagon-Like Peptide 1 Receptor) gene a valid
    therapeutic target for Type 2 Diabetes treatment?

    Please investigate:
    1. The biological function of GLP1R gene and its role in glucose metabolism
    2. Known molecular pathways involving GLP1R in diabetes pathogenesis
    3. Current drugs targeting GLP1R and their clinical efficacy
    4. Evidence from scientific literature supporting GLP1R as a diabetes target
    5. Any genetic variants or mutations in GLP1R associated with diabetes risk

    Provide a comprehensive analysis with supporting evidence from multiple sources.
    """

    print("=" * 70)
    print("Query: Gene Target Analysis for Type 2 Diabetes")
    print("=" * 70)
    print(gene_query)
    print("=" * 70)
    print()

    # Solve the query with multiple reasoning steps
    try:
        result = agent.run(
            agent_query=gene_query,
            input_files={},  # No files for this query
            output_types="direct,final",
            output_schema=None,
            max_steps=1,
            max_time=300,
            logging_path=os.path.abspath("test.log")
        )
        
    finally:
        if hasattr(agent, 'cleanup'):
            agent.cleanup()

    # save the result to a json file
    print("Saving result to test_result.json...")
    with open("test_result.json", "w") as f:
        json.dump(result, f, indent=4)

    print()
    print("=" * 70)
    print("Result Summary:")
    print("=" * 70)
    
    outputs = result.get('outputs', {})
    execution_metadata = outputs.get('execution_metadata', {})
    print(f"Steps Executed: {execution_metadata.get('steps_completed', 'N/A')}")
    print(f"Execution Time: {outputs.get('total_execution_time', 0):.2f}s")
    print(f"Final Status: {outputs.get('final_status', 'N/A')}")
    print()

    print("=" * 70)
    print("Direct Output:")
    print("=" * 70)
    final_output = outputs.get('final_output', {})
    direct_output = final_output.get('direct', {})
    print(direct_output.get('response', 'N/A') if isinstance(direct_output, dict) else direct_output)
    print()

    print("=" * 70)
    print("Final Output:")
    print("=" * 70)
    final = final_output.get('final', {})
    print(final.get('response', 'N/A') if isinstance(final, dict) else final)
    print()

    print("=" * 70)
    print(f"Log saved to: test.log")
    print(f"Result saved to: test_result.json")
    print("=" * 70)
