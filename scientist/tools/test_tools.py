"""
Parallel Tool Testing Script for Scientist Tools

This script tests all tools in parallel to verify they are properly configured
with the unified Tool interface.

Usage:
    cd scientist/tools
    python test_tools.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import time

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
scientist_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(scientist_dir)
sys.path.insert(0, project_root)


class ToolTest:
    """Class to represent a tool test"""

    def __init__(self, name: str, module_path: str, class_name: str):
        self.name = name
        self.module_path = module_path
        self.class_name = class_name
        self.result = None
        self.error = None
        self.duration = 0

    def run(self) -> Tuple[str, bool, str, float]:
        """Run the test for this tool"""
        start_time = time.time()
        try:
            # Import the tool class
            module = __import__(self.module_path, fromlist=[self.class_name])
            tool_class = getattr(module, self.class_name)

            try:
                tool = tool_class(llm=None)
            except TypeError:
                try:
                    tool = tool_class()
                except TypeError as e:
                    raise AssertionError(f"Initialization failed. Does the tool accept 'llm=None' or no arguments? Error: {str(e)}")

            # Verify required attributes
            checks = []
            checks.append(('name', hasattr(tool, 'name')))
            checks.append(('description', hasattr(tool, 'description')))
            checks.append(('run', hasattr(tool, 'run')))
            checks.append(('require_llm_engine', hasattr(tool, 'require_llm_engine')))
            checks.append(('get_metadata', hasattr(tool, 'get_metadata')))

            # Check if all required attributes exist
            failed_checks = [name for name, passed in checks if not passed]
            if failed_checks:
                raise AssertionError(f"Missing attributes: {', '.join(failed_checks)}")

            # Get metadata
            metadata = tool.get_metadata()
            required_metadata_keys = ['name', 'description', 'require_llm_engine']
            missing_keys = [key for key in required_metadata_keys if key not in metadata]
            if missing_keys:
                raise AssertionError(f"Missing metadata keys: {', '.join(missing_keys)}")

            self.duration = time.time() - start_time
            self.result = {
                'name': tool.name,
                'requires_llm': tool.require_llm_engine,
                'metadata_keys': list(metadata.keys())
            }
            return (self.name, True, None, self.duration)

        except Exception as e:
            self.duration = time.time() - start_time
            self.error = str(e)
            return (self.name, False, str(e), self.duration)


def get_all_tools() -> List[ToolTest]:
    """Get all tools to test"""
    return [
        ToolTest(
            "base_generator",
            "scientist.tools.base_generator.tool",
            "Base_Generator_Tool"
        ),
        ToolTest(
            "wikipedia_search",
            "scientist.tools.wikipedia_search.tool",
            "Wikipedia_Search_Tool"
        ),
        ToolTest(
            "kegg_gene_search",
            "scientist.tools.kegg_gene_search.tool",
            "KEGG_Gene_Search_Tool"
        ),
        ToolTest(
            "pubmed_search",
            "scientist.tools.pubmed_search.tool",
            "PubMed_Search_Tool"
        ),
        ToolTest(
            "google_search",
            "scientist.tools.google_search.tool",
            "Google_Search_Tool"
        ),
        ToolTest(
            "url_context_search",
            "scientist.tools.url_context_search.tool",
            "URL_Context_Search_Tool"
        ),
        ToolTest(
            "kegg_organism_search",
            "scientist.tools.kegg_organism_search.tool",
            "KEGG_Organism_Search_Tool"
        ),
        ToolTest(
            "kegg_drug_search",
            "scientist.tools.kegg_drug_search.tool",
            "KEGG_Drug_Search_Tool"
        ),
        ToolTest(
            "kegg_disease_search",
            "scientist.tools.kegg_disease_search.tool",
            "KEGG_Disease_Search_Tool"
        ),
        ToolTest(
            "perplexity_search",
            "scientist.tools.perplexity_search.tool",
            "Perplexity_Search_Tool"
        ),
        ToolTest(
            "mdipid_disease_search",
            "scientist.tools.mdipid_disease_search.tool",
            "MDIPID_Disease_Search_Tool"
        ),
        ToolTest(
            "mdipid_microbe_search",
            "scientist.tools.mdipid_microbe_search.tool",
            "MDIPID_Microbe_Search_Tool"
        ),
        ToolTest(
            "mdipid_gene_search",
            "scientist.tools.mdipid_gene_search.tool",
            "MDIPID_Gene_Search_Tool"
        ),
        ToolTest(
            "python_coder",
            "scientist.tools.python_coder.tool",
            "Python_Coder_Tool"
        )
    ]


def test_tool_base_class() -> Tuple[bool, str]:
    """Test the Tool base class"""
    try:
        from scientist.tools.base_tool import Tool

        # Check required attributes
        checks = {
            'require_llm_engine': hasattr(Tool, 'require_llm_engine'),
            'set_llm_engine': hasattr(Tool, 'set_llm_engine'),
            '__init__': hasattr(Tool, '__init__'),
            'run': hasattr(Tool, 'run'),
            'get_metadata': hasattr(Tool, 'get_metadata'),
        }

        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            return False, f"Missing attributes: {', '.join(failed)}"

        return True, "All required attributes present"
    except Exception as e:
        return False, str(e)


def run_parallel_tests(tools: List[ToolTest], max_workers: int = 6) -> Dict:
    """Run tests in parallel"""
    results = {
        'passed': [],
        'failed': [],
        'total': len(tools),
        'total_time': 0
    }

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tests
        future_to_tool = {executor.submit(tool.run): tool for tool in tools}

        # Collect results as they complete
        for future in as_completed(future_to_tool):
            tool = future_to_tool[future]
            try:
                name, passed, error, duration = future.result()
                if passed:
                    results['passed'].append({
                        'name': name,
                        'duration': duration,
                        'details': tool.result
                    })
                else:
                    results['failed'].append({
                        'name': name,
                        'error': error,
                        'duration': duration
                    })
            except Exception as e:
                results['failed'].append({
                    'name': tool.name,
                    'error': f"Test execution error: {str(e)}",
                    'duration': 0
                })

    results['total_time'] = time.time() - start_time
    return results


def print_results(base_test_result: Tuple[bool, str], test_results: Dict):
    """Print formatted test results"""
    print("\n" + "="*70)
    print("TOOL TEST RESULTS")
    print("="*70)

    # Base class test
    print("\n[1] Base Class Test")
    print("-" * 70)
    base_passed, base_msg = base_test_result
    if base_passed:
        print(f"✓ Tool base class: {base_msg}")
    else:
        print(f"✗ Tool base class: {base_msg}")

    # Passed tests
    print(f"\n[2] Tool Tests - Passed ({len(test_results['passed'])}/{test_results['total']})")
    print("-" * 70)
    if test_results['passed']:
        for result in sorted(test_results['passed'], key=lambda x: x['name']):
            name = result['name']
            duration = result['duration']
            details = result['details']
            requires_llm = details.get('requires_llm', 'Unknown')
            llm_status = "Requires LLM" if requires_llm else "No LLM needed"
            print(f"✓ {name:25s} ({duration:.3f}s) - {llm_status}")
    else:
        print("  (none)")

    # Failed tests
    print(f"\n[3] Tool Tests - Failed ({len(test_results['failed'])}/{test_results['total']})")
    print("-" * 70)
    if test_results['failed']:
        for result in sorted(test_results['failed'], key=lambda x: x['name']):
            name = result['name']
            error = result['error']
            print(f"✗ {name:25s}")
            print(f"  Error: {error}")
    else:
        print("  (none)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Base Class Test: {'PASSED' if base_passed else 'FAILED'}")
    print(f"Tools Tested: {test_results['total']}")
    print(f"Tools Passed: {len(test_results['passed'])}")
    print(f"Tools Failed: {len(test_results['failed'])}")
    print(f"Total Time: {test_results['total_time']:.3f}s")
    print(f"Success Rate: {len(test_results['passed'])/test_results['total']*100:.1f}%")

    # Tool categorization
    if test_results['passed']:
        tools_with_llm = [r for r in test_results['passed']
                         if r['details'].get('requires_llm', False)]
        tools_without_llm = [r for r in test_results['passed']
                            if not r['details'].get('requires_llm', False)]
        print(f"\nTools requiring LLM: {len(tools_with_llm)}")
        for r in tools_with_llm:
            print(f"  - {r['name']}")
        print(f"\nTools not requiring LLM: {len(tools_without_llm)}")
        for r in tools_without_llm:
            print(f"  - {r['name']}")

    print("="*70)

    # Return exit code
    all_passed = base_passed and len(test_results['failed']) == 0
    return 0 if all_passed else 1


def main():
    """Main test function"""
    print("Starting Parallel Tool Tests...")
    print("="*70)

    # Test base class
    print("\nTesting Tool base class...")
    base_test_result = test_tool_base_class()

    # Get all tools
    tools = get_all_tools()
    print(f"Found {len(tools)} tools to test")

    # Run tests in parallel
    print(f"\nRunning tests in parallel (max 6 workers)...")
    test_results = run_parallel_tests(tools, max_workers=6)

    # Print results
    exit_code = print_results(base_test_result, test_results)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
