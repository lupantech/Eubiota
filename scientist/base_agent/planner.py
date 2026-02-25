import os
import re
import json
from typing import Dict, Any, Optional, Tuple, Union
from difflib import get_close_matches

from scientist.utils.utils import load_yaml
from scientist.base_agent.formatters import StepPlan

class Planner:
    def __init__(self, llm: Any, role_prompt: str, prompts_path: str = None, logger=None, **kwargs):
        self.llm = llm
        self.role_prompt = role_prompt
        self.logger = logger
        self.prompts = self._load_prompts(prompts_path)
        self.is_validation = kwargs.pop('is_validation', False)
    
    def _load_prompts(self, prompts_path: str = None) -> Dict[str, str]:
        if prompts_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, 'prompts', 'planner.yaml')
        
        return load_yaml(prompts_path)
    
    def generate_global_plan(self, agent_query: str, input_files: Dict[str, Any], available_tools: list, toolbox_metadata: dict) -> str:
        """
        Generate a global plan for the entire query.
        """
        prompt = self.prompts.get('global_planning', '').format(
            agent_query=agent_query,
            available_tools=available_tools,
            toolbox_metadata=json.dumps(toolbox_metadata, indent=4),
            file_info=json.dumps(input_files, indent=4) if input_files else "No input files available"
        )
        
        # Generate the global plan (response_format will be discarded for models that don't support it)
        response = self.llm.generate(
            content=prompt,
            system_prompt=self.role_prompt
        )

        return str(response).strip()
        
    def generate_one_step_plan(self, agent_query: str, input_files: Dict[str, Any], available_tools: list, toolbox_metadata: dict, 
                          global_plan: str, memory, step_count: int, max_steps: int) -> str:
        """
        Generate a one-step plan.
        """
        previous_steps = memory.get_concise_steps()
        
        prompt = self.prompts.get('step_planning', '').format(
            agent_query=agent_query,
            global_plan=global_plan,
            available_tools=available_tools,
            toolbox_metadata=json.dumps(toolbox_metadata, indent=4),
            file_info=json.dumps(input_files, indent=4) if input_files else "No input files available",
            previous_steps=previous_steps,
            current_step=step_count,
            max_steps=max_steps,
            remaining_steps=max_steps - step_count - 1
        )

        # self.logger.info(f"==> Prompt (generate_one_step_plan): \n{prompt}")
        
        # Generate the one-step plan
        one_step_plan = self.llm.generate(
            content=prompt,
            system_prompt=self.role_prompt,
            response_format=StepPlan
        )
        
        return one_step_plan
    
    def parse_next_step(self, local_plan: Union[str, StepPlan]) -> Tuple[str, str, str]:
        """Parse the local plan to extract the next action details using multiple methods."""
        
        # Method 1: Handle StepPlan object directly (like reference code)
        if isinstance(local_plan, StepPlan):
            tool_name = self._normalize_tool_name(local_plan.tool_name)
            return (
                tool_name,
                local_plan.step_goal.strip(),
                local_plan.step_context.strip()
            )
        
        # Method 2: Regex parsing (from octotools script)
        try:
            text = local_plan.replace("**", "")
            # Pattern to match the exact format
            pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)(?=\n\n|\Z)"
            matches = re.findall(pattern, text, re.DOTALL)
            
            if matches:
                # Return the last match (most recent/relevant)
                context, sub_goal, tool_name = matches[-1]
                normalized_tool_name = self._normalize_tool_name(tool_name.strip())
                return (
                    normalized_tool_name,
                    sub_goal.strip(), 
                    context.strip()
                )
        except:
            pass
        
        # Method 3: More flexible regex patterns (not verified at scale)
        try:
            # Try to find tool name
            tool_pattern = r"(?:Tool Name|Tool):\s*([^\n]+)"
            tool_match = re.search(tool_pattern, local_plan, re.IGNORECASE)
            
            # Try to find goal
            goal_pattern = r"(?:Sub-Goal|Goal|Step Goal):\s*([^\n]+)"
            goal_match = re.search(goal_pattern, local_plan, re.IGNORECASE)
            
            # Try to find context
            context_pattern = r"(?:Context|Step Context):\s*(.*?)(?=(?:Sub-Goal|Goal|Tool Name|$))"
            context_match = re.search(context_pattern, local_plan, re.IGNORECASE | re.DOTALL)
            
            if tool_match and goal_match and context_match:
                normalized_tool_name = self._normalize_tool_name(tool_match.group(1).strip())
                return (
                    normalized_tool_name,
                    goal_match.group(1).strip(),
                    context_match.group(1).strip()
                )
        except:
            pass
        
        raise ValueError(f"Invalid local plan format, could not parse: {local_plan}")
    
    def _normalize_tool_name(self, tool_name: str) -> str:
        """
        Main entry point for tool name normalization.
        Dispatches to robust or simple logic based on self.is_val.
        """
        if self.is_validation:
            return self._normalize_tool_name_robust(tool_name)
        else:
            return self._normalize_tool_name_simple(tool_name)

    def _normalize_tool_name_simple(self, tool_name: str) -> str:
        """
        Logic for is_val = False (The first version provided)
        Strategies: Direct -> Case Insensitive -> Partial Match
        """
        tool_name = tool_name.strip()
        
        # Get available tools from the planner's context
        available_tools = getattr(self, 'available_tools', [])
        
        # Direct match
        if tool_name in available_tools:
            return tool_name
        
        # Case insensitive match
        for tool in available_tools:
            if tool.lower() == tool_name.lower():
                return tool
        
        # Partial match
        for tool in available_tools:
            if tool.lower() in tool_name.lower() or tool_name.lower() in tool.lower():
                return tool
        
        return f"No matched tool given: {tool_name}"

    def _normalize_tool_name_robust(self, tool_name: str) -> str:
        """
        Logic for is_val = True (The second version provided)
        Strategies: Exact -> Case Insensitive -> Canonical (ignore separators) -> Fuzzy
        """
        tool_name = tool_name.strip()
        
        available_tools = getattr(self, 'available_tools', [])

        # Direct match
        if tool_name in available_tools:
            return tool_name
        
        # Case Insensitive match
        tool_map_lower = {t.lower(): t for t in available_tools}
        if tool_name.lower() in tool_map_lower:
            return tool_map_lower[tool_name.lower()]
        
        # Canonical match
        def to_canonical(s):
            return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

        canonical_input = to_canonical(tool_name)
        canonical_map = {to_canonical(t): t for t in available_tools}
        
        if canonical_input in canonical_map:
            return canonical_map[canonical_input]

        # Fuzzy match
        matches = get_close_matches(tool_name, available_tools, n=1, cutoff=0.8)
        if matches:
            return matches[0]
        
        return f"No matched tool given: {tool_name}"