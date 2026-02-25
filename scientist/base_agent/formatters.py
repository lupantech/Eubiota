from pydantic import BaseModel
from typing import List

# Step Planning - used by planner
class StepPlan(BaseModel):
    step_analysis: str
    tool_name: str
    step_goal: str
    step_context: str
    expected_outcome: str

# Tool Command - used by executor
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str

# Verification and Reflection - used by verifier
class ReflectionResult(BaseModel):
    analysis: str
    stop_signal: bool
