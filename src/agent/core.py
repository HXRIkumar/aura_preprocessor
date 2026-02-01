"""
AURA Agent Core
===============
This module implements the central brain of the agentic system.

It defines:
1. AgentState: The persisted memory of the agent (strictly safe, no raw data).
2. AgentLoop: The Observe-Reason-Act cycle.

Architecture:
- The Agent NEVER sees raw data rows.
- The Agent operates purely on metadata snapshots.
- Decisions are made by the LLM and executed via `src.agent.tools`.

Safety:
- Max steps limit enforced.
- PrivacyViolationError aborts execution immediately.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import Privacy Firewall and Tools
from src.agent.sanitizer import PrivacyViolationError
from src.agent.tools import (
    inspect_metadata, 
    run_preprocessing_step
)
from src.llm_service import get_llm_service

logger = logging.getLogger(__name__)

# =============================================================================
# Constants & System Prompt
# =============================================================================

MAX_STEPS = 15

SYSTEM_PROMPT = """You are AURA, an expert Data Engineer Agent.
Your goal is to preprocess a dataset to make it ready for machine learning.

CRITICAL RULES:
1. You NEVER see raw data. You only work with metadata.
2. You must return your response in strictly VALID JSON format.
3. You must step-by-step improve the data quality.

AVAILABLE ACTIONS:
- "inspect_metadata": Get column types, missing values, and stats.
- "run_preprocessing_step": Execute a transformation.
    - action="impute": Fill missing values. 
      Params: strategy="mean"|"median"|"mode"|"drop", columns={"col_name": "strategy"}
    - action="encode": Encode categorical features.
      Params: strategy="onehot"|"label", columns={"col_name": "strategy"}
    - action="scale": Scale numerical features.
      Params: strategy="standard"|"minmax"|"robust"
    - action="drop_col": Remove columns.
      Params: columns=["col1", "col2"]
- "DONE": Signal that preprocessing is complete.

RESPONSE FORMAT:
{
  "thought": "Reasoning about the current state and what to do next...",
  "action": "ACTION_NAME",
  "params": { ... parameters for the action ... }
}

Example:
{
  "thought": "The 'age' column has 5% missing values. I should fill them with the median.",
  "action": "run_preprocessing_step",
  "params": {
    "action": "impute",
    "strategy": "median",
    "columns": {"age": "median"}
  }
}
"""

# =============================================================================
# Agent State Definition
# =============================================================================

@dataclass
class AgentState:
    """
    Persisted state of the agent session.
    STRICTLY FORBIDDEN: Raw Pandas DataFrames or large lists of values.
    """
    dataset_id: str
    status: str = "PLANNING"  # PLANNING, EXECUTING, WAITING_USER, DONE, FAILED
    step_count: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata_snapshot: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[List[str]] = None
    last_error: Optional[str] = None
    
    def add_message(self, role: str, content: Any):
        """Append a message to history."""
        self.messages.append({
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "content": content
        })


# =============================================================================
# Agent Core Logic
# =============================================================================

class AuraAgent:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.state = AgentState(dataset_id=dataset_id)
        self.llm_service = get_llm_service()
        
        # Initialize state with metadata
        logger.info(f"Initializing Agent for dataset {dataset_id}")
        self._refresh_metadata()

    def _refresh_metadata(self):
        """Helper to refresh the metadata snapshot from the tool."""
        try:
            meta = inspect_metadata(self.dataset_id)
            if "error" in meta:
                self.state.last_error = meta["error"]
            else:
                self.state.metadata_snapshot = meta
        except Exception as e:
            logger.error(f"Failed to refresh metadata: {e}")
            self.state.last_error = str(e)

    def run(self) -> AgentState:
        """
        Main Agent Loop: Observe -> Reason -> Act -> Repeat.
        """
        self.state.status = "EXECUTING"
        logger.info(f"🤖 Agent started for dataset {self.dataset_id}")
        
        while self.state.status == "EXECUTING":
            self.state.step_count += 1
            
            # 1. Check Safety Limits
            if self.state.step_count > MAX_STEPS:
                logger.warning("Max steps reached. Stopping agent.")
                self.state.status = "DONE"
                self.state.add_message("system", "Max steps reached. Forcing completion.")
                break

            # 2. OBSERVE & REASON
            try:
                # Construct prompt context
                prompt = self._build_prompt()
                
                # Call LLM
                response_str = self.llm_service.chat(prompt, dataset_context=None)
                
                # Parse Decision
                decision = self._parse_llm_response(response_str)
                self.state.add_message("assistant", decision)
                
                logger.info(f"🧠 Step {self.state.step_count}: {decision.get('thought')}")
                logger.info(f"   Action: {decision.get('action')}")

            except Exception as e:
                logger.error(f"Reasoning failed: {e}")
                self.state.last_error = f"Reasoning error: {str(e)}"
                continue

            # 3. ACT & EXECUTE
            action = decision.get("action")
            params = decision.get("params", {})
            
            if action == "DONE":
                self.state.status = "DONE"
                logger.info("✅ Agent decided task is complete.")
                break
                
            result = self._execute_tool(action, params)
            
            # 4. UPDATE
            self.state.add_message("tool", result)
            
            # If tool modified data, refresh metadata
            if action == "run_preprocessing_step" and result.get("status") == "success":
                self._refresh_metadata()
                
            # Check for critical errors
            if "error" in result:
                logger.error(f"❌ Tool Error: {result['error']}")
        
        return self.state

    def _build_prompt(self) -> str:
        """Constructs the prompt with current state context."""
        context = {
            "step": self.state.step_count,
            "metadata_summary": self.state.metadata_snapshot,
            "recent_history": self.state.messages[-3:], # Last 3 messages for context
            "last_error": self.state.last_error
        }
        
        return f"""
{SYSTEM_PROMPT}

CURRENT STATE:
{json.dumps(context, indent=2)}

Decide the next step. Respond with JSON only.
"""

    def _parse_llm_response(self, response_str: str) -> Dict[str, Any]:
        """Safely parses LLM JSON response."""
        try:
            # Clean generic markdown fences if present
            clean_str = response_str.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            
            return json.loads(clean_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON. Retrying or defaulting.")
            # Simple fallback or retry logic could go here.
            # For now, return a no-op to prevent crash
            return {
                "thought": "I failed to format my response as JSON. I should try again.",
                "action": "inspect_metadata", # Safe fallback
                "params": {}
            }

    def _execute_tool(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the chosen tool and handles safety."""
        try:
            if action == "inspect_metadata":
                return inspect_metadata(self.dataset_id)
            
            elif action == "run_preprocessing_step":
                # Extract inner action for specific step
                step_action = params.get("action") # e.g., "impute"
                # Remove 'action' from params to avoid duplication if passed
                clean_params = {k:v for k,v in params.items() if k != "action"}
                
                if not step_action:
                    return {"error": "Missing 'action' parameter for run_preprocessing_step"}
                    
                return run_preprocessing_step(
                    self.dataset_id, 
                    step_action, 
                    clean_params
                )
            
            else:
                return {"error": f"Unknown tool action: {action}"}
                
        except PrivacyViolationError as e:
            logger.critical(f"🚨 PRIVACY VIOLATION STOPPED: {e}")
            self.state.status = "FAILED"
            self.state.last_error = f"Privacy Violation: {str(e)}"
            return {"error": "Privacy Violation detected. Action blocked."}
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
