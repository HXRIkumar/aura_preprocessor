"""
AURA Agent Tools
================
This module defines the implementation of tools available to the Agent.
It acts as a wrapper around the existing logic in `src/steps/`, ensuring
that all inputs are correctly formatted and all outputs are sanitized
via the Privacy Firewall.

Capabilities:
- simple internal DataManager (DATA_STORE)
- inspect_metadata(dataset_id)
- run_preprocessing_step(dataset_id, action, params)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union

from src.agent.sanitizer import (
    extract_metadata, 
    sanitize_tool_output, 
    PrivacyViolationError
)

# Import existing core logic
from src.steps.missing_values import MissingValueHandler
from src.steps.encoding import FeatureEncoder
from src.steps.scaling import FeatureScaler

logger = logging.getLogger(__name__)

# =============================================================================
# Internal Data Store (Simple In-Memory Manager)
# =============================================================================
# In a production system, this would be a database or Redis cache.
# For this prototype, a global dictionary suffices.

# =============================================================================
# Internal Data Store (Simple In-Memory Manager)
# =============================================================================
# In a production system, this would be a database or Redis cache.
# For this prototype, a global dictionary suffices.

DATA_STORE: Dict[str, pd.DataFrame] = {}

def register_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    """Internal helper to load a dataset into the tool memory."""
    DATA_STORE[dataset_id] = df.copy()
    logger.info(f"Registered dataset {dataset_id} in Agent Tool Store")

def get_dataset(dataset_id: str) -> pd.DataFrame:
    """Retrieve dataset by ID. Raises ValueError if not found."""
    if dataset_id not in DATA_STORE:
        raise ValueError(f"Dataset {dataset_id} not found in active memory.")
    return DATA_STORE[dataset_id]

def update_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    """Centralized function to update dataset state."""
    DATA_STORE[dataset_id] = df.copy()
    logger.info(f"Updated dataset {dataset_id} in Agent Tool Store")

# =============================================================================
# Tool Definitions
# =============================================================================

def inspect_metadata(dataset_id: str) -> Dict[str, Any]:
    """
    Tool: Inspect the metadata of a dataset.
    
    Args:
        dataset_id: The UUID of the dataset to inspect.
        
    Returns:
        Sanitized metadata dictionary (columns, types, missing %, stats).
    """
    try:
        df = get_dataset(dataset_id)
        # firewall: extract safe metadata
        metadata = extract_metadata(df)
        return sanitize_tool_output(metadata)
    except PrivacyViolationError:
        raise
    except Exception as e:
        logger.error(f"Error in inspect_metadata: {e}")
        return {"error": str(e)}


def run_preprocessing_step(
    dataset_id: str, 
    action: str, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Tool: Execute a preprocessing step on the dataset.
    
    Args:
        dataset_id: The UUID of the dataset.
        action: One of ["impute", "encode", "scale", "drop_col"].
        params: Dictionary of parameters for the action.
                Example for impute: {"strategy": "mean", "columns": {"age": "mean"}}
    
    Returns:
        Sanitized summary of the operation.
    """
    try:
        df = get_dataset(dataset_id)
        result_info = {}
        
        # ---------------------------------------------------------
        # Action: IMPUTE (Missing Values)
        # ---------------------------------------------------------
        if action == "impute":
            # Map params to what MissingValueHandler expects
            # It expects llm_recommendations to contain "strategy" or "columns" keys
            llm_rec = {
                "strategy": params.get("strategy", "mean"),
                "columns": params.get("columns", {})
            }
            
            # Use "auto" mode to bypass interactive prompts, passing our specific config
            handler = MissingValueHandler(mode="auto", llm_recommendations=llm_rec)
            
            # Execute
            processed_df, info = handler.process(df)
            
            # Update Store
            update_dataset(dataset_id, processed_df)
            
            result_info = {
                "action": "impute",
                "status": "success",
                "details": info
            }

        # ---------------------------------------------------------
        # Action: ENCODE (Categorical)
        # ---------------------------------------------------------
        elif action == "encode":
            llm_rec = {
                "strategy": params.get("strategy", "onehot"),
                "columns": params.get("columns", {})
            }
            
            encoder = FeatureEncoder(mode="auto", llm_recommendations=llm_rec)
            
            # Detect target column from params if present, else None
            target_col = params.get("target_column")
            
            processed_df, info = encoder.encode_features(df, target_col=target_col)
            
            update_dataset(dataset_id, processed_df)
            
            result_info = {
                "action": "encode",
                "status": "success",
                "details": info
            }

        # ---------------------------------------------------------
        # Action: SCALE (Numerical)
        # ---------------------------------------------------------
        elif action == "scale":
            llm_rec = {
                "strategy": params.get("strategy", "standard")
            }
            
            scaler = FeatureScaler(mode="auto", llm_recommendations=llm_rec)
            target_col = params.get("target_column")
            
            # Note: FeatureScaler returns numpy array (X_scaled), not DF
            X_scaled, info = scaler.scale_features(df, target_col=target_col)
            
            # We must reconstruct the DataFrame to persist it
            # Identify numeric columns that were scaled
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in numeric_cols:
                numeric_cols.remove(target_col)
                
            processed_df = df.copy()
            if numeric_cols and X_scaled is not None:
                # Update the numeric columns with scaled values
                # Caution: X_scaled might lose column info, we assume order is preserved
                if X_scaled.shape[1] == len(numeric_cols):
                    processed_df[numeric_cols] = X_scaled
                else:
                    error_msg = (
                        f"Scaling failed: shape mismatch. Scaler returned {X_scaled.shape}, "
                        f"expected columns match with {len(numeric_cols)} numeric columns."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            update_dataset(dataset_id, processed_df)
            
            result_info = {
                "action": "scale",
                "status": "success",
                "details": info
            }

        # ---------------------------------------------------------
        # Action: DROP COLUMN (Simple Utility)
        # ---------------------------------------------------------
        elif action == "drop_col":
             cols_to_drop = params.get("columns", [])
             if isinstance(cols_to_drop, str):
                 cols_to_drop = [cols_to_drop]
             
             # Validation
             existing_cols = [c for c in cols_to_drop if c in df.columns]
             
             if existing_cols:
                 processed_df = df.drop(columns=existing_cols)
                 update_dataset(dataset_id, processed_df)
                 result_info = {
                     "action": "drop_col",
                     "status": "success",
                     "dropped_columns": existing_cols
                 }
             else:
                 result_info = {
                     "action": "drop_col",
                     "status": "warning",
                     "message": "No matching columns found to drop"
                 }

        else:
            return {"error": f"Unknown action: {action}"}

        # Calculate impact summary
        result_info["new_shape"] = list(get_dataset(dataset_id).shape)
        
        # Firewall: Sanitize Output
        return sanitize_tool_output(result_info)

    except PrivacyViolationError:
        logger.critical("PRIVACY VIOLATION DETECTED")
        raise
    except Exception as e:
        logger.error(f"Error in run_preprocessing_step ({action}): {e}")
        # Firewall: Do not return stack trace, just error message
        return {"error": f"Execution failed: {str(e)}"}
