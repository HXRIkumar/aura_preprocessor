"""
AURA Agent Sanitizer
====================
This module acts as the "Privacy Firewall" between the raw data processing layer
and the LLM Agent. It ensures that NO raw data, PII, or large datasets are ever
exposed to the agent's context.

Responsibility:
1. Extract safe, aggregated metadata from Pandas DataFrames.
2. Sanitize tool outputs to remove unsafe objects (DataFrames, arrays).
3. Enforce cardinality thresholds for categorical data.

Design Rules:
- Input: Raw DataFrame or Tool Output Dict
- Output: JSON-serializable Dictionary (Safe for LLM)
- Exception: Raise PrivacyViolationError if leaks are detected or rules violated.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import json

# =============================================================================
# Configuration & Constants
# =============================================================================

SAFE_AGGREGATES = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
CATEGORICAL_CARDINALITY_THRESHOLD = 20  # Max unique values to show for a category
MAX_LIST_LENGTH = 100  # Max length for any returned list in tool output


class PrivacyViolationError(Exception):
    """Raised when a tool attempts to return unsafe data types or raw PII."""
    pass


# =============================================================================
# Metadata Extraction (Input Firewall)
# =============================================================================

def extract_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts purely statistical metadata from a DataFrame.
    
    GUARANTEES:
    - No raw rows are returned.
    - No string values are returned for high-cardinality (>20) columns.
    - All numerics are aggregated (mean, std, etc.).
    
    Args:
        df: The pandas DataFrame to analyze.
        
    Returns:
        A dictionary containing safe metadata (shape, columns, types, stats).
    """
    if df is None:
        return {"error": "Dataset is None"}
        
    metadata = {
        "shape": list(df.shape),
        "columns": {},
        "summary": {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
        }
    }

    # Column-wise Analysis
    for col in df.columns:
        col_type = str(df[col].dtype)
        col_data = df[col]
        missing_count = int(col_data.isnull().sum())
        missing_pct = float((missing_count / len(df)) * 100) if len(df) > 0 else 0.0
        
        col_info = {
            "dtype": col_type,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4)
        }

        # Handle Numeric Columns
        if pd.api.types.is_numeric_dtype(col_data):
            # Calculate aggregates safely
            desc = col_data.describe(percentiles=[.25, .5, .75]).to_dict()
            safe_stats = {k: v for k, v in desc.items() if k in SAFE_AGGREGATES or k == 'count'}
            # Convert numpy floats to python floats for JSON
            col_info["stats"] = {k: float(v) for k, v in safe_stats.items()}

        # Handle Categorical / Object Columns
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            unique_count = int(col_data.nunique())
            col_info["unique_count"] = unique_count
            
            # PRIVACY CHECK: Only show values if cardinality is low
            if unique_count <= CATEGORICAL_CARDINALITY_THRESHOLD:
                # Safe to show value counts
                try:
                    # Get top values, convert index (keys) to string to ensure JSON safety
                    val_counts = col_data.value_counts().head(CATEGORICAL_CARDINALITY_THRESHOLD).to_dict()
                    col_info["value_counts"] = {str(k): int(v) for k, v in val_counts.items()}
                except Exception:
                    col_info["value_counts"] = "Error extracting counts"
            else:
                # REDACTED due to high cardinality
                col_info["value_counts"] = "[HIGH_CARDINALITY_REDACTED]"
                col_info["most_frequent_note"] = "Values hidden for privacy (too many unique values)"

        metadata["columns"][col] = col_info

    return metadata


# =============================================================================
# Output Sanitization (Output Firewall)
# =============================================================================

def sanitize_tool_output(output: Any) -> Any:
    """
    Recursively scans and cleans tool outputs to enforce privacy rules.
    
    RULES:
    1. No pandas Objects (DataFrame/Series).
    2. No numpy Arrays.
    3. No lists > MAX_LIST_LENGTH.
    4. Convert numpy scalars to Python types.
    
    Args:
        output: Raw output from a tool function.
        
    Returns:
        Sanitized, JSON-serializable object.
        
    Raises:
        PrivacyViolationError: If a DataFrame or massive list is detected.
    """
    # Rule 1: Block DataFrames and Series
    if isinstance(output, (pd.DataFrame, pd.Series)):
        raise PrivacyViolationError(
            f"Privacy Violation: Tool attempted to return a raw pandas {type(output).__name__}. "
            "Tools must return metadata or operation summaries only."
        )

    # Rule 2: Handle Numpy Arrays (Convert or Block if too large)
    if isinstance(output, np.ndarray):
        if output.size > MAX_LIST_LENGTH:
            raise PrivacyViolationError(
                f"Privacy Violation: Tool returned a numpy array with {output.size} elements. "
                f"Limit is {MAX_LIST_LENGTH}."
            )
        return output.tolist()

    # Rule 3: Numpy Scalars -> Python Scalars
    if isinstance(output, np.generic):
        return output.item()

    # Recursive steps for Container Types
    if isinstance(output, dict):
        return {str(k): sanitize_tool_output(v) for k, v in output.items()}
    
    if isinstance(output, list):
        if len(output) > MAX_LIST_LENGTH:
             raise PrivacyViolationError(
                f"Privacy Violation: Tool returned a list with {len(output)} elements. "
                f"Limit is {MAX_LIST_LENGTH}."
            )
        return [sanitize_tool_output(item) for item in output]
    
    if isinstance(output, tuple):
        return tuple(sanitize_tool_output(item) for item in output)

    # Basic types pass through
    return output


# =============================================================================
# Example Usage (Commented)
# =============================================================================
#
# df = pd.DataFrame({
#     "age": [25, 30, 35, np.nan],
#     "city": ["New York", "London", "Paris", "London"]
# })
# 
# # 1. Extract Safe Metadata
# metadata = extract_metadata(df)
# # Result:
# # {
# #   "columns": {
# #      "age": {"stats": {"mean": 30.0, ...}, "missing_pct": 25.0},
# #      "city": {"value_counts": {"London": 2, "New York": 1, ...}}  <-- Allowed (Low Card)
# #   }
# # }
#
# # 2. Sanitize Output
# try:
#     sanitize_tool_output(df)  # Raises PrivacyViolationError!
# except PrivacyViolationError as e:
#     print("Blocked DF Leak!")
#
# safe_output = sanitize_tool_output({"status": "success", "rows_filled": 10})
# # Result: {"status": "success", "rows_filled": 10}
