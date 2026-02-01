"""
Missing Values Handling Module for AURA Preprocessor 2.0

Handles missing values with multiple strategies and comprehensive reporting.
Supports both interactive and automatic modes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """
    Handles missing values in datasets with multiple strategies.
    """
    
    def __init__(self, mode: str = "auto", llm_recommendations: Optional[Dict] = None):
        """
        Initialize the missing value handler.
        
        Args:
            mode: Execution mode - "auto" or "step"
            llm_recommendations: LLM recommendations for missing value handling
        """
        self.mode = mode
        self.llm_recommendations = llm_recommendations
        self.handling_info = {}  # Store handling decisions for reporting
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Process missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (processed_df, handling_info)
        """
        df_processed = df.copy()
        missing = df_processed.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            logger.info("No missing values detected")
            print("✨ No missing values detected.")
            return df_processed, self.handling_info

        print("\n🔍 Missing Values Detected:")
        print(missing)
        logger.info(f"Found missing values in {len(missing)} columns")

        for col, count in missing.items():
            df_processed, col_info = self._handle_column(df_processed, col, count)
            self.handling_info[col] = col_info

        print("\n✨ Missing value handling completed.")
        return df_processed, self.handling_info
    
    def _handle_column(self, df: pd.DataFrame, col: str, count: int) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Handle missing values for a single column.
        
        Args:
            df: DataFrame containing the column
            col: Column name
            count: Number of missing values
            
        Returns:
            Tuple of (processed_df, column_info)
        """
        perc = (count / len(df)) * 100
        print(f"\n⚠️ Column: {col} → {count} missing ({perc:.2f}%)")
        
        col_info = {
            "original_column": col,
            "missing_count": count,
            "missing_percentage": perc,
            "data_type": str(df[col].dtype),
            "handling_method": None,
            "action_taken": None
        }
        
        if self.mode == "step":
            choice = self._get_user_choice(col, perc)
        else:
            choice = self._get_auto_choice(col, perc, df)
        
        # Apply the chosen method
        if choice == "1":  # Drop column
            df, col_info = self._drop_column(df, col, col_info)
        elif choice == "2":  # Fill with mean
            df, col_info = self._fill_with_mean(df, col, col_info)
        elif choice == "3":  # Fill with median
            df, col_info = self._fill_with_median(df, col, col_info)
        elif choice == "4":  # Fill with mode
            df, col_info = self._fill_with_mode(df, col, col_info)
        elif choice == "5":  # Skip
            col_info["handling_method"] = "skipped"
            col_info["action_taken"] = "No action taken"
            print(f"⏭️ Skipped {col}")
        else:
            col_info["handling_method"] = "error"
            col_info["action_taken"] = "Invalid choice, skipped"
            print(f"⚠️ Invalid choice for {col}, skipped.")
        
        return df, col_info
    
    def _get_user_choice(self, col: str, perc: float) -> str:
        """
        Get user choice for handling method in step mode.
        
        Args:
            col: Column name
            perc: Missing percentage
            
        Returns:
            User's choice as string
        """
        print("Options:")
        print("   1) Drop column")
        print("   2) Fill with mean (numeric only)")
        print("   3) Fill with median (numeric only)")
        print("   4) Fill with mode")
        print("   5) Skip")
        
        while True:
            choice = input(f"👉 Enter choice for {col}: ").strip()
            if choice in ["1", "2", "3", "4", "5"]:
                return choice
            print("⚠️ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
    
    def _get_auto_choice(self, col: str, perc: float, df: pd.DataFrame) -> str:
        """
        Automatically choose handling method based on LLM recommendations or heuristics.
        
        Args:
            col: Column name
            perc: Missing percentage
            df: DataFrame for analysis
            
        Returns:
            Auto-selected choice
        """
        # Check if LLM has specific recommendation for this column
        if self.llm_recommendations and "columns" in self.llm_recommendations:
            column_recs = self.llm_recommendations["columns"]
            if col in column_recs:
                strategy = column_recs[col].lower()
                logger.info(f"Using LLM recommendation for {col}: {strategy}")
                print(f"🤖 LLM recommends: {strategy} for {col}")
                
                if strategy in ["drop", "remove"]:
                    return "1"
                elif strategy == "mean":
                    return "2"
                elif strategy == "median":
                    return "3"
                elif strategy in ["mode", "most_frequent"]:
                    return "4"
        
        # Check for general strategy recommendation from LLM
        if self.llm_recommendations and "strategy" in self.llm_recommendations:
            general_strategy = self.llm_recommendations["strategy"].lower()
            
            # High missing percentage (>50%) - drop column
            if perc > 50:
                logger.info(f"Auto-dropping column {col} (high missing percentage: {perc:.2f}%)")
                return "1"
            
            # Apply general LLM strategy based on column type
            if df[col].dtype in ["float64", "int64"]:
                if general_strategy == "mean":
                    logger.info(f"LLM: Using mean for numeric column {col}")
                    return "2"
                elif general_strategy == "median":
                    logger.info(f"LLM: Using median for numeric column {col}")
                    return "3"
                else:
                    # Default to mean for numeric
                    return "2"
            else:
                # Categorical - use mode
                logger.info(f"LLM: Using mode for categorical column {col}")
                return "4"
        
        # Fallback to original heuristics if no LLM recommendations
        # High missing percentage (>50%) - drop column
        if perc > 50:
            logger.info(f"Auto-dropping column {col} (high missing percentage: {perc:.2f}%)")
            return "1"
        
        # Numeric columns - use mean
        if df[col].dtype in ["float64", "int64"]:
            logger.info(f"Auto-filling numeric column {col} with mean")
            return "2"
        
        # Categorical columns - use mode
        else:
            logger.info(f"Auto-filling categorical column {col} with mode")
            return "4"
    
    def _drop_column(self, df: pd.DataFrame, col: str, col_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Drop a column with missing values."""
        df = df.drop(columns=[col])
        col_info["handling_method"] = "dropped"
        col_info["action_taken"] = f"Dropped column {col}"
        print(f"🗑️ Dropped column {col}")
        logger.info(f"Dropped column {col}")
        return df, col_info
    
    def _fill_with_mean(self, df: pd.DataFrame, col: str, col_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Fill missing values with mean."""
        if df[col].dtype in ["float64", "int64"]:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            col_info["handling_method"] = "mean_fill"
            col_info["action_taken"] = f"Filled with mean: {mean_val:.4f}"
            col_info["fill_value"] = mean_val
            print(f"✅ Filled {col} with mean: {mean_val:.4f}")
            logger.info(f"Filled {col} with mean: {mean_val:.4f}")
        else:
            col_info["handling_method"] = "error"
            col_info["action_taken"] = "Cannot fill non-numeric column with mean"
            print(f"⚠️ Cannot fill non-numeric column {col} with mean")
        return df, col_info
    
    def _fill_with_median(self, df: pd.DataFrame, col: str, col_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Fill missing values with median."""
        if df[col].dtype in ["float64", "int64"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            col_info["handling_method"] = "median_fill"
            col_info["action_taken"] = f"Filled with median: {median_val:.4f}"
            col_info["fill_value"] = median_val
            print(f"✅ Filled {col} with median: {median_val:.4f}")
            logger.info(f"Filled {col} with median: {median_val:.4f}")
        else:
            col_info["handling_method"] = "error"
            col_info["action_taken"] = "Cannot fill non-numeric column with median"
            print(f"⚠️ Cannot fill non-numeric column {col} with median")
        return df, col_info
    
    def _fill_with_mode(self, df: pd.DataFrame, col: str, col_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Fill missing values with mode."""
        try:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            col_info["handling_method"] = "mode_fill"
            col_info["action_taken"] = f"Filled with mode: {mode_val}"
            col_info["fill_value"] = str(mode_val)
            print(f"✅ Filled {col} with mode: {mode_val}")
            logger.info(f"Filled {col} with mode: {mode_val}")
        except Exception as e:
            col_info["handling_method"] = "error"
            col_info["action_taken"] = f"Error filling with mode: {str(e)}"
            print(f"⚠️ Error filling {col} with mode: {str(e)}")
        return df, col_info


def process(df: pd.DataFrame, mode: str = "auto") -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Convenience function to process missing values.
    
    Args:
        df: Input DataFrame
        mode: Execution mode
        
    Returns:
        Tuple of (processed_df, handling_info)
    """
    handler = MissingValueHandler(mode)
    return handler.process(df)
