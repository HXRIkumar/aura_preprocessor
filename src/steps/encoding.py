"""
Feature Encoding Module for AURA Preprocessor 2.0

Handles categorical feature encoding with multiple strategies.
Supports both interactive and automatic modes.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Handles encoding of categorical features with multiple strategies.
    """
    
    def __init__(self, mode: str = "auto", llm_recommendations: Optional[Dict] = None):
        """
        Initialize the feature encoder.
        
        Args:
            mode: Execution mode - "auto" or "step"
            llm_recommendations: LLM recommendations for encoding
        """
        self.mode = mode
        self.llm_recommendations = llm_recommendations
        self.encoding_info = {}  # Store encoding decisions for reporting
    
    def encode_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Encode categorical features in the dataset.
        
        Args:
            df: Input DataFrame
            target_col: Optional target column name to preserve during encoding
            
        Returns:
            Tuple of (encoded_df, encoding_info)
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from encoding if it exists and is categorical
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
            logger.info(f"Preserving target column '{target_col}' from encoding")
            
            # But we still need to encode it for the model - apply label encoding to target
            if target_col in df_encoded.columns and df_encoded[target_col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[target_col] = le.fit_transform(df_encoded[target_col])
                self.encoding_info[target_col] = {
                    "original_column": target_col,
                    "unique_values": len(le.classes_),
                    "encoding_method": "label_encoding (target)",
                    "new_columns": [],
                    "is_target": True
                }
                logger.info(f"Applied label encoding to target column '{target_col}'")
        
        if not categorical_cols:
            logger.info("No categorical feature columns found for encoding")
            return df_encoded, self.encoding_info
        
        logger.info(f"Found {len(categorical_cols)} categorical feature columns: {categorical_cols}")
        
        for col in categorical_cols:
            df_encoded, col_info = self._encode_column(df_encoded, col)
            self.encoding_info[col] = col_info
        
        return df_encoded, self.encoding_info
    
    def _encode_column(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Encode a single categorical column.
        
        Args:
            df: DataFrame containing the column
            col: Column name to encode
            
        Returns:
            Tuple of (encoded_df, column_info)
        """
        unique_values = df[col].nunique()
        col_info = {
            "original_column": col,
            "unique_values": unique_values,
            "encoding_method": None,
            "new_columns": []
        }
        
        if self.mode == "step":
            choice = self._get_user_choice(col, unique_values)
        else:
            choice = self._get_auto_choice(col, unique_values, df)
        
        if choice == "1":  # Label Encoding
            df, col_info = self._apply_label_encoding(df, col, col_info)
        elif choice == "2":  # One-Hot Encoding
            df, col_info = self._apply_onehot_encoding(df, col, col_info)
        elif choice == "3":  # Drop column
            df = df.drop(columns=[col])
            col_info["encoding_method"] = "dropped"
            logger.info(f"Dropped column: {col}")
            print(f"🗑️  Dropped column '{col}' (not useful for model)")
        else:
            col_info["encoding_method"] = "skipped"
            logger.warning(f"Invalid choice for column {col}, skipping")
        
        return df, col_info
    
    def _get_user_choice(self, col: str, unique_values: int) -> str:
        """
        Get user choice for encoding method in step mode.
        
        Args:
            col: Column name
            unique_values: Number of unique values
            
        Returns:
            User's choice as string
        """
        print(f"\n⚡ Encoding options for column '{col}' ({unique_values} unique values):")
        print("   1) Label Encode (recommended for ordinal data)")
        print("   2) One-Hot Encode (recommended for nominal data)")
        print("   3) Drop column (remove from dataset)")
        
        while True:
            choice = input(f"👉 Enter choice for {col}: ").strip()
            if choice in ["1", "2", "3"]:
                return choice
            print("⚠️ Invalid choice. Please enter 1, 2, or 3.")
    
    def _get_auto_choice(self, col: str, unique_values: int, df: pd.DataFrame) -> str:
        """
        Automatically choose encoding method based on LLM recommendations or heuristics.
        
        Args:
            col: Column name
            unique_values: Number of unique values
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
                
                if strategy in ["drop", "remove", "skip"]:
                    return "3"  # Skip this column
                elif strategy in ["label", "label_encoding"]:
                    return "1"
                elif strategy in ["onehot", "one-hot", "onehot_encoding"]:
                    return "2"
        
        # Check for general strategy recommendation from LLM
        if self.llm_recommendations and "strategy" in self.llm_recommendations:
            general_strategy = self.llm_recommendations["strategy"].lower()
            logger.info(f"Using LLM general encoding strategy: {general_strategy}")
            
            if general_strategy in ["label", "label_encoding"]:
                return "1"
            elif general_strategy in ["onehot", "one-hot", "onehot_encoding"]:
                return "2"
        
        # Fallback to original heuristics if no LLM recommendations
        # Heuristic: Use one-hot for high cardinality, label for low cardinality
        if unique_values > 10:
            logger.info(f"Auto-selecting one-hot encoding for {col} (high cardinality: {unique_values})")
            return "2"
        else:
            logger.info(f"Auto-selecting label encoding for {col} (low cardinality: {unique_values})")
            return "1"
    
    def _apply_label_encoding(self, df: pd.DataFrame, col: str, col_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply label encoding to a column.
        
        Args:
            df: DataFrame
            col: Column name
            col_info: Column information dictionary
            
        Returns:
            Tuple of (encoded_df, updated_col_info)
        """
        try:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
            col_info["encoding_method"] = "label_encoding"
            col_info["new_columns"] = [col]
            col_info["label_mapping"] = dict(zip(le.classes_, le.transform(le.classes_)))
            
            print(f"✅ Label encoded '{col}'")
            logger.info(f"Applied label encoding to {col}")
            
        except Exception as e:
            logger.error(f"Error in label encoding for {col}: {str(e)}")
            col_info["encoding_method"] = "error"
            col_info["error"] = str(e)
        
        return df, col_info
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, col: str, col_info: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply one-hot encoding to a column.
        
        Args:
            df: DataFrame
            col: Column name
            col_info: Column information dictionary
            
        Returns:
            Tuple of (encoded_df, updated_col_info)
        """
        try:
            # Get original unique values for reporting
            original_values = df[col].unique().tolist()
            
            # Apply one-hot encoding
            df_encoded = pd.get_dummies(df, columns=[col], prefix=col)
            
            # Get new column names
            new_cols = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
            
            col_info["encoding_method"] = "onehot_encoding"
            col_info["new_columns"] = new_cols
            col_info["original_values"] = original_values
            
            print(f"✅ One-hot encoded '{col}' → {len(new_cols)} new columns")
            logger.info(f"Applied one-hot encoding to {col}, created {len(new_cols)} columns")
            
            return df_encoded, col_info
            
        except Exception as e:
            logger.error(f"Error in one-hot encoding for {col}: {str(e)}")
            col_info["encoding_method"] = "error"
            col_info["error"] = str(e)
            return df, col_info


def encode_categorical_features(df: pd.DataFrame, mode: str = "auto") -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Convenience function to encode categorical features.
    
    Args:
        df: Input DataFrame
        mode: Execution mode
        
    Returns:
        Tuple of (encoded_df, encoding_info)
    """
    encoder = FeatureEncoder(mode)
    return encoder.encode_features(df)

