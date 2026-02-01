"""
Feature Scaling Module for AURA Preprocessor 2.0

Handles feature scaling with multiple strategies.
Supports both interactive and automatic modes.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Handles scaling of numerical features with multiple strategies.
    """
    
    def __init__(self, mode: str = "auto", llm_recommendations: Optional[Dict] = None):
        """
        Initialize the feature scaler.
        
        Args:
            mode: Execution mode - "auto" or "step"
            llm_recommendations: LLM recommendations for scaling
        """
        self.mode = mode
        self.llm_recommendations = llm_recommendations
        self.scaling_info = {}  # Store scaling decisions for reporting
        self.scaler = None
    
    def scale_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Scale numerical features in the dataset.
        
        Args:
            df: Input DataFrame
            target_col: Target column name (will be excluded from scaling)
            
        Returns:
            Tuple of (scaled_features, scaling_info)
        """
        # Identify numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if specified
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if not numeric_cols:
            logger.info("No numerical columns found for scaling")
            return df.values, self.scaling_info
        
        logger.info(f"Found {len(numeric_cols)} numerical columns for scaling: {numeric_cols}")
        
        # Extract numerical features
        X = df[numeric_cols].values
        
        if self.mode == "step":
            scaler_type = self._get_user_scaler_choice()
        else:
            scaler_type = self._get_auto_scaler_choice(X)
        
        # Apply scaling
        X_scaled, scaling_info = self._apply_scaling(X, scaler_type, numeric_cols)
        
        return X_scaled, scaling_info
    
    def _get_user_scaler_choice(self) -> str:
        """
        Get user choice for scaling method in step mode.
        
        Returns:
            User's choice as string
        """
        print("\n⚡ Scaling options:")
        print("   1) StandardScaler (mean=0, std=1)")
        print("   2) MinMaxScaler (range 0-1)")
        print("   3) RobustScaler (median=0, IQR=1)")
        print("   4) Skip scaling")
        
        while True:
            choice = input("👉 Enter choice: ").strip()
            if choice in ["1", "2", "3", "4"]:
                return choice
            print("⚠️ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    def _get_auto_scaler_choice(self, X: np.ndarray) -> str:
        """
        Automatically choose scaling method based on LLM recommendations or data characteristics.
        
        Args:
            X: Feature matrix
            
        Returns:
            Auto-selected scaler type
        """
        # Check if LLM has recommendation for scaling
        if self.llm_recommendations and "strategy" in self.llm_recommendations:
            strategy = self.llm_recommendations["strategy"].lower()
            logger.info(f"Using LLM recommendation for scaling: {strategy}")
            print(f"🤖 LLM recommends: {strategy} scaling")
            
            if strategy in ["standard", "standardscaler"]:
                return "1"
            elif strategy in ["minmax", "minmaxscaler"]:
                return "2"
            elif strategy in ["robust", "robustscaler"]:
                return "3"
            elif strategy in ["none", "skip"]:
                return "4"
        
        # Fallback to original heuristics if no LLM recommendations
        # Check for outliers using IQR method
        has_outliers = self._detect_outliers(X)
        
        if has_outliers:
            logger.info("Detected outliers, using RobustScaler")
            return "3"  # RobustScaler
        else:
            logger.info("No significant outliers detected, using StandardScaler")
            return "1"  # StandardScaler
    
    def _detect_outliers(self, X: np.ndarray, threshold: float = 1.5) -> bool:
        """
        Detect outliers using IQR method.
        
        Args:
            X: Feature matrix
            threshold: IQR threshold for outlier detection
            
        Returns:
            True if outliers are detected
        """
        try:
            # Calculate IQR for each feature
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Check for outliers
            outliers = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            outlier_percentage = np.mean(outliers) * 100
            
            logger.info(f"Outlier detection: {outlier_percentage:.2f}% of samples are outliers")
            
            # Consider outliers significant if > 5% of data
            return outlier_percentage > 5
            
        except Exception as e:
            logger.warning(f"Error in outlier detection: {str(e)}")
            return False
    
    def _apply_scaling(self, X: np.ndarray, scaler_type: str, feature_names: List[str]) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Apply the selected scaling method.
        
        Args:
            X: Feature matrix
            scaler_type: Type of scaler to apply
            feature_names: Names of features being scaled
            
        Returns:
            Tuple of (scaled_features, scaling_info)
        """
        scaling_info = {
            "scaler_type": None,
            "feature_names": feature_names,
            "original_shape": X.shape,
            "scaling_applied": False
        }
        
        try:
            if scaler_type == "1":  # StandardScaler
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                scaling_info["scaler_type"] = "StandardScaler"
                print("✅ StandardScaler applied")
                
            elif scaler_type == "2":  # MinMaxScaler
                self.scaler = MinMaxScaler()
                X_scaled = self.scaler.fit_transform(X)
                scaling_info["scaler_type"] = "MinMaxScaler"
                print("✅ MinMaxScaler applied")
                
            elif scaler_type == "3":  # RobustScaler
                self.scaler = RobustScaler()
                X_scaled = self.scaler.fit_transform(X)
                scaling_info["scaler_type"] = "RobustScaler"
                print("✅ RobustScaler applied")
                
            elif scaler_type == "4":  # Skip scaling
                X_scaled = X
                scaling_info["scaler_type"] = "None"
                print("⏭️ Skipped scaling")
                
            else:
                raise ValueError(f"Invalid scaler type: {scaler_type}")
            
            scaling_info["scaling_applied"] = scaler_type != "4"
            scaling_info["scaled_shape"] = X_scaled.shape
            
            # Add statistics for reporting
            if scaling_info["scaling_applied"]:
                scaling_info["original_stats"] = {
                    "mean": np.mean(X, axis=0).tolist(),
                    "std": np.std(X, axis=0).tolist(),
                    "min": np.min(X, axis=0).tolist(),
                    "max": np.max(X, axis=0).tolist()
                }
                scaling_info["scaled_stats"] = {
                    "mean": np.mean(X_scaled, axis=0).tolist(),
                    "std": np.std(X_scaled, axis=0).tolist(),
                    "min": np.min(X_scaled, axis=0).tolist(),
                    "max": np.max(X_scaled, axis=0).tolist()
                }
            
            logger.info(f"Applied {scaling_info['scaler_type']} to {len(feature_names)} features")
            
        except Exception as e:
            logger.error(f"Error in scaling: {str(e)}")
            X_scaled = X
            scaling_info["error"] = str(e)
            scaling_info["scaler_type"] = "Error"
        
        return X_scaled, scaling_info
    
    def get_scaler(self):
        """
        Get the fitted scaler for later use.
        
        Returns:
            Fitted scaler object
        """
        return self.scaler


def scale_numerical_features(df: pd.DataFrame, mode: str = "auto", target_col: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Convenience function to scale numerical features.
    
    Args:
        df: Input DataFrame
        mode: Execution mode
        target_col: Target column name
        
    Returns:
        Tuple of (scaled_features, scaling_info)
    """
    scaler = FeatureScaler(mode)
    return scaler.scale_features(df, target_col)

