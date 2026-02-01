"""
AURA Preprocessor 2.0 - Main Pipeline Module

Dataset-agnostic preprocessing pipeline with comprehensive reporting and LLM explanations.
Supports both interactive and automatic modes for educational and production use.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Import our modular steps
from backend.core.steps.missing_values import MissingValueHandler
from backend.core.steps.encoding import FeatureEncoder
from backend.core.steps.scaling import FeatureScaler
from backend.core.steps.model_training import ModelTrainer
from backend.services.report_service import ReportGenerator
from backend.core.llm.client import LLMHelper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AuraPipeline:
    """
    Main pipeline class for AURA Preprocessor 2.0.
    
    This class orchestrates the entire data preprocessing pipeline,
    from data loading to model training, with comprehensive reporting
    and LLM-powered explanations.
    """
    
    def __init__(self, filepath: str, mode: str = "auto", target_col: Optional[str] = None, 
                 llm_recommendations: Optional[Dict[str, Any]] = None):
        """
        Initialize the AURA pipeline.
        
        Args:
            filepath: Path to the CSV dataset
            mode: Execution mode - "auto" or "step"
            target_col: Target column name (auto-detected if None)
            llm_recommendations: LLM recommendations for preprocessing (optional)
        """
        self.filepath = filepath
        self.mode = mode
        self.target_col = target_col
        self.original_df = None
        self.processed_df = None
        self.preprocessing_steps = []
        self.pipeline_info = {}
        self.llm_recommendations = llm_recommendations  # Store LLM recommendations
        
        # Initialize components
        self.llm_helper = LLMHelper()
        self.report_generator = ReportGenerator()
        
        # Load and analyze the dataset
        self._load_dataset()
        self._detect_target_column()
        
        # Initialize processed_df with original data
        self.processed_df = self.original_df.copy()
        
        logger.info(f"AURA Pipeline initialized for {filepath} in {mode} mode")
        if llm_recommendations:
            logger.info("LLM recommendations will be used for preprocessing decisions")
    
    def _load_dataset(self) -> None:
        """
        Load the dataset from file and perform initial analysis.
        """
        try:
            self.original_df = pd.read_csv(self.filepath)
            logger.info(f"Successfully loaded dataset: {self.original_df.shape}")
            print(f"✅ Loaded dataset with shape {self.original_df.shape}")
            
            # Store initial dataset info
            self.pipeline_info = {
                "filepath": self.filepath,
                "original_shape": self.original_df.shape,
                "columns": self.original_df.columns.tolist(),
                "dtypes": self.original_df.dtypes.astype(str).to_dict(),
                "missing_values": self.original_df.isnull().sum().to_dict()
            }
            
        except Exception as e:
            error_msg = f"Error loading dataset from {self.filepath}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _detect_target_column(self) -> None:
        """
        Auto-detect the target column if not specified.
        """
        if self.original_df is None:
            raise ValueError("Dataset not loaded. Cannot detect target column.")
        
        if self.target_col is None:
            # Common target column names
            target_candidates = [
                'target', 'label', 'y', 'class', 'outcome', 'result',
                'survived', 'price', 'sales', 'revenue', 'profit', 'pclass'
            ]
            
            # Exclude ID-like columns and names
            exclude_patterns = ['id', 'ticket', 'name', 'passenger', 'index', 'cabin']
            
            def is_valid_target(col_name: str) -> bool:
                """Check if column name is a valid target (not an ID column)"""
                col_lower = col_name.lower()
                return not any(pattern in col_lower for pattern in exclude_patterns)
            
            # Look for exact matches first
            for candidate in target_candidates:
                if candidate in self.original_df.columns and is_valid_target(candidate):
                    self.target_col = candidate
                    break
            
            # If no exact match, look for partial matches
            if self.target_col is None:
                for col in self.original_df.columns:
                    col_lower = col.lower()
                    if any(candidate in col_lower for candidate in target_candidates) and is_valid_target(col):
                        self.target_col = col
                        break
            
            # If still no match, use the last column that's not an ID column
            if self.target_col is None:
                valid_columns = [col for col in self.original_df.columns if is_valid_target(col)]
                if valid_columns:
                    self.target_col = valid_columns[-1]
                else:
                    self.target_col = self.original_df.columns[-1]
                logger.warning(f"No target column detected, using: {self.target_col}")
        
        logger.info(f"Target column: {self.target_col}")
        print(f"🎯 Target column: {self.target_col}")
    
    def handle_missing_values(self) -> None:
        """
        Handle missing values using appropriate strategies.
        """
        print("\n" + "="*60)
        print("STEP 1: Handle Missing Values")
        print("="*60)
        
        try:
            # Pass LLM recommendations to the handler
            llm_missing_rec = None
            if self.llm_recommendations and "missing" in self.llm_recommendations:
                llm_missing_rec = self.llm_recommendations["missing"]
                logger.info("Using LLM recommendations for missing values")
            
            handler = MissingValueHandler(self.mode, llm_recommendations=llm_missing_rec)
            self.processed_df, missing_info = handler.process(self.processed_df)
            
            # Log the step
            self.preprocessing_steps.append({
                "step_name": "missing_values_handling",
                "details": missing_info
            })
            
            if self.mode == "step":
                self.llm_helper.explain_step(
                    "Missing values handled",
                    data_sample=self.processed_df.head(),
                    additional_info={"missing_info": missing_info}
                )
        
        except Exception as e:
            error_msg = f"Error handling missing values: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def encode_features(self) -> None:
        """
        Encode categorical features in the dataset.
        """
        if self.processed_df is None:
            raise ValueError("No processed dataset available. Run handle_missing_values() first.")
        
        print("\n=== STEP 2: Encode Categorical Features ===")
        
        try:
            # Pass LLM recommendations to the encoder
            llm_encoding_rec = None
            if self.llm_recommendations and "encoding" in self.llm_recommendations:
                llm_encoding_rec = self.llm_recommendations["encoding"]
                logger.info("Using LLM recommendations for encoding")
            
            encoder = FeatureEncoder(self.mode, llm_recommendations=llm_encoding_rec)
            # Pass target column to preserve it during encoding
            self.processed_df, encoding_info = encoder.encode_features(self.processed_df, self.target_col)
            
            # Store step information
            step_info = {
                "step_name": "feature_encoding",
                "timestamp": datetime.now().isoformat(),
                "details": encoding_info,
                "data_shape_before": self.processed_df.shape,
                "data_shape_after": self.processed_df.shape
            }
            self.preprocessing_steps.append(step_info)
            
            # Generate LLM explanations for each encoded column
            if self.mode == "step":
                for col, info in encoding_info.items():
                    if info["encoding_method"] == "label_encoding":
                        self.llm_helper.explain_step(
                            f"Label encoded column '{col}'",
                            self.processed_df[[col]].head() if col in self.processed_df.columns else None
                        )
                    elif info["encoding_method"] == "onehot_encoding":
                        new_cols = info.get("new_columns", [])
                        if new_cols:
                            sample_data = self.processed_df[new_cols].head()
                            self.llm_helper.explain_step(
                                f"One-hot encoded column '{col}'",
                                sample_data
                            )
            
            logger.info("Feature encoding completed")
            
        except Exception as e:
            error_msg = f"Error in feature encoding: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def scale_features(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Scale numerical features in the dataset.
        
        Returns:
            Tuple of (scaled_features, scaling_info)
        """
        if self.processed_df is None:
            raise ValueError("No processed dataset available. Run encode_features() first.")
        
        if self.target_col is None:
            raise ValueError("Target column not set. Cannot scale features.")
        
        print("\n=== STEP 3: Scale Numerical Features ===")
        
        try:
            # Pass LLM recommendations to the scaler
            llm_scaling_rec = None
            if self.llm_recommendations and "scaling" in self.llm_recommendations:
                llm_scaling_rec = self.llm_recommendations["scaling"]
                logger.info("Using LLM recommendations for scaling")
            
            scaler = FeatureScaler(self.mode, llm_recommendations=llm_scaling_rec)
            X_scaled, scaling_info = scaler.scale_features(self.processed_df, self.target_col)
            
            # Store step information
            step_info = {
                "step_name": "feature_scaling",
                "timestamp": datetime.now().isoformat(),
                "details": scaling_info,
                "data_shape_before": self.processed_df.shape,
                "data_shape_after": X_scaled.shape
            }
            self.preprocessing_steps.append(step_info)
            
            # Generate LLM explanation
            if self.mode == "step":
                scaler_type = scaling_info.get("scaler_type", "None")
                if scaler_type != "None":
                    self.llm_helper.explain_step(
                        f"{scaler_type} applied to numerical features",
                        pd.DataFrame(X_scaled).head()
                    )
                else:
                    self.llm_helper.explain_step(
                        "No scaling applied",
                        pd.DataFrame(X_scaled).head()
                    )
            
            logger.info("Feature scaling completed")
            return X_scaled, scaling_info
            
        except Exception as e:
            error_msg = f"Error in feature scaling: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def train_model(self, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train a machine learning model on the processed data.
        
        Args:
            test_size: Proportion of data for testing
            
        Returns:
            Model training results
        """
        if self.processed_df is None:
            raise ValueError("No processed dataset available. Run encode_features() first.")
        
        if self.target_col is None:
            raise ValueError("Target column not set. Cannot train model.")
        
        print("\n=== STEP 4: Train Machine Learning Model ===")
        
        try:
            # Prepare features and target
            if self.target_col not in self.processed_df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in processed dataset!")
            
            X = self.processed_df.drop(self.target_col, axis=1)
            y = self.processed_df[self.target_col]
            
            # Scale features
            X_scaled, scaling_info = self.scale_features()
            
            # Train the model
            trainer = ModelTrainer(self.mode)
            model_results = trainer.train_model(X_scaled, y, self.target_col, test_size)
            
            # Store step information
            step_info = {
                "step_name": "model_training",
                "timestamp": datetime.now().isoformat(),
                "details": model_results,
                "test_size": test_size
            }
            self.preprocessing_steps.append(step_info)
            
            # Generate LLM explanation
            if self.mode == "step" and "results" in model_results:
                accuracy = model_results["results"].get("accuracy", 0)
                self.llm_helper.explain_step(
                    f"Trained {model_results.get('model_name', 'ML model')} with accuracy {accuracy:.3f}",
                    pd.DataFrame(X_scaled).head(),
                    {"accuracy": accuracy, "model_results": model_results}
                )
            
            logger.info("Model training completed")
            return model_results
            
        except Exception as e:
            error_msg = f"Error in model training: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def generate_report(self, model_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the pipeline execution.
        
        Args:
            model_results: Model training results (optional)
            
        Returns:
            Complete report dictionary
        """
        if self.original_df is None or self.processed_df is None:
            raise ValueError("Datasets not loaded. Cannot generate report.")
        
        if self.target_col is None:
            raise ValueError("Target column not set. Cannot generate report.")
        
        print("\n=== STEP 5: Generate Comprehensive Report ===")
        
        try:
            # Get encoding and scaling info from steps
            encoding_info = None
            scaling_info = None
            
            for step in self.preprocessing_steps:
                if step["step_name"] == "feature_encoding":
                    encoding_info = step["details"]
                elif step["step_name"] == "feature_scaling":
                    scaling_info = step["details"]
            
            # Generate the report
            report = self.report_generator.generate_pipeline_report(
                original_df=self.original_df,
                processed_df=self.processed_df,
                target_col=self.target_col,
                preprocessing_steps=self.preprocessing_steps,
                model_results=model_results,
                encoding_info=encoding_info,
                scaling_info=scaling_info
            )
            
            # Save the report
            self.report_generator.save_report()
            
            # Print summary
            self.report_generator.print_summary()
            
            logger.info("Report generation completed")
            return report
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def save_processed_data(self, output_path: Optional[str] = None) -> str:
        """
        Save the processed dataset to a CSV file.
        
        Args:
            output_path: Path to save the processed data
            
        Returns:
            Path where the data was saved
        """
        if self.processed_df is None:
            raise ValueError("No processed dataset available. Cannot save data.")
        
        if output_path is None:
            # Generate default filename based on input file
            base_name = os.path.splitext(os.path.basename(self.filepath))[0]
            output_path = f"outputs/{base_name}_processed.csv"
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the processed data
            self.processed_df.to_csv(output_path, index=False)
            
            print(f"💾 Saved processed dataset to {output_path}")
            logger.info(f"Processed data saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            error_msg = f"Error saving processed data: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def save_explanations(self, filename: str = "aura_explanations.json") -> None:
        """
        Save LLM explanations to a JSON file.
        
        Args:
            filename: Name of the explanations file
        """
        self.llm_helper.save_explanations(filename)
    
    def run_full_pipeline(self, 
                         test_size: float = 0.2,
                         save_data: bool = True,
                         save_explanations: bool = True) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            test_size: Proportion of data for testing
            save_data: Whether to save processed data
            save_explanations: Whether to save LLM explanations
            
        Returns:
            Complete pipeline results
        """
        print("🚀 Starting AURA Preprocessor 2.0 Pipeline...")
        print(f"📁 Dataset: {self.filepath}")
        print(f"🎯 Target: {self.target_col}")
        print(f"⚙️ Mode: {self.mode}")
        
        try:
            # Step 1: Handle missing values
            self.handle_missing_values()
            
            # Step 2: Encode categorical features
            self.encode_features()
            
            # Step 3: Train model (includes scaling)
            model_results = self.train_model(test_size)
            
            # Step 4: Generate report
            report = self.generate_report(model_results)
            
            # Step 5: Save outputs
            if save_data:
                data_path = self.save_processed_data()
                report["processed_data_path"] = data_path
            
            if save_explanations:
                self.save_explanations()
            
            print("\n✅ Pipeline completed successfully!")
            logger.info("Full pipeline execution completed successfully")
            
            return {
                "pipeline_info": self.pipeline_info,
                "preprocessing_steps": self.preprocessing_steps,
                "model_results": model_results,
                "report": report,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(error_msg)
            print(f"\n❌ Pipeline failed: {error_msg}")
            
            return {
                "pipeline_info": self.pipeline_info,
                "preprocessing_steps": self.preprocessing_steps,
                "error": error_msg,
                "success": False
            }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded dataset.
        
        Returns:
            Dataset information dictionary
        """
        if self.original_df is None:
            return {"error": "No dataset loaded"}
        
        info = {
            "filepath": self.filepath,
            "shape": self.original_df.shape,
            "columns": self.original_df.columns.tolist(),
            "dtypes": self.original_df.dtypes.astype(str).to_dict(),
            "missing_values": self.original_df.isnull().sum().to_dict(),
            "target_column": self.target_col,
            "numeric_columns": self.original_df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": self.original_df.select_dtypes(include=['object']).columns.tolist()
        }
        
        if self.target_col in self.original_df.columns:
            info["target_info"] = {
                "dtype": str(self.original_df[self.target_col].dtype),
                "unique_values": int(self.original_df[self.target_col].nunique()),
                "value_counts": self.original_df[self.target_col].value_counts().to_dict()
            }
        
        return info