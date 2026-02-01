"""
Model Training Module for AURA Preprocessor 2.0

Handles machine learning model training with multiple algorithms.
Supports both interactive and automatic modes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import joblib
import os

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles machine learning model training with multiple algorithms.
    """
    
    def __init__(self, mode: str = "auto"):
        """
        Initialize the model trainer.
        
        Args:
            mode: Execution mode - "auto" or "step"
        """
        self.mode = mode
        self.training_info = {}
        self.model = None
        self.model_name = None
    
    def train_model(self, 
                   X: Union[pd.DataFrame, np.ndarray], 
                   y: Union[pd.Series, np.ndarray],
                   target_col: str,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Args:
            X: Feature matrix
            y: Target vector
            target_col: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            # Convert to numpy arrays if needed
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Data split: Train {X_train.shape[0]} samples, Test {X_test.shape[0]} samples")
            
            # Select model
            if self.mode == "step":
                model_name = self._get_user_model_choice()
            else:
                model_name = self._get_auto_model_choice(X_train, y_train)
            
            # Train the model
            self.model, actual_model_name = self._train_selected_model(model_name, X_train, y_train)
            self.model_name = actual_model_name
            
            # Evaluate the model
            results = self._evaluate_model(X_test, y_test, target_col)
            
            # Store training information
            self.training_info = {
                "model_name": actual_model_name,
                "target_column": target_col,
                "train_size": X_train.shape[0],
                "test_size": X_test.shape[0],
                "feature_count": X_train.shape[1],
                "results": results
            }
            
            return self.training_info
            
        except Exception as e:
            error_msg = f"Error in model training: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _get_user_model_choice(self) -> str:
        """
        Get user choice for model type in step mode.
        
        Returns:
            User's choice as string
        """
        print("\n⚡ Model selection options:")
        print("   1) Random Forest (recommended for most cases)")
        print("   2) Gradient Boosting (good for complex patterns)")
        print("   3) Logistic Regression (fast, interpretable)")
        print("   4) Support Vector Machine (good for small datasets)")
        
        while True:
            choice = input("👉 Enter choice: ").strip()
            if choice in ["1", "2", "3", "4"]:
                return choice
            print("⚠️ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    def _get_auto_model_choice(self, X_train: np.ndarray, y_train: np.ndarray) -> str:
        """
        Automatically choose model based on data characteristics.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Auto-selected model type
        """
        n_samples, n_features = X_train.shape
        
        # Heuristic-based model selection
        if n_samples < 1000:
            logger.info(f"Small dataset ({n_samples} samples), using Logistic Regression")
            return "3"  # Logistic Regression
        elif n_features > n_samples * 0.1:  # High dimensional
            logger.info(f"High dimensional data ({n_features} features), using Random Forest")
            return "1"  # Random Forest
        else:
            logger.info(f"Standard dataset, using Random Forest")
            return "1"  # Random Forest
    
    def _train_selected_model(self, model_choice: str, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the selected model.
        
        Args:
            model_choice: Selected model type
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Tuple of (trained_model, model_name_string)
        """
        if model_choice == "1":  # Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            model_name = "Random Forest"
            print("🌲 Training Random Forest...")
            
        elif model_choice == "2":  # Gradient Boosting
            model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            )
            model_name = "Gradient Boosting"
            print("📈 Training Gradient Boosting...")
            
        elif model_choice == "3":  # Logistic Regression
            model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            model_name = "Logistic Regression"
            print("📊 Training Logistic Regression...")
            
        elif model_choice == "4":  # SVM
            model = SVC(
                random_state=42,
                probability=True
            )
            model_name = "Support Vector Machine"
            print("🎯 Training Support Vector Machine...")
            
        else:
            raise ValueError(f"Invalid model choice: {model_choice}")
        
        # Train the model
        model.fit(X_train, y_train)
        logger.info(f"Successfully trained {model_name}")
        
        return model, model_name
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, target_col: str) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            target_col: Target column name
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results = {
                "accuracy": float(accuracy),
                "cv_mean": float(np.mean(cv_scores)),
                "cv_std": float(np.std(cv_scores)),
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "predictions": y_pred.tolist(),
                "probabilities": y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            print(f"📊 Model Performance:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            
            logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error in model evaluation: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, filepath)
            
            # Save training info
            info_filepath = filepath.replace('.pkl', '_info.json')
            import json
            with open(info_filepath, 'w') as f:
                json.dump(self.training_info, f, indent=2)
            
            logger.info(f"Model saved to {filepath}")
            print(f"💾 Model saved to {filepath}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            logger.error(error_msg)
            print(f"⚠️ {error_msg}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = joblib.load(filepath)
            
            # Load training info if available
            info_filepath = filepath.replace('.pkl', '_info.json')
            if os.path.exists(info_filepath):
                import json
                with open(info_filepath, 'r') as f:
                    self.training_info = json.load(f)
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            return False


def train_ml_model(X: Union[pd.DataFrame, np.ndarray], 
                   y: Union[pd.Series, np.ndarray],
                   target_col: str,
                   mode: str = "auto",
                   test_size: float = 0.2) -> Dict[str, Any]:
    """
    Convenience function to train a machine learning model.
    
    Args:
        X: Feature matrix
        y: Target vector
        target_col: Target column name
        mode: Execution mode
        test_size: Test set proportion
        
    Returns:
        Training results dictionary
    """
    trainer = ModelTrainer(mode)
    return trainer.train_model(X, y, target_col, test_size)

