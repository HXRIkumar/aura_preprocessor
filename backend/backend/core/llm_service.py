"""
Groq LLM Service for AURA Preprocessor
Provides intelligent recommendations for preprocessing strategies.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class GroqLLMService:
    """Service for interacting with Groq API for preprocessing recommendations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq LLM service.
        
        Args:
            api_key: Groq API key. If not provided, uses GROQ_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Groq's latest powerful model
        
    def analyze_dataset_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze dataset metadata and provide preprocessing recommendations.
        
        Args:
            metadata: Dictionary containing dataset information including columns,
                     types, missing values, etc.
        
        Returns:
            Dictionary with recommendations for missing values, encoding, scaling, and model.
        """
        prompt = self._build_metadata_analysis_prompt(metadata)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert ML engineer specializing in data preprocessing. 
                        Analyze the dataset metadata and provide specific, actionable recommendations.
                        Always respond in valid JSON format with the exact structure requested."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            
            # Log recommendations to console
            logger.info("=" * 70)
            logger.info("🤖 LLM RECOMMENDATIONS RECEIVED")
            logger.info("=" * 70)
            
            if "recommendations" in recommendations:
                recs = recommendations["recommendations"]
                
                # Log Missing Values Strategy
                if "missing" in recs:
                    logger.info(f"📊 Missing Values:")
                    logger.info(f"   Strategy: {recs['missing'].get('strategy', 'N/A')}")
                    if "columns" in recs['missing']:
                        logger.info(f"   Column-specific strategies:")
                        for col, strategy in recs['missing']['columns'].items():
                            logger.info(f"      - {col}: {strategy}")
                
                # Log Encoding Strategy
                if "encoding" in recs:
                    logger.info(f"🔤 Encoding:")
                    logger.info(f"   Strategy: {recs['encoding'].get('strategy', 'N/A')}")
                    if "columns" in recs['encoding']:
                        logger.info(f"   Column-specific strategies:")
                        for col, strategy in recs['encoding']['columns'].items():
                            logger.info(f"      - {col}: {strategy}")
                
                # Log Scaling Strategy
                if "scaling" in recs:
                    logger.info(f"⚖️  Scaling:")
                    logger.info(f"   Strategy: {recs['scaling'].get('strategy', 'N/A')}")
                
                # Log Model Recommendation
                if "model" in recs:
                    logger.info(f"🤖 Model:")
                    logger.info(f"   Algorithm: {recs['model'].get('algorithm', 'N/A')}")
                
                logger.info("=" * 70)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error calling Groq API: {e}")
            raise  # Re-raise the exception instead of returning fallback
    
    def chat(self, message: str, dataset_context: Optional[Dict[str, Any]] = None,
             conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Handle chat interactions with context about the dataset.
        
        Args:
            message: User's message/question
            dataset_context: Context about the current dataset
            conversation_history: Previous conversation messages
        
        Returns:
            LLM response string
        """
        messages = [
            {
                "role": "system",
                "content": """You are AURA, an AI assistant specialized in machine learning preprocessing.
                You help users understand their data and make informed decisions about preprocessing strategies.
                Be concise, helpful, and explain technical concepts clearly."""
            }
        ]
        
        # Add dataset context if available
        if dataset_context:
            context_str = f"""Current Dataset Context:
- Dataset: {dataset_context.get('dataset_name', 'Unknown')}
- Columns: {len(dataset_context.get('columns', []))}
- Target: {dataset_context.get('target_column', 'Not set')}

Column Details:
{self._format_columns_for_context(dataset_context.get('columns', []))}
"""
            messages.append({
                "role": "system",
                "content": context_str
            })
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages for context
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try again."
    
    def _build_metadata_analysis_prompt(self, metadata: Dict[str, Any]) -> str:
        """Build a detailed prompt for metadata analysis."""
        columns = metadata.get('columns', [])
        dataset_name = metadata.get('dataset_name', 'Unknown')
        target_column = metadata.get('target_column')
        
        columns_info = []
        for col in columns:
            col_str = f"- {col['name']}: {col['type']}"
            if col.get('missing_pct', 0) > 0:
                col_str += f" (missing: {col['missing_pct']:.1f}%)"
            if col['type'] == 'categorical':
                col_str += f" (unique values: {col.get('nunique', 'N/A')})"
            columns_info.append(col_str)
        
        prompt = f"""You are an expert data scientist analyzing a dataset for machine learning preprocessing.

CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no explanations outside the JSON structure.

Dataset Information:
- Name: {dataset_name}
- Target Column: {target_column or 'Not specified'}
- Total Columns: {len(columns)}

Column Details:
{chr(10).join(columns_info)}

REQUIRED OUTPUT FORMAT:
Return EXACTLY this JSON structure with your recommendations. Do not add any text before or after the JSON.

{{
  "recommendations": {{
    "missing": {{
      "strategy": "median",
      "columns": {{
        "column_name": "mean|median|mode|drop"
      }},
      "explain": "Concise explanation (1-2 sentences)",
      "risk": ["risk1", "risk2"]
    }},
    "encoding": {{
      "strategy": "onehot",
      "columns": {{
        "column_name": "onehot|label"
      }},
      "explain": "Concise explanation (1-2 sentences)",
      "risk": ["risk1"]
    }},
    "scaling": {{
      "strategy": "standard",
      "explain": "Concise explanation (1-2 sentences)",
      "risk": ["risk1"]
    }},
    "model": {{
      "algorithm": "random_forest",
      "explain": "Concise explanation (1-2 sentences)",
      "risk": ["risk1"]
    }}
  }}
}}

STRATEGY OPTIONS:
- missing.strategy: "mean", "median", "mode", or "drop"
- missing.columns: Specify strategy for EACH column with missing values
- encoding.strategy: "label" or "onehot" (general approach)
- encoding.columns: Specify "label" or "onehot" for EACH categorical column
- scaling.strategy: "standard", "minmax", "robust", or "none"
- model.algorithm: "random_forest", "gradient_boosting", "logistic_regression", or "svm"

ANALYSIS GUIDELINES:
1. Missing Values:
   - High missing (>50%): recommend "drop"
   - Numeric columns: use "mean" or "median"
   - Categorical columns: use "mode"
   - Specify strategy for EACH column with missing data

2. Encoding:
   - Low cardinality (<10 unique): prefer "label"
   - High cardinality: prefer "onehot" OR "drop" if too high
   - Consider column meaning (e.g., ordinal vs nominal)
   - Specify strategy for EACH categorical column

3. Scaling:
   - Classification problems: usually "standard" or "minmax"
   - Outliers present: prefer "robust"
   - Tree-based models: "none" is acceptable
   
4. Model:
   - Small datasets (<1000 rows): "random_forest" or "logistic_regression"
   - Large datasets: "gradient_boosting" or "svm"
   - Binary classification: any algorithm works
   - High cardinality features: prefer tree-based models

REMEMBER: Return ONLY the JSON object, nothing else. The output will be parsed programmatically."""

        return prompt
    
    def _format_columns_for_context(self, columns: List[Dict[str, Any]]) -> str:
        """Format column information for chat context."""
        if not columns:
            return "No column information available"
        
        formatted = []
        for col in columns[:10]:  # Limit to first 10 columns
            col_str = f"- {col['name']} ({col['type']})"
            if col.get('missing_pct', 0) > 0:
                col_str += f" - {col['missing_pct']:.1f}% missing"
            formatted.append(col_str)
        
        if len(columns) > 10:
            formatted.append(f"... and {len(columns) - 10} more columns")
        
        return "\n".join(formatted)
    
    def _get_fallback_recommendations(self) -> Dict[str, Any]:
        """Return fallback recommendations if LLM call fails."""
        return {
            "recommendations": {
                "missing": {
                    "strategy": "mean",
                    "columns": {},
                    "explain": "Using mean imputation as a safe default for numeric columns.",
                    "risk": ["May not be optimal without analyzing data distribution"]
                },
                "encoding": {
                    "strategy": "onehot",
                    "columns": {},
                    "explain": "One-hot encoding as a general-purpose approach.",
                    "risk": ["May increase dimensionality significantly"]
                },
                "scaling": {
                    "strategy": "standard",
                    "explain": "StandardScaler as a common preprocessing choice.",
                    "risk": ["May not be optimal for non-normal distributions"]
                },
                "model": {
                    "algorithm": "random_forest",
                    "explain": "Random Forest as a robust general-purpose model.",
                    "risk": ["May require hyperparameter tuning"]
                }
            }
        }


# Global instance
_llm_service = None

def get_llm_service() -> GroqLLMService:
    """Get or create global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = GroqLLMService()
    return _llm_service
