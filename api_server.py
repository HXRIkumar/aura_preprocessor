"""
FastAPI REST API Wrapper for AURA Preprocessor Backend V1
This provides REST endpoints for the existing pipeline code.
"""

import os
import uuid
import json
import shutil
import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd

from src.pipeline import AuraPipeline
from src.llm_service import get_llm_service
from src.agent.core import AuraAgent
from src.agent.tools import register_dataset, DATA_STORE, get_dataset

# Configure logging
logger = logging.getLogger(__name__)


# Custom JSON encoder to handle NaN, inf, etc.
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    This handles numpy.int64, numpy.float64, etc. that can't be serialized by FastAPI.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# Initialize FastAPI app
app = FastAPI(
    title="AURA Preprocessor API",
    description="REST API for ML preprocessing pipeline",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# In-memory storage (replace with database in production)
datasets: Dict[str, Dict[str, Any]] = {}
pipelines: Dict[str, Dict[str, Any]] = {}

# Directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# Pydantic Models
class PipelineConfig(BaseModel):
    dataset_id: str
    mode: str = "auto"
    target_column: Optional[str] = None
    test_size: Optional[float] = None
    save_options: Optional[Dict[str, bool]] = None
    manual_config: Optional[Dict[str, Any]] = None  # For manual mode - contains nested dicts
    
    model_config = {"extra": "ignore"}  # Updated for Pydantic V2


class ChatRequest(BaseModel):
    dataset_id: str
    message: str
    conversation_history: List[Dict[str, str]] = []


class LLMAnalysisRequest(BaseModel):
    dataset_id: str
    metadata: Dict[str, Any]


class AgentRunRequest(BaseModel):
    dataset_id: str


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AURA Preprocessor API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/v1/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{dataset_id}.csv"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read and analyze dataset
        df = pd.read_csv(file_path)
        
        # Register with Agent Data Store
        register_dataset(dataset_id, df)
        
        # Store dataset info
        datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "filepath": str(file_path),
            "size": int(os.path.getsize(file_path)),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "upload_timestamp": datetime.now().isoformat(),
        }
        
        return datasets[dataset_id]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/v1/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get dataset information"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        filepath = datasets[dataset_id]["filepath"]
        df = pd.read_csv(filepath)
        
        # Detect target column (last column if numeric)
        target_column = None
        if df.iloc[:, -1].dtype in ['int64', 'float64', 'int32', 'float32']:
            target_column = df.columns[-1]
        
        info = {
            "dataset_id": dataset_id,
            "filename": datasets[dataset_id]["filename"],
            "shape": [int(len(df)), int(len(df.columns))],
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            "target_column": target_column,
        }
        
        return info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")


@app.get("/api/v1/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 10):
    """Preview dataset rows"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        filepath = datasets[dataset_id]["filepath"]
        df = pd.read_csv(filepath)
        
        preview_df = df.head(limit)
        
        # Convert to JSON-safe format manually
        data_rows = []
        for _, row in preview_df.iterrows():
            json_row = []
            for val in row:
                if pd.isna(val):
                    json_row.append(None)
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    json_row.append(int(val))
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    if np.isnan(val) or np.isinf(val):
                        json_row.append(None)
                    else:
                        json_row.append(float(val))
                else:
                    json_row.append(val)
            data_rows.append(json_row)
        
        return JSONResponse(content={
            "dataset_id": dataset_id,
            "columns": df.columns.tolist(),
            "data": data_rows,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview dataset: {str(e)}")


def get_dataset_metadata(dataset_id: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for LLM analysis.
    
    Args:
        dataset_id: Dataset identifier
        target_col: Target column name
        
    Returns:
        Dictionary with dataset metadata
    """
    filepath = datasets[dataset_id]["filepath"]
    df = pd.read_csv(filepath)
    
    # Analyze columns
    columns_info = []
    for col in df.columns:
        col_info = {
            "name": col,
            "type": "categorical" if df[col].dtype == 'object' else "numeric",
            "missing_pct": float((df[col].isnull().sum() / len(df)) * 100),
            "nunique": int(df[col].nunique())
        }
        columns_info.append(col_info)
    
    metadata = {
        "dataset_name": datasets[dataset_id]["filename"],
        "target_column": target_col,
        "columns": columns_info,
        "sample_rows": df.head(3).values.tolist()  # First 3 rows for context
    }
    
    return metadata


def run_pipeline_task(pipeline_id: str, config: PipelineConfig):
    """Background task to run the pipeline"""
    try:
        # Update status
        pipelines[pipeline_id]["status"] = "running"
        pipelines[pipeline_id]["progress"] = 10
        pipelines[pipeline_id]["current_step"] = "initializing"
        
        # Get dataset filepath
        filepath = datasets[config.dataset_id]["filepath"]
        
        # Get configuration from manual_config if in manual mode
        manual_config = config.manual_config if config.mode == "step" and config.manual_config else {}
        
        # Extract target column and test size
        target_col = manual_config.get("target_column") if manual_config else config.target_column
        test_size = float(manual_config.get("test_size", 0.2)) if manual_config else (config.test_size if config.test_size else 0.2)
        
        # Get save options with defaults
        save_options = config.save_options if config.save_options else {
            "processed_data": True,
            "report": True,
            "explanations": True
        }
        
        # Get LLM recommendations if in AUTO mode
        llm_recommendations = None
        recommendations_file = None
        
        if config.mode == "auto":
            try:
                pipelines[pipeline_id]["current_step"] = "getting_llm_recommendations"
                pipelines[pipeline_id]["progress"] = 12
                
                logger.info("🤖 AUTO MODE: Requesting LLM recommendations from Groq API...")
                
                # Generate metadata
                metadata = get_dataset_metadata(config.dataset_id, target_col)
                
                # Get LLM recommendations
                llm_service = get_llm_service()
                recommendations_response = llm_service.analyze_dataset_metadata(metadata)
                
                # Extract recommendations from response
                if "recommendations" in recommendations_response:
                    llm_recommendations = recommendations_response["recommendations"]
                    
                    # Save recommendations to file
                    recommendations_file = OUTPUT_DIR / f"llm_recommendations_{pipeline_id}.json"
                    with open(recommendations_file, 'w') as f:
                        json.dump(recommendations_response, f, indent=2)
                    
                    logger.info(f"✅ LLM recommendations received and saved to: {recommendations_file}")
                    
                else:
                    error_msg = "LLM response missing 'recommendations' key"
                    logger.error(f"❌ {error_msg}")
                    raise ValueError(error_msg)
                    
            except Exception as e:
                error_msg = f"Failed to get LLM recommendations in AUTO mode: {str(e)}"
                logger.error(f"❌ {error_msg}")
                
                # Update pipeline status to failed
                pipelines[pipeline_id]["status"] = "failed"
                pipelines[pipeline_id]["error"] = f"LLM Service Error: Unable to get preprocessing recommendations. {str(e)}"
                pipelines[pipeline_id]["current_step"] = "llm_error"
                
                # Return early - don't continue with pipeline
                return
        
        # Initialize pipeline
        pipelines[pipeline_id]["progress"] = 15
        pipelines[pipeline_id]["current_step"] = "loading_data"
        
        # IMPORTANT: Always use "auto" mode to avoid interactive prompts
        # Pass LLM recommendations to the pipeline
        pipeline = AuraPipeline(
            filepath=filepath,
            mode="auto",  # Force auto mode to prevent interactive prompts
            target_col=target_col,
            llm_recommendations=llm_recommendations  # Pass LLM recommendations
        )
        
        pipelines[pipeline_id]["progress"] = 25
        pipelines[pipeline_id]["current_step"] = "missing_values"
        
        # Progress will be updated as pipeline processes
        time.sleep(0.3)  # Give frontend time to update
        
        pipelines[pipeline_id]["progress"] = 45
        pipelines[pipeline_id]["current_step"] = "encoding"
        
        time.sleep(0.3)
        
        pipelines[pipeline_id]["progress"] = 60
        pipelines[pipeline_id]["current_step"] = "scaling"
        
        time.sleep(0.3)
        
        # TODO: Pass manual strategies to pipeline when implementing per-column processing
        # For now, the pipeline will use default auto mode strategies
        
        # Run pipeline (this takes most of the time)
        results = pipeline.run_full_pipeline(
            test_size=test_size,
            save_data=save_options.get("processed_data", True),
            save_explanations=save_options.get("explanations", True)
        )
        
        pipelines[pipeline_id]["progress"] = 85
        pipelines[pipeline_id]["current_step"] = "model_training"
        
        # Small delay to show progress
        time.sleep(0.5)
        
        pipelines[pipeline_id]["progress"] = 95
        pipelines[pipeline_id]["current_step"] = "report"
        
        if results["success"]:
            # Store results
            pipelines[pipeline_id]["status"] = "completed"
            pipelines[pipeline_id]["progress"] = 100
            pipelines[pipeline_id]["current_step"] = "completed"
            
            # Build model metrics if available
            model_metrics = {}
            model_name = None
            if "model_results" in results and results["model_results"] and "results" in results["model_results"]:
                model_results = results["model_results"]["results"]
                model_name = results["model_results"].get("model_name", "Unknown")
                model_metrics = {
                    "model_name": model_name,
                    "accuracy": model_results.get("accuracy"),
                    "cv_score": model_results.get("cv_mean"),  # Fixed: use cv_mean instead of cv_score
                    "cv_std": model_results.get("cv_std"),
                }
            
            # Extract preprocessing summary from steps
            preprocessing_summary = {}
            for step in results.get("preprocessing_steps", []):
                step_name = step.get("step_name", "")
                if step_name == "missing_values_handling":
                    # missing_info is a dict with column names as keys
                    missing_info = step.get("details", {})
                    handled_columns = [
                        f"{col} ({info.get('handling_method', 'unknown')})" 
                        for col, info in missing_info.items()
                        if info.get("handling_method") not in ["skipped", "error", None]
                    ]
                    preprocessing_summary["missing_values_handled"] = handled_columns
                elif step_name == "feature_encoding":
                    # encoding_info is a dict with column names as keys
                    encoding_info = step.get("details", {})
                    encoded_columns = [
                        f"{col} ({info.get('encoding_method', 'unknown')})" 
                        for col, info in encoding_info.items()
                        if info.get("encoding_method") not in ["skipped", "error", None]
                    ]
                    preprocessing_summary["encoding_applied"] = encoded_columns
                elif step_name == "feature_scaling":
                    # scaling_info contains scaler_type and feature_names
                    scaling_details = step.get("details", {})
                    scaler_type = scaling_details.get("scaler_type", "None")
                    feature_names = scaling_details.get("feature_names", [])
                    if scaler_type != "None" and feature_names:
                        # Show first few features if there are many
                        if len(feature_names) > 5:
                            features_display = f"{', '.join(feature_names[:5])} and {len(feature_names) - 5} more"
                        else:
                            features_display = ', '.join(feature_names)
                        scaling_summary = f"{scaler_type} ({features_display})"
                    else:
                        scaling_summary = scaler_type
                    preprocessing_summary["scaling_applied"] = scaling_summary
            
            # Get dataset info from pipeline_info or report
            original_shape = results.get("pipeline_info", {}).get("original_shape")
            if not original_shape and "report" in results:
                original_shape = results["report"].get("original_shape", (0, 0))
            
            # Add model name to preprocessing summary
            if model_name:
                preprocessing_summary["model_trained"] = model_name
            
            pipelines[pipeline_id]["results"] = {
                "model_metrics": model_metrics,
                "preprocessing_summary": {
                    "original_shape": original_shape,
                    **preprocessing_summary
                },
                "report": results.get("report", {}),
            }
            
            # Store file paths - use dataset-specific file names
            dataset_id = pipelines[pipeline_id]["dataset_id"]
            output_files = {
                "processed_data": f"outputs/{dataset_id}_processed.csv",
                "report": "outputs/report.json",
                "explanations": "outputs/aura_explanations.json",
            }
            
            # Add LLM recommendations file if it was generated
            if recommendations_file:
                output_files["llm_recommendations"] = str(recommendations_file)
            
            pipelines[pipeline_id]["output_files"] = output_files
        else:
            pipelines[pipeline_id]["status"] = "failed"
            pipelines[pipeline_id]["error"] = results.get("error", "Unknown error")
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Pipeline {pipeline_id} failed with error:")
        print(error_details)
        pipelines[pipeline_id]["status"] = "failed"
        pipelines[pipeline_id]["error"] = str(e)
        pipelines[pipeline_id]["error_details"] = error_details


@app.post("/api/v1/pipeline/start")
async def start_pipeline(config: PipelineConfig, background_tasks: BackgroundTasks):
    """Start pipeline execution"""
    print(f"Received pipeline config: {config.dict()}")
    
    if config.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Generate pipeline ID
    pipeline_id = str(uuid.uuid4())
    
    # Initialize pipeline status
    pipelines[pipeline_id] = {
        "pipeline_id": pipeline_id,
        "dataset_id": config.dataset_id,
        "status": "pending",
        "progress": 0,
        "current_step": "initializing",
        "steps_completed": [],
        "started_at": datetime.now().isoformat(),
    }
    
    # Run pipeline in background
    background_tasks.add_task(run_pipeline_task, pipeline_id, config)
    
    return pipelines[pipeline_id]


@app.get("/api/v1/pipeline/status/{pipeline_id}")
async def get_pipeline_status(pipeline_id: str):
    """Get pipeline execution status"""
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Convert NumPy types to native Python types for JSON serialization
    status = convert_numpy_types(pipelines[pipeline_id])
    return status


@app.get("/api/v1/results/{pipeline_id}")
async def get_results(pipeline_id: str):
    """Get pipeline results"""
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    if pipelines[pipeline_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Pipeline not completed yet")
    
    # Convert NumPy types to native Python types for JSON serialization
    results = convert_numpy_types(pipelines[pipeline_id]["results"])
    return results


@app.get("/api/v1/download/processed/{pipeline_id}")
async def download_processed_data(pipeline_id: str):
    """Download processed dataset"""
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    try:
        # Use the most recent processed file
        files = list(OUTPUT_DIR.glob("*_processed.csv"))
        if not files:
            raise HTTPException(status_code=404, detail="Processed data not found")
        
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return FileResponse(
            path=latest_file,
            filename="processed_data.csv",
            media_type="text/csv"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/api/v1/download/report/{pipeline_id}")
async def download_report(pipeline_id: str):
    """Download report JSON"""
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    report_path = OUTPUT_DIR / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=report_path,
        filename="report.json",
        media_type="application/json"
    )


@app.get("/api/v1/download/explanations/{pipeline_id}")
async def download_explanations(pipeline_id: str):
    """Download explanations JSON"""
    if pipeline_id not in pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    exp_path = OUTPUT_DIR / "aura_explanations.json"
    if not exp_path.exists():
        raise HTTPException(status_code=404, detail="Explanations not found")
    
    return FileResponse(
        path=exp_path,
        filename="aura_explanations.json",
        media_type="application/json"
    )


@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """LLM chat endpoint with Groq integration"""
    try:
        # Get dataset context if available
        dataset_context = None
        if request.dataset_id in datasets:
            dataset_info = datasets[request.dataset_id]
            dataset_context = {
                "dataset_name": dataset_info.get("filename", "Unknown"),
                "columns": dataset_info.get("columns", []),
                "target_column": dataset_info.get("target_column")
            }
        
        # Get LLM service and generate response
        llm_service = get_llm_service()
        response_message = llm_service.chat(
            message=request.message,
            dataset_context=dataset_context,
            conversation_history=request.conversation_history
        )
        
        return {
            "message": response_message,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/v1/llm/analyze-metadata")
async def analyze_metadata(request: LLMAnalysisRequest):
    """LLM metadata analysis with Groq integration"""
    try:
        # Get LLM service and analyze metadata
        llm_service = get_llm_service()
        recommendations = llm_service.analyze_dataset_metadata(request.metadata)
        
        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/api/v1/agent/run")
async def run_agent(request: AgentRunRequest):
    """Run the Agentic Preprocessor synchronously."""
    dataset_id = request.dataset_id
    
    # 1. Validation: Check if dataset is already in memory
    # If not in memory but present in file system, try to load it
    if dataset_id not in DATA_STORE:
        if dataset_id in datasets:
             # Lazy load from file system
             try:
                 filepath = datasets[dataset_id]["filepath"]
                 df = pd.read_csv(filepath)
                 register_dataset(dataset_id, df)
             except Exception as e:
                 raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")
        else:
            raise HTTPException(status_code=404, detail="Dataset not found or not loaded.")
            
    try:
        # 2. Instantiate and Run Agent
        agent = AuraAgent(dataset_id)
        final_state = agent.run()
        
        # 3. Format Response
        return {
            "status": final_state.status,
            "step_count": final_state.step_count,
            "last_error": final_state.last_error,
            "metadata_snapshot": final_state.metadata_snapshot,
            "recent_history": final_state.messages[-10:] if final_state.messages else []
        }
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
