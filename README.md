# AURA Preprocessor 2.0

**Dataset-Agnostic Machine Learning Pipeline with LLM-Powered Chatbot**

AURA Preprocessor 2.0 is a full-stack ML preprocessing platform that automatically handles any CSV dataset. Features intelligent target detection, modular processing steps, comprehensive reporting, and an **AI-powered chatbot** using Groq API for guided preprocessing.

## ✅ Running the Project (important)

- **Backend API (FastAPI)** serves **JSON** (and Swagger docs) on `http://localhost:8000`
- **Frontend UI (React/Vite)** runs separately and is the URL you should open in your browser (usually `http://localhost:3000`, or `3001/3002` if busy)

➡️ Follow: **`HOW_TO_RUN.md`**

## 🚀 Key Features

### Backend (Python ML Pipeline)
- **Dataset Agnostic**: Works with any CSV file, not just Titanic
- **Intelligent Target Detection**: Automatically identifies target columns
- **Modular Architecture**: Clean separation of preprocessing steps
- **Dual Modes**: Interactive (`step`) and automatic (`auto`) execution
- **Comprehensive Reporting**: Detailed JSON reports with recommendations
- **Error Handling**: Robust error handling and logging

### Frontend (React + TypeScript) 🆕
- **Modern Web UI**: React 18 with Material-UI
- **LLM Chatbot**: Floating chatbot with conversation history (Groq API)
- **Auto Mode**: LLM recommends preprocessing strategies with explanations
- **Manual Mode**: Step-by-step wizard for custom configuration
- **Real-time Progress**: Live pipeline execution tracking
- **Interactive Visualizations**: Charts and metrics dashboard
- **File Downloads**: Export processed data and reports

### LLM Integration (Groq API) ✨
- **Intelligent Analysis**: Automatically analyzes dataset characteristics
- **Smart Recommendations**: Column-specific strategies for missing values, encoding, scaling
- **Risk Awareness**: Highlights potential issues with each approach
- **Interactive Chat**: Ask questions and get context-aware guidance
- **Educational**: Detailed explanations for every recommendation

## 📁 Project Structure

```
aura_preprocessor/
├── frontend/                    # React + TypeScript Frontend
│   ├── src/
│   │   ├── api/                # Backend API integration
│   │   ├── components/         # React components
│   │   ├── context/            # State management (Chat, Pipeline, Wizard)
│   │   ├── pages/              # Landing, Dataset, Pipeline, Results, Wizard
│   │   └── types/              # TypeScript definitions
│   ├── package.json
│   └── README.md               # Frontend documentation
│
├── src/                        # Backend (Python ML Pipeline)
│   ├── pipeline.py             # Main pipeline orchestrator
│   ├── llm_helper.py           # LLM explanation generator (optional)
│   └── steps/
│       ├── missing_values.py   # Missing value handling
│       ├── encoding.py         # Feature encoding
│       ├── scaling.py          # Feature scaling
│       ├── model_training.py   # ML model training
│       └── report_generator.py # Report generation
│
├── data/                       # Input datasets
├── outputs/                    # Generated outputs
├── uploads/                    # API uploaded files
├── main.py                     # CLI entry point
├── api_server.py               # FastAPI REST server
├── requirements.txt            # Python dependencies
├── QUICK_START.md              # Quick start guide
└── README.md                   # This file
```

## 🛠️ Installation

### Backend (Python)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd aura_preprocessor
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Groq API** (for LLM features):
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key from https://console.groq.com/
   ```

### Frontend (React) 🆕

1. **Navigate to frontend**:
   ```bash
   cd frontend
   ```

2. **Install npm packages**:
   ```bash
   npm install
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your backend URL
   ```

4. **Start development server**:
   ```bash
   npm run dev
   ```

Frontend runs on `http://localhost:3000`

**📚 See `frontend/README.md` for complete frontend documentation**

## 🎯 Usage

### Basic Usage

```bash
python main.py
```

This will process the default Titanic dataset in automatic mode.

### Advanced Usage

```bash
# Process any CSV file
python main.py data/your_dataset.csv

# Use interactive mode
python main.py data/your_dataset.csv step

# Specify target column
python main.py data/your_dataset.csv auto target_column_name
```

### Programmatic Usage

```python
from src.pipeline import AuraPipeline

# Initialize pipeline
pipeline = AuraPipeline(
    filepath="data/your_dataset.csv",
    mode="auto",  # or "step"
    target_col=None  # Auto-detect if None
)

# Run complete pipeline
results = pipeline.run_full_pipeline()

# Or run individual steps
pipeline.handle_missing_values()
pipeline.encode_features()
model_results = pipeline.train_model()
report = pipeline.generate_report(model_results)
```

## 🔧 Configuration

### Execution Modes

- **`auto`**: Fully automated processing with intelligent defaults
- **`step`**: Interactive mode with user choices and explanations

### Target Column Detection

The pipeline automatically detects target columns by looking for common names:
- `target`, `label`, `y`, `class`, `outcome`, `result`
- `survived`, `price`, `sales`, `revenue`, `profit`
- Falls back to the last column if no match found

### Preprocessing Steps

1. **Missing Values**: Intelligent handling based on data type and missing percentage
2. **Feature Encoding**: Automatic choice between label and one-hot encoding
3. **Feature Scaling**: Automatic scaler selection based on data characteristics
4. **Model Training**: Multiple algorithms with automatic selection
5. **Report Generation**: Comprehensive analysis and recommendations

## 📊 Outputs

The pipeline generates several output files in the `outputs/` directory:

- **`{dataset}_processed.csv`**: Cleaned and processed dataset
- **`report.json`**: Comprehensive pipeline report
- **`aura_explanations.json`**: LLM explanations for each step
- **`{model}_info.json`**: Model training information

### Report Contents

- Dataset summary (before/after processing)
- Preprocessing step details
- Model performance metrics
- Recommendations for improvement
- Feature analysis and statistics

## 🤖 LLM Explanations

The system provides AI-powered explanations for each preprocessing step:

- **Missing Values**: Why and how missing values were handled
- **Encoding**: Explanation of label vs one-hot encoding choices
- **Scaling**: Why specific scalers were selected
- **Model Training**: Performance interpretation and next steps

## 🔍 Supported Algorithms

### Missing Value Handling
- Drop columns with high missing percentage (>50%)
- Mean/median filling for numeric columns
- Mode filling for categorical columns

### Feature Encoding
- Label Encoding (for ordinal data)
- One-Hot Encoding (for nominal data)
- Automatic selection based on cardinality

### Feature Scaling
- StandardScaler (mean=0, std=1)
- MinMaxScaler (range 0-1)
- RobustScaler (median=0, IQR=1)
- Automatic selection based on outlier detection

### Machine Learning Models
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Support Vector Machine
- Automatic selection based on dataset characteristics

## 📈 Performance Features

- **Intelligent Defaults**: Automatic parameter selection
- **Outlier Detection**: IQR-based outlier identification
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Stratified Splitting**: Maintains class distribution in train/test splits
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score

## 🛡️ Error Handling

- **File Validation**: Checks for file existence and format
- **Data Validation**: Validates target column presence
- **Graceful Degradation**: Continues processing when possible
- **Detailed Logging**: Comprehensive error messages and logging
- **User-Friendly Messages**: Clear error explanations

## 🧪 Testing

Test the pipeline with different datasets:

```bash
# Test with Titanic dataset
python main.py data/titanic.csv

# Test with other datasets
python main.py data/your_dataset.csv step
```

## 🤖 LLM Chatbot Integration 🆕

### Features

1. **Floating Chatbot Button**
   - Always accessible in bottom-right corner
   - Maintains conversation history
   - Dataset-aware responses

2. **Auto Mode (LLM-Powered)**
   - User selects "Auto Mode"
   - Chatbot window opens automatically
   - LLM analyzes dataset metadata (via Groq API)
   - Recommends preprocessing strategies
   - User can review & override before execution

3. **Conversation Context**
   - LLM has access to dataset metadata
   - Remembers previous questions
   - Provides context-aware recommendations

### How It Works

```
1. Upload CSV → Metadata extracted
2. Select "Auto Mode" → Chat opens
3. LLM analyzes → Recommends strategies
4. User reviews → Can override
5. Click "Run" → Pipeline executes
```

**Status:** ⏳ Awaiting Groq API key integration

## 🔮 Current Status & Roadmap

### ✅ Completed
- Python ML pipeline (Backend V1)
- React frontend structure
- LLM chatbot UI (ready for integration)
- API client & type definitions
- Comprehensive documentation

### ⏳ In Progress
- Backend V2 (FastAPI wrapper) - Awaiting from teammate
- Groq API integration for LLM
- Frontend component implementation
- End-to-end testing

### 📋 Future Enhancements
- **AutoML Integration**: Automated hyperparameter tuning
- **Feature Engineering**: Advanced feature creation
- **Model Interpretability**: SHAP/LIME explanations
- **WebSocket**: Real-time updates (instead of polling)
- **User Authentication**: Login and project history
- **Dark Mode**: Theme toggle

## 📝 Dependencies

- **Core ML**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Utilities**: scipy, joblib, python-dateutil
- **Logging**: Built-in Python logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with scikit-learn and pandas
- Inspired by modern MLOps practices
- Designed for educational and production use

---

## 📚 Documentation

- **Quick Start**: `QUICK_START.md` - Get started quickly
- **Frontend Guide**: `frontend/README.md` - React app documentation
- **LLM Integration**: `LLM_INTEGRATION.md` - Groq API setup and usage

## 🚦 Quick Start Guide

### For Backend Development
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the ML pipeline
python main.py
```

### For Frontend Development
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev

# Open browser at http://localhost:3000
```

### For Full-Stack Development
```bash
# Terminal 1: Backend API Server
python api_server.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

---

**AURA Preprocessor 2.0** - Making machine learning preprocessing accessible and intelligent! 🚀


````

