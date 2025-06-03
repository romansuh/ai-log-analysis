# Logs Filtration using AI - Notebooks

## 📔 Jupyter Notebooks for Log Analysis

This directory contains interactive Jupyter notebooks for analyzing and preprocessing log data for machine learning.

### 🚀 Getting Started

1. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the main analysis notebook:**
   - Navigate to `log_analysis_and_preprocessing.ipynb`
   - Run cells sequentially to see the full analysis

### 📊 Available Notebooks

#### `log_analysis_and_preprocessing.ipynb`
**Main analysis notebook with comprehensive visualizations including:**

- **Setup & Data Loading**: Import libraries and load sample log data
- **Preprocessing Pipeline**: Step-by-step log cleaning with visualizations
- **Exploratory Data Analysis**: Interactive charts and word clouds
- **Feature Engineering**: ML feature creation and analysis
- **Statistical Analysis**: Comprehensive statistical insights
- **ML Dataset Preparation**: Export clean data for machine learning

### 🎯 Key Features

- **Interactive Visualizations**: Plotly charts, matplotlib plots, seaborn graphs
- **Word Clouds**: Visual representation of log content
- **Statistical Dashboard**: Comprehensive data analysis
- **Feature Correlation**: ML feature relationship analysis
- **Export Functionality**: Save processed data for ML models

### 📈 Visualizations Included

1. **Text Length Analysis**: Before/after preprocessing comparison
2. **Log Level Distribution**: Pie charts and bar graphs
3. **Word Clouds**: All messages and error-specific clouds
4. **Feature Correlation Heatmaps**: ML feature relationships
5. **Token Analysis**: Preserved meaningful tokens visualization
6. **Statistical Dashboards**: Multi-panel analysis views

### 🔧 Requirements

All required packages are listed in `../requirements.txt`:
- pandas, numpy (data manipulation)
- matplotlib, seaborn, plotly (visualizations) 
- wordcloud (text visualization)
- jupyter, ipykernel (notebook environment)

### 📁 Output Files

The notebook generates:
- `../data/ml_ready_logs.csv` - Processed dataset for ML
- `../data/feature_info.csv` - Feature metadata
- Interactive plots and visualizations

### 💡 Tips for Usage

1. **Run cells in order** - Each cell builds on the previous ones
2. **Interactive plots** - Plotly charts are interactive (zoom, hover, etc.)
3. **Modify sample data** - Edit the sample_logs list to test your own data
4. **Export results** - Use the final cells to save processed data
5. **Restart kernel** - If plots don't show, restart kernel and run all cells

### 🎓 For Coursework

This notebook is specifically designed for the "Logs Filtration using AI" coursework and includes:

- Theoretical explanations of each preprocessing step
- Visual proof of concept for AI-based log filtering
- ML-ready dataset preparation
- Statistical analysis suitable for academic presentation
- Comprehensive documentation and conclusions

**Perfect for demonstrating the practical implementation of AI-based log filtering systems!** 🚀 