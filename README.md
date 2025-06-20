# UEFA Champions League Finals - Extra Time Prediction

This machine learning project analyzes UEFA Champions League finals data from 1955-2023 to predict whether matches will go to extra time.

## Project Overview

The project uses various machine learning algorithms to predict if a Champions League final will require extra time based on historical match data including:

- Country of the final
- Winners and Runners-up teams
- Venue information
- Attendance numbers
- Season data
- Team performance metrics (SW, SR, SW-SR)

## Features

- **Multiple ML Models**: Logistic Regression, Decision Tree, K-Nearest Neighbors (k=3,5,7), Random Forest
- **Comprehensive Evaluation**: 10-fold cross-validation with multiple metrics
- **Visualization**: ROC curves, confusion matrices, and performance comparison charts
- **Detailed Analysis**: Accuracy, Precision, Recall, F1-score, and ROC AUC metrics

## Files Description

- `main.py`: Main script containing the ML pipeline and analysis
- `UCL_Finals_1955-2023.csv`: Historical Champions League finals data
- `UCL_AllTime_Performance_Table.csv`: Team performance statistics
- `requirements.txt`: Python dependencies
- `resultados_validacao_cruzada.csv`: Cross-validation results
- Various PNG files: Generated visualizations and charts

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:

```bash
python main.py
```

This will:

- Load and preprocess the Champions League finals data
- Train multiple machine learning models
- Perform 10-fold cross-validation
- Generate performance visualizations
- Save results to CSV and image files

## Results

The analysis generates several visualizations:

- Performance comparison charts for each metric
- ROC curves for all models
- Confusion matrices for each classifier
- Detailed cross-validation results

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## License

This project is for educational and research purposes.
