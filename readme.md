# Microplastic Risk Assessment Pipeline

A concise framework for probabilistic microplastic risk assessment across groundwater, surface water, and sediment environments using ensemble machine learning with Bayesian uncertainty quantification.

## Features
- **Data Loading & Preprocessing**: Reads Excel datasets, handles missing values, engineers hazard & diversity features.
- **Risk Indices**: Calculates Pollution Load Index (PLI), Risk Quotient (RQ), Hazard Quotient (HQ) with Monte Carlo uncertainty, and an integrated risk score.
- **Modeling**: Trains ensemble regressors (Random Forest, SVM, Gaussian Process) and classifiers (or rule-based fallback) with cross-validation.
- **Prediction**: Provides abundance and risk category predictions with confidence intervals.
- **Cross-Environment Analysis**: Compares abundance ratios, morphology & polymer distributions, and pathway similarities.
- **Reporting & Visualization**: Generates text report and saves plots summarizing results.

## Requirements
- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- matplotlib
- openpyxl

## Installation
```bash
git clone https://your-repo-url.git
cd your-repo
pip install -r requirements.txt