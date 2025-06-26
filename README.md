# Credit Risk Explainer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-explain.streamlit.app)

An explainable AI dashboard for credit risk assessment using machine learning models with SHAP and LIME explanations, plus comprehensive fairness analysis.

## Features

- **Model Performance**: Compare XGBoost, Random Forest, and Logistic Regression models
- **Individual Predictions**: Detailed SHAP and LIME explanations for single predictions
- **Global Explanations**: Feature importance analysis across all predictions
- **Fairness Analysis**: Bias detection across demographic groups
- **Interactive Predictions**: Real-time prediction with adjustable feature values

## Live Demo

[View the live dashboard](https://credit-risk-explain.streamlit.app)

## Run Locally

```bash
# Clone repository
git clone https://github.com/yourusername/credit-risk-explainer.git
cd credit-risk-explainer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run streamlit_app.py
```

The app will automatically download data, train models, and launch the dashboard. To add extra models, add them to src/models.py before starting the dashboard.

## Tech Stack

- **Frontend**: Streamlit
- **ML Models**: XGBoost, Random Forest, Logistic Regression
- **Explainability**: SHAP, LIME
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dataset**: German Credit Dataset (UCI ML Repository)
