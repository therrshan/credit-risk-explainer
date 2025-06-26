import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import sys
import os
from pathlib import Path

# Since this file is in the root, we can import directly
try:
    import config
    from src.model import CreditRiskModels
    from src.explain import CreditExplainer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error(f"Current working directory: {os.getcwd()}")
    st.error(f"Files in current directory: {os.listdir('.')}")
    st.stop()

st.set_page_config(
    page_title="Credit Risk Explainer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def train_models_if_needed():
    """Train models if they don't exist"""
    models_dir = Path(config.MODELS_DIR)
    required_files = ["xgboost.pkl", "random_forest.pkl", "logistic_regression.pkl", "model_scores.pkl"]
    
    missing_files = [f for f in required_files if not (models_dir / f).exists()]
    
    if missing_files:
        st.info("🔄 Models not found. Training models now... This may take a few minutes.")
        
        with st.spinner("Training models..."):
            try:
                os.makedirs(config.MODELS_DIR, exist_ok=True)
                os.makedirs(config.DATA_DIR, exist_ok=True)
                os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
                
                from src.model import train_baseline_models
                models = train_baseline_models(hyperparameter_tuning=False, save_models=True)
                
                st.success("✅ Models trained successfully!")
                return True
                
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
                return False
    
    return True

@st.cache_data
def load_model_scores():
    scores_path = Path(config.MODELS_DIR) / "model_scores.pkl"
    if scores_path.exists():
        return joblib.load(str(scores_path))
    return None

@st.cache_resource
def load_models_and_data():
    models = CreditRiskModels()
    models.load_models()
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = models.get_data()
    return models, X_train, X_test, y_train, y_test, sensitive_train, sensitive_test

@st.cache_resource
def load_explainer(_models):
    explainer = CreditExplainer(_models)
    explainer.load_models_and_data()
    explainer.initialize_shap_explainers()
    explainer.compute_shap_values(sample_size=200)
    return explainer

def main():
    st.title("🏦 Credit Risk Explainer Dashboard")
    st.markdown("### Explainable AI for Credit Risk Assessment")
    
    # Check if models need to be trained
    if not train_models_if_needed():
        st.error("Failed to train models. Please check the logs.")
        return
    
    # Load data
    scores = load_model_scores()
    if scores is None:
        st.error("❌ Model scores not found. Please refresh the page.")
        return
    
    try:
        models, X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = load_models_and_data()
        explainer = load_explainer(models)
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Configuration")
    
    # Get best model
    best_model = max(scores.keys(), key=lambda x: scores[x]['test_auc'])
    st.sidebar.success(f"🏆 Using Best Model: **{best_model.title()}**")
    st.sidebar.metric("AUC Score", f"{scores[best_model]['test_auc']:.4f}")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["📊 Model Performance", 
         "🔍 Individual Prediction", 
         "📈 Global Explanations", 
         "⚖️ Fairness Analysis",
         "📋 Interactive Prediction"]
    )
    
    if page == "📊 Model Performance":
        show_model_performance(scores)
    elif page == "🔍 Individual Prediction":
        show_individual_prediction(explainer, best_model, X_test, y_test)
    elif page == "📈 Global Explanations":
        show_global_explanations(explainer, best_model)
    elif page == "⚖️ Fairness Analysis":
        show_fairness_analysis(explainer, best_model, X_test, y_test, sensitive_test)
    elif page == "📋 Interactive Prediction":
        show_interactive_prediction(models, best_model, explainer)

def show_model_performance(scores):
    st.header("📊 Model Performance Comparison")
    
    # Performance metrics table
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame(scores).T
    display_metrics = ['test_accuracy', 'test_auc', 'test_precision', 'test_recall', 'test_f1']
    st.dataframe(metrics_df[display_metrics].round(4), use_container_width=True)
    
    # Best model highlight
    best_model = metrics_df['test_auc'].idxmax()
    st.success(f"🏆 Best Model (by AUC): **{best_model.title()}** (AUC: {metrics_df.loc[best_model, 'test_auc']:.4f})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AUC Comparison")
        fig_auc = px.bar(
            x=metrics_df.index,
            y=metrics_df['test_auc'],
            title="Test AUC by Model",
            labels={'x': 'Model', 'y': 'AUC Score'}
        )
        fig_auc.update_layout(showlegend=False)
        st.plotly_chart(fig_auc, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curves")
        fig_roc = go.Figure()
        
        for model_name in scores.keys():
            fpr = scores[model_name]['roc_curve']['fpr']
            tpr = scores[model_name]['roc_curve']['tpr']
            auc = scores[model_name]['test_auc']
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name.title()} (AUC: {auc:.3f})',
                line=dict(width=2)
            ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(len(scores))
    
    for i, (model_name, model_scores) in enumerate(scores.items()):
        with cols[i]:
            cm = np.array(model_scores['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Bad Credit', 'Good Credit'],
                       yticklabels=['Bad Credit', 'Good Credit'],
                       ax=ax)
            ax.set_title(f'{model_name.title()}')
            st.pyplot(fig)

def show_individual_prediction(explainer, model_name, X_test, y_test):
    st.header("🔍 Individual Prediction Analysis")
    
    # Sample selection
    sample_idx = st.selectbox("Select Sample", range(min(50, len(X_test))))
    
    # Get prediction
    model = explainer.models.get_model(model_name)
    sample_data = X_test.iloc[sample_idx:sample_idx+1]
    prediction = model.predict(sample_data)[0]
    probability = model.predict_proba(sample_data)[0]
    actual = y_test.iloc[sample_idx]
    
    # Display prediction info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Actual Label", 
                 "Good Credit" if actual == 1 else "Bad Credit")
    
    with col2:
        st.metric("Predicted Label", 
                 "Good Credit" if prediction == 1 else "Bad Credit",
                 delta="✅ Correct" if prediction == actual else "❌ Wrong")
    
    with col3:
        st.metric("Confidence", 
                 f"{max(probability):.1%}",
                 delta=f"Good: {probability[1]:.1%}")
    
    # Feature values
    st.subheader("Feature Values")
    feature_df = pd.DataFrame({
        'Feature': explainer.feature_names,
        'Value': sample_data.iloc[0].values
    })
    st.dataframe(feature_df, use_container_width=True)
    
    # SHAP explanation as bar chart (more reliable than waterfall)
    st.subheader("SHAP Feature Contributions")
    if model_name in explainer.shap_values:
        try:
            shap_vals = explainer.shap_values[model_name]
            sample_shap = shap_vals[sample_idx]
            
            contrib_df = pd.DataFrame({
                'Feature': explainer.feature_names,
                'Contribution': sample_shap
            }).sort_values('Contribution', key=abs, ascending=False).head(10)
            
            fig_shap = px.bar(
                contrib_df,
                x='Contribution',
                y='Feature',
                orientation='h',
                title="Top 10 Feature Contributions (SHAP Values)",
                color='Contribution',
                color_continuous_scale='RdBu',
                height=400
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
    
    # LIME Explanation
    st.subheader("LIME Explanation")
    try:
        if explainer.lime_explainer is None:
            explainer.initialize_lime_explainer()
        
        explanation = explainer.lime_explainer.explain_instance(
            sample_data.iloc[0].values, 
            model.predict_proba, 
            num_features=10
        )
        
        # Convert LIME explanation to DataFrame
        lime_data = explanation.as_list()
        lime_df = pd.DataFrame(lime_data, columns=['Feature', 'Impact'])
        lime_df = lime_df.sort_values('Impact', key=abs, ascending=False)
        
        fig_lime = px.bar(
            lime_df, 
            x='Impact', 
            y='Feature',
            orientation='h',
            title="LIME Feature Impact",
            color='Impact',
            color_continuous_scale='RdBu',
            height=400
        )
        st.plotly_chart(fig_lime, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")

def show_global_explanations(explainer, model_name):
    st.header("📈 Global Model Explanations")
    
    if model_name not in explainer.shap_values:
        st.error("SHAP values not computed for this model.")
        return
    
    # Global feature importance
    st.subheader("Global Feature Importance")
    shap_vals = explainer.shap_values[model_name]
    
    # Calculate mean absolute SHAP values
    feature_importance = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': explainer.feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True).tail(15)
    
    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top 15 Features - {model_name.title()}",
        height=600
    )
    st.plotly_chart(fig_importance, use_container_width=True)

def show_fairness_analysis(explainer, model_name, X_test, y_test, sensitive_test):
    st.header("⚖️ Fairness Analysis")
    
    if not sensitive_test:
        st.warning("No sensitive attributes available for fairness analysis.")
        return
    
    # Select sensitive attribute
    sensitive_attr = st.selectbox(
        "Select Sensitive Attribute",
        list(sensitive_test.keys())
    )
    
    model = explainer.models.get_model(model_name)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Performance by group
    st.subheader(f"Performance by {sensitive_attr}")
    
    groups = sensitive_test[sensitive_attr].unique()
    group_stats = []
    
    for group in groups:
        group_mask = sensitive_test[sensitive_attr] == group
        group_predictions = predictions[group_mask]
        group_probabilities = probabilities[group_mask]
        group_actual = y_test[group_mask]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
        stats = {
            'Group': f'Group {group}',
            'Sample Size': len(group_actual),
            'Accuracy': accuracy_score(group_actual, group_predictions),
            'Precision': precision_score(group_actual, group_predictions),
            'Recall': recall_score(group_actual, group_predictions),
            'AUC': roc_auc_score(group_actual, group_probabilities),
            'Positive Rate': (group_predictions == 1).mean()
        }
        group_stats.append(stats)
    
    stats_df = pd.DataFrame(group_stats)
    st.dataframe(stats_df.round(4), use_container_width=True)

def show_interactive_prediction(models, model_name, explainer):
    st.header("📋 Interactive Credit Risk Prediction")
    
    st.markdown("Adjust the feature values below to see how they affect the credit risk prediction:")
    
    # Create input widgets for features
    feature_inputs = {}
    
    # Get sample data to understand feature ranges
    sample_data = explainer.X_test.iloc[0]
    
    col1, col2 = st.columns(2)
    
    features_half = len(explainer.feature_names) // 2
    
    with col1:
        for feature in explainer.feature_names[:features_half]:
            min_val = float(explainer.X_test[feature].min())
            max_val = float(explainer.X_test[feature].max())
            default_val = float(sample_data[feature])
            
            feature_inputs[feature] = st.slider(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"slider_{feature}"
            )
    
    with col2:
        for feature in explainer.feature_names[features_half:]:
            min_val = float(explainer.X_test[feature].min())
            max_val = float(explainer.X_test[feature].max())
            default_val = float(sample_data[feature])
            
            feature_inputs[feature] = st.slider(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"slider_{feature}_2"
            )
    
    # Make prediction
    input_data = pd.DataFrame([feature_inputs])
    model = models.get_model(model_name)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction", "Good Credit" if prediction == 1 else "Bad Credit")
    
    with col2:
        st.metric("Confidence", f"{max(probability):.1%}")
    
    with col3:
        st.metric("Good Credit Probability", f"{probability[1]:.1%}")
    
    # Probability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability[1] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Good Credit Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    st.plotly_chart(fig_gauge, use_container_width=True)

if __name__ == "__main__":
    main()