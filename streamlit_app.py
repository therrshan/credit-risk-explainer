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
    """Train models if they don't exist - lightweight version"""
    models_dir = Path(config.MODELS_DIR)
    required_files = ["xgboost.pkl", "random_forest.pkl", "logistic_regression.pkl", "model_scores.pkl"]
    
    missing_files = [f for f in required_files if not (models_dir / f).exists()]
    
    if missing_files:
        st.info("🔄 Training models... This will take 2-3 minutes.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create directories
            status_text.text("Creating directories...")
            progress_bar.progress(10)
            
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            os.makedirs(config.DATA_DIR, exist_ok=True)
            os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
            
            # Load data
            status_text.text("Loading data...")
            progress_bar.progress(20)
            from src.data_loader import load_german_credit
            X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, loader = load_german_credit()
            
            # Train models with reduced complexity
            status_text.text("Training XGBoost...")
            progress_bar.progress(40)
            
            import xgboost as xgb
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
            
            models = {}
            scores = {}
            
            # Simplified XGBoost
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=50,  # Reduced from 200
                max_depth=3,      # Reduced from 6
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            models['xgboost'].fit(X_train, y_train)
            
            status_text.text("Training Random Forest...")
            progress_bar.progress(60)
            
            # Simplified Random Forest
            models['random_forest'] = RandomForestClassifier(
                n_estimators=50,  # Reduced from 200
                max_depth=5,      # Reduced from 10
                random_state=42
            )
            models['random_forest'].fit(X_train, y_train)
            
            status_text.text("Training Logistic Regression...")
            progress_bar.progress(80)
            
            # Logistic Regression (fast)
            models['logistic_regression'] = LogisticRegression(
                random_state=42,
                max_iter=500
            )
            models['logistic_regression'].fit(X_train, y_train)
            
            status_text.text("Evaluating models...")
            progress_bar.progress(90)
            
            # Quick evaluation with full metrics
            for name, model in models.items():
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                cm = confusion_matrix(y_test, y_pred)
                
                scores[name] = {
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_auc': roc_auc_score(y_test, y_proba),
                    'test_precision': precision_score(y_test, y_pred),
                    'test_recall': recall_score(y_test, y_pred),
                    'test_f1': f1_score(y_test, y_pred),
                    'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
                    'pr_curve': {'precision': precision.tolist(), 'recall': recall.tolist()},
                    'confusion_matrix': cm.tolist()
                }
            
            status_text.text("Saving models...")
            progress_bar.progress(95)
            
            # Save models
            for name, model in models.items():
                joblib.dump(model, models_dir / f"{name}.pkl")
            
            # Save scores
            joblib.dump(scores, models_dir / "model_scores.pkl")
            
            progress_bar.progress(100)
            status_text.text("✅ Models trained successfully!")
            
            return True
                
        except Exception as e:
            st.error(f"❌ Error training models: {str(e)}")
            st.error("Try refreshing the page or check the GitHub repository for pre-trained models.")
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
    
    try:
        explainer.initialize_shap_explainers()
        # Reduced sample size for deployment
        explainer.compute_shap_values()  # Much smaller
    except Exception as e:
        st.warning(f"⚠️ SHAP initialization failed: {str(e)}")
        st.info("Some explainability features may be limited")
    
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
        st.plotly_chart(fig_auc, use_container_width=True, key="model_performance_auc")
    
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
        st.plotly_chart(fig_roc, use_container_width=True, key="model_performance_roc")
    
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
            st.plotly_chart(fig_shap, use_container_width=True, key="individual_shap_explanation")
            
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
        st.plotly_chart(fig_lime, use_container_width=True, key="individual_lime_explanation")
        
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")

def show_global_explanations(explainer, model_name):
    st.header("📈 Global Model Explanations")
    
    if not hasattr(explainer, 'shap_values') or model_name not in explainer.shap_values:
        st.warning("⚠️ SHAP values not available. This may be due to memory constraints in the deployed environment.")
        st.info("Global explanations work best when run locally with sufficient memory.")
        return
    
    shap_vals = explainer.shap_values[model_name]
    if shap_vals is None or len(shap_vals) == 0:
        st.warning("⚠️ SHAP values are empty. Explainability features may be limited in deployment.")
        return
    
    # Global feature importance (fallback method)
    st.subheader("Global Feature Importance")
    
    try:
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': explainer.feature_names[:len(feature_importance)],  # Match array sizes
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
        st.plotly_chart(fig_importance, use_container_width=True, key="global_feature_importance")
        
    except Exception as e:
        st.error(f"Error generating feature importance: {str(e)}")
        return
    
    # SHAP Summary Plots (with reduced complexity for deployment)
    st.subheader("SHAP Summary Plots")
    
    try:
        X_sample = explainer.X_test.iloc[:min(len(shap_vals), 50)]  # Limit sample size
        shap_vals_sample = shap_vals[:len(X_sample)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bar Plot (Feature Importance)**")
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                shap.summary_plot(shap_vals_sample, X_sample, 
                                 feature_names=explainer.feature_names,
                                 max_display=10, show=False, plot_type="bar")  # Reduced features
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Bar plot failed: {str(e)}")
        
        with col2:
            st.write("**Beeswarm Plot (Value Impact)**")
            try:
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                shap.summary_plot(shap_vals_sample, X_sample, 
                                 feature_names=explainer.feature_names,
                                 max_display=10, show=False)  # Reduced features
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"Beeswarm plot failed: {str(e)}")
                
    except Exception as e:
        st.error(f"Error generating SHAP summary plots: {str(e)}")
        st.info("SHAP summary plots may not work in deployment due to memory constraints.")

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
    
    try:
        model = explainer.models.get_model(model_name)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Performance by group
        st.subheader(f"Performance by {sensitive_attr}")
        
        groups = sensitive_test[sensitive_attr].unique()
        group_stats = []
        
        for group in groups:
            try:
                group_mask = sensitive_test[sensitive_attr] == group
                group_predictions = predictions[group_mask]
                group_probabilities = probabilities[group_mask]
                group_actual = y_test[group_mask]
                
                if len(group_actual) == 0:
                    continue
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
                
                # Handle edge cases
                try:
                    precision = precision_score(group_actual, group_predictions, zero_division=0)
                    recall = recall_score(group_actual, group_predictions, zero_division=0)
                    auc = roc_auc_score(group_actual, group_probabilities) if len(np.unique(group_actual)) > 1 else 0.5
                except:
                    precision = recall = auc = 0.0
                
                stats = {
                    'Group': f'Group {group}',
                    'Sample Size': len(group_actual),
                    'Accuracy': accuracy_score(group_actual, group_predictions),
                    'Precision': precision,
                    'Recall': recall,
                    'AUC': auc,
                    'Positive Rate': (group_predictions == 1).mean()
                }
                group_stats.append(stats)
            except Exception as e:
                st.error(f"Error processing group {group}: {str(e)}")
                continue
        
        if group_stats:
            stats_df = pd.DataFrame(group_stats)
            st.dataframe(stats_df.round(4), use_container_width=True)
            
            # Fairness metrics
            st.subheader("Fairness Metrics")
            
            if len(groups) == 2 and len(group_stats) == 2:
                try:
                    group_0_mask = sensitive_test[sensitive_attr] == groups[0]
                    group_1_mask = sensitive_test[sensitive_attr] == groups[1]
                    
                    pos_rate_0 = (predictions[group_0_mask] == 1).mean()
                    pos_rate_1 = (predictions[group_1_mask] == 1).mean()
                    
                    disparate_impact = pos_rate_0 / pos_rate_1 if pos_rate_1 > 0 else float('inf')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        di_status = "Perfect: 1.0" if abs(disparate_impact - 1.0) < 0.1 else "Biased" if disparate_impact < 0.8 or disparate_impact > 1.2 else "Acceptable"
                        st.metric("Disparate Impact", f"{disparate_impact:.3f}", delta=di_status)
                    
                    with col2:
                        equal_opp_diff = abs(stats_df.iloc[0]['Recall'] - stats_df.iloc[1]['Recall'])
                        eo_status = "Perfect: 0.0" if equal_opp_diff < 0.05 else "Acceptable" if equal_opp_diff < 0.1 else "Biased"
                        st.metric("Equal Opportunity Difference", f"{equal_opp_diff:.3f}", delta=eo_status)
                        
                except Exception as e:
                    st.error(f"Error calculating fairness metrics: {str(e)}")
            
            # SHAP-based bias analysis (with error handling)
            if hasattr(explainer, 'shap_values') and model_name in explainer.shap_values:
                st.subheader("SHAP-based Bias Analysis")
                
                try:
                    shap_vals = explainer.shap_values[model_name]
                    if shap_vals is not None and len(shap_vals) > 0:
                        sensitive_vals = sensitive_test[sensitive_attr].iloc[:min(len(shap_vals), len(sensitive_test[sensitive_attr]))]
                        
                        bias_data = []
                        for group in groups:
                            group_mask = sensitive_vals == group
                            if group_mask.sum() > 0:
                                group_shap = shap_vals[group_mask]
                                mean_abs_shap = np.abs(group_shap).mean(axis=0)
                                
                                for i, feature in enumerate(explainer.feature_names[:len(mean_abs_shap)]):
                                    bias_data.append({
                                        'Group': f'Group {group}',
                                        'Feature': feature,
                                        'Mean_Abs_SHAP': mean_abs_shap[i]
                                    })
                        
                        if bias_data:
                            bias_df = pd.DataFrame(bias_data)
                            top_features = bias_df.groupby('Feature')['Mean_Abs_SHAP'].sum().nlargest(8).index
                            bias_subset = bias_df[bias_df['Feature'].isin(top_features)]
                            
                            fig_bias = px.bar(
                                bias_subset,
                                x='Mean_Abs_SHAP',
                                y='Feature',
                                color='Group',
                                orientation='h',
                                title=f"Feature Impact by {sensitive_attr}",
                                barmode='group',
                                height=400
                            )
                            st.plotly_chart(fig_bias, use_container_width=True, key="fairness_shap_bias_analysis")
                        else:
                            st.info("No SHAP bias data available")
                    else:
                        st.info("SHAP values not available for bias analysis")
                        
                except Exception as e:
                    st.error(f"Error in SHAP bias analysis: {str(e)}")
                    st.info("SHAP-based bias analysis unavailable due to memory constraints")
            else:
                st.info("SHAP explainer not available for bias analysis")
        else:
            st.error("Could not process group statistics")
            
    except Exception as e:
        st.error(f"Error in fairness analysis: {str(e)}")
        st.info("Fairness analysis unavailable. This may be due to deployment resource constraints.")

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
    
    st.plotly_chart(fig_gauge, use_container_width=True, key="interactive_probability_gauge")

def show_demo_mode():
    """Show demo mode with sample data"""
    st.info("🎯 Demo Mode: Showing sample results")
    
    # Sample performance data
    demo_scores = {
        'xgboost': {
            'test_accuracy': 0.783,
            'test_auc': 0.759,
            'test_precision': 0.712,
            'test_recall': 0.634,
            'test_f1': 0.671
        },
        'random_forest': {
            'test_accuracy': 0.765,
            'test_auc': 0.748,
            'test_precision': 0.694,
            'test_recall': 0.612,
            'test_f1': 0.651
        },
        'logistic_regression': {
            'test_accuracy': 0.754,
            'test_auc': 0.735,
            'test_precision': 0.681,
            'test_recall': 0.587,
            'test_f1': 0.631
        }
    }
    
    st.header("📊 Model Performance Comparison (Demo)")
    
    # Performance metrics table
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame(demo_scores).T
    display_metrics = ['test_accuracy', 'test_auc', 'test_precision', 'test_recall', 'test_f1']
    st.dataframe(metrics_df[display_metrics].round(4), use_container_width=True)
    
    # Best model highlight
    best_model = metrics_df['test_auc'].idxmax()
    st.success(f"🏆 Best Model (by AUC): **{best_model.title()}** (AUC: {metrics_df.loc[best_model, 'test_auc']:.4f})")
    
    # AUC Comparison
    st.subheader("AUC Comparison")
    fig_auc = px.bar(
        x=metrics_df.index,
        y=metrics_df['test_auc'],
        title="Test AUC by Model",
        labels={'x': 'Model', 'y': 'AUC Score'}
    )
    fig_auc.update_layout(showlegend=False)
    st.plotly_chart(fig_auc, use_container_width=True, key="demo_auc_comparison")
    
    st.info("💡 To access full functionality, the models need to be trained. This requires pre-training the models locally and pushing them to the repository.")

if __name__ == "__main__":
    main()