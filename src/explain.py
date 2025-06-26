import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import CreditRiskModels

class CreditExplainer:
    def __init__(self, models=None):
        self.models = models
        self.shap_explainers = {}
        self.shap_values = {}
        self.lime_explainer = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sensitive_train = None
        self.sensitive_test = None
        
    def load_models_and_data(self, model_path=None):
        if model_path is None:
            model_path = config.MODELS_DIR
            
        if self.models is None:
            self.models = CreditRiskModels()
            self.models.load_models(model_path)
            
        self.X_train, self.X_test, self.y_train, self.y_test, self.sensitive_train, self.sensitive_test = self.models.get_data()
        self.feature_names = self.models.get_feature_names()
        print(f"Loaded data: {len(self.feature_names)} features")
        
    def initialize_shap_explainers(self, model_names=None):
        if model_names is None:
            model_names = list(self.models.models.keys())
            
        print("Initializing SHAP explainers...")
        for model_name in model_names:
            model = self.models.get_model(model_name)
            
            if model_name in ['xgboost', 'random_forest']:
                self.shap_explainers[model_name] = shap.TreeExplainer(model)
            else:
                self.shap_explainers[model_name] = shap.Explainer(model, self.X_train)
                
            print(f"SHAP explainer initialized for {model_name}")
            
    def compute_shap_values(self, model_names=None, sample_size=None):
        if model_names is None:
            model_names = list(self.shap_explainers.keys())
            
        print("Computing SHAP values...")
        for model_name in model_names:
            explainer = self.shap_explainers[model_name]
            
            if sample_size:
                X_sample = self.X_test.sample(n=min(sample_size, len(self.X_test)), random_state=42)
            else:
                X_sample = self.X_test
                
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-output models (binary classification often returns list of arrays)
            if isinstance(shap_values, list):
                # For binary classification, use positive class (index 1)
                self.shap_values[model_name] = shap_values[1]
            else:
                # For single output or already processed values
                if len(shap_values.shape) > 2:
                    # If 3D array, take the positive class
                    self.shap_values[model_name] = shap_values[:, :, 1]
                else:
                    self.shap_values[model_name] = shap_values
                
            print(f"SHAP values computed for {model_name} ({X_sample.shape[0]} samples)")
            print(f"SHAP values shape: {self.shap_values[model_name].shape}")
            
    def plot_shap_global_importance(self, model_name, max_features=20, save_path=None):
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not computed for {model_name}")
            
        shap_vals = self.shap_values[model_name]
        X_sample = self.X_test.iloc[:shap_vals.shape[0]]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_sample, feature_names=self.feature_names, 
                         max_display=max_features, show=False)
        plt.title(f'SHAP Global Feature Importance - {model_name.title()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_shap_summary(self, model_name, max_features=20, save_path=None):
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not computed for {model_name}")
            
        shap_vals = self.shap_values[model_name]
        X_sample = self.X_test.iloc[:shap_vals.shape[0]]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_sample, feature_names=self.feature_names, 
                         max_display=max_features, plot_type="bar", show=False)
        plt.title(f'SHAP Summary Plot - {model_name.title()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_shap_waterfall(self, model_name, sample_idx=0, save_path=None):
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not computed for {model_name}")
            
        shap_vals = self.shap_values[model_name]
        X_sample = self.X_test.iloc[:shap_vals.shape[0]]
        
        explainer = self.shap_explainers[model_name]
        
        if hasattr(explainer, 'expected_value'):
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = shap_vals.mean()
        
        # Handle multi-dimensional SHAP values (for binary classification)
        if len(shap_vals.shape) > 1 and shap_vals.shape[1] > 1:
            # For binary classification, use the positive class (index 1)
            sample_shap_vals = shap_vals[sample_idx, 1] if shap_vals.shape[1] == 2 else shap_vals[sample_idx, :]
            if isinstance(expected_value, np.ndarray) and len(expected_value) > 1:
                expected_value = expected_value[1]
        else:
            sample_shap_vals = shap_vals[sample_idx]
        
        shap_explanation = shap.Explanation(
            values=sample_shap_vals, 
            base_values=expected_value,
            data=X_sample.iloc[sample_idx].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - {model_name.title()} (Sample {sample_idx})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def initialize_lime_explainer(self):
        print("Initializing LIME explainer...")
        self.lime_explainer = LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['Bad Credit', 'Good Credit'],
            mode='classification',
            discretize_continuous=True
        )
        print("LIME explainer initialized")
        
    def explain_lime_instance(self, model_name, sample_idx=0, num_features=10, save_path=None):
        if self.lime_explainer is None:
            self.initialize_lime_explainer()
            
        model = self.models.get_model(model_name)
        instance = self.X_test.iloc[sample_idx].values
        
        explanation = self.lime_explainer.explain_instance(
            instance, 
            model.predict_proba, 
            num_features=num_features
        )
        
        if save_path:
            explanation.save_to_file(save_path)
        
        explanation.show_in_notebook(show_table=True)
        
        return explanation
        
    def compare_shap_across_groups(self, model_name, sensitive_attribute='age_group', max_features=15, save_path=None):
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not computed for {model_name}")
            
        if sensitive_attribute not in self.sensitive_test:
            raise ValueError(f"Sensitive attribute {sensitive_attribute} not found")
            
        shap_vals = self.shap_values[model_name]
        X_sample = self.X_test.iloc[:shap_vals.shape[0]]
        sensitive_vals = self.sensitive_test[sensitive_attribute].iloc[:shap_vals.shape[0]]
        
        groups = sensitive_vals.unique()
        
        fig, axes = plt.subplots(1, len(groups), figsize=(15, 6))
        if len(groups) == 1:
            axes = [axes]
            
        for i, group in enumerate(groups):
            group_mask = sensitive_vals == group
            group_shap = shap_vals[group_mask]
            group_X = X_sample[group_mask]
            
            mean_abs_shap = np.abs(group_shap).mean(axis=0)
            top_features = np.argsort(mean_abs_shap)[-max_features:]
            
            axes[i].barh(range(len(top_features)), mean_abs_shap[top_features])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels([self.feature_names[j] for j in top_features])
            axes[i].set_title(f'Group {group} (n={group_mask.sum()})')
            axes[i].set_xlabel('Mean |SHAP value|')
            
        plt.suptitle(f'SHAP Feature Importance by {sensitive_attribute} - {model_name.title()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_bias_with_shap(self, model_name, sensitive_attribute='age_group', top_features=10):
        if model_name not in self.shap_values:
            raise ValueError(f"SHAP values not computed for {model_name}")
            
        shap_vals = self.shap_values[model_name]
        X_sample = self.X_test.iloc[:shap_vals.shape[0]]
        sensitive_vals = self.sensitive_test[sensitive_attribute].iloc[:shap_vals.shape[0]]
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_feature_indices = np.argsort(mean_abs_shap)[-top_features:]
        
        bias_analysis = {}
        
        for group in sensitive_vals.unique():
            group_mask = sensitive_vals == group
            group_shap = shap_vals[group_mask]
            
            group_mean_shap = group_shap.mean(axis=0)
            
            bias_analysis[f'group_{group}'] = {
                'mean_shap': group_mean_shap[top_feature_indices],
                'feature_names': [self.feature_names[i] for i in top_feature_indices],
                'sample_size': group_mask.sum()
            }
            
        print(f"\nBias Analysis for {model_name} by {sensitive_attribute}:")
        print("="*60)
        
        for group, analysis in bias_analysis.items():
            print(f"\n{group.upper()} (n={analysis['sample_size']}):")
            for feat, shap_val in zip(analysis['feature_names'], analysis['mean_shap']):
                print(f"  {feat}: {shap_val:.4f}")
                
        return bias_analysis
        
    def save_explanations(self, save_dir=None):
        if save_dir is None:
            save_dir = config.EXPLANATIONS_DIR
        
        os.makedirs(str(save_dir), exist_ok=True)
        
        for model_name in self.shap_values.keys():
            model_dir = save_dir / model_name
            os.makedirs(str(model_dir), exist_ok=True)
            
            shap_values_path = model_dir / "shap_values.npy"
            np.save(str(shap_values_path), self.shap_values[model_name])
            
            joblib.dump(self.shap_explainers[model_name], 
                       str(model_dir / "shap_explainer.pkl"))
            
        if self.lime_explainer:
            joblib.dump(self.lime_explainer, str(save_dir / "lime_explainer.pkl"))
            
        print(f"Explanations saved to {save_dir}")

def generate_all_explanations(model_path=None, save_explanations=True):
    if model_path is None:
        model_path = config.MODELS_DIR
    explainer = CreditExplainer()
    explainer.load_models_and_data(model_path)
    explainer.initialize_shap_explainers()
    explainer.compute_shap_values(sample_size=500)
    explainer.initialize_lime_explainer()
    
    model_names = list(explainer.models.models.keys())
    
    for model_name in model_names:
        print(f"\nGenerating explanations for {model_name}...")
        
        explainer.plot_shap_global_importance(model_name)
        explainer.plot_shap_summary(model_name)
        explainer.plot_shap_waterfall(model_name, sample_idx=0)
        
        if 'age_group' in explainer.sensitive_test:
            explainer.compare_shap_across_groups(model_name, 'age_group')
            explainer.analyze_bias_with_shap(model_name, 'age_group')
            
        explainer.explain_lime_instance(model_name, sample_idx=0)
        
    if save_explanations:
        explainer.save_explanations()
        
    return explainer

if __name__ == "__main__":
    explainer = generate_all_explanations()
    print("\nExplainability analysis completed!")