import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_german_credit

class CreditRiskModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.feature_names = None
        
    def load_data(self, test_size=0.2, scale_features=False):
        self.X_train, self.X_test, self.y_train, self.y_test, self.sensitive_train, self.sensitive_test, self.loader = load_german_credit(
            test_size=test_size, 
            random_state=self.random_state,
            scale_features=scale_features
        )
        self.feature_names = self.loader.get_feature_names()
        print(f"Data loaded: {self.X_train.shape[0]} train samples, {self.X_test.shape[0]} test samples")
        
    def train_xgboost(self, hyperparameter_tuning=False):
        print("Training XGBoost model...")
        
        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            self.models['xgboost'] = grid_search.best_estimator_
            print(f"Best XGBoost parameters: {grid_search.best_params_}")
        else:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )
            self.models['xgboost'].fit(self.X_train, self.y_train)
        
        self._evaluate_model('xgboost')
        
    def train_random_forest(self, hyperparameter_tuning=False):
        print("Training Random Forest model...")
        
        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf_model = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            self.models['random_forest'] = grid_search.best_estimator_
            print(f"Best Random Forest parameters: {grid_search.best_params_}")
        else:
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            )
            self.models['random_forest'].fit(self.X_train, self.y_train)
        
        self._evaluate_model('random_forest')
        
    def train_logistic_regression(self):
        print("Training Logistic Regression model...")
        
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.models['logistic_regression'].fit(self.X_train, self.y_train)
        
        self._evaluate_model('logistic_regression')
        
    def _evaluate_model(self, model_name):
        model = self.models[model_name]
        
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        
        train_proba = model.predict_proba(self.X_train)[:, 1]
        test_proba = model.predict_proba(self.X_test)[:, 1]
        
        scores = {
            'train_accuracy': accuracy_score(self.y_train, train_pred),
            'test_accuracy': accuracy_score(self.y_test, test_pred),
            'train_auc': roc_auc_score(self.y_train, train_proba),
            'test_auc': roc_auc_score(self.y_test, test_proba),
            'test_precision': precision_score(self.y_test, test_pred),
            'test_recall': recall_score(self.y_test, test_pred),
            'test_f1': f1_score(self.y_test, test_pred),
            'confusion_matrix': confusion_matrix(self.y_test, test_pred).tolist(),
            'classification_report': classification_report(self.y_test, test_pred, output_dict=True)
        }
        
        fpr, tpr, _ = roc_curve(self.y_test, test_proba)
        precision, recall, _ = precision_recall_curve(self.y_test, test_proba)
        
        scores['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        scores['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        scores['cv_auc_mean'] = cv_scores.mean()
        scores['cv_auc_std'] = cv_scores.std()
        scores['cv_scores'] = cv_scores.tolist()
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            scores['feature_importance'] = feature_importance
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(self.feature_names, abs(model.coef_[0])))
            scores['feature_importance'] = feature_importance
        
        self.model_scores[model_name] = scores
        
        print(f"\n{model_name.upper()} Results:")
        print(f"Train Accuracy: {scores['train_accuracy']:.4f}")
        print(f"Test Accuracy: {scores['test_accuracy']:.4f}")
        print(f"Train AUC: {scores['train_auc']:.4f}")
        print(f"Test AUC: {scores['test_auc']:.4f}")
        print(f"Precision: {scores['test_precision']:.4f}")
        print(f"Recall: {scores['test_recall']:.4f}")
        print(f"F1-Score: {scores['test_f1']:.4f}")
        print(f"CV AUC: {scores['cv_auc_mean']:.4f} (+/- {scores['cv_auc_std']*2:.4f})")
        
        print(f"\nClassification Report ({model_name}):")
        print(classification_report(self.y_test, test_pred))
        
    def train_all_models(self, hyperparameter_tuning=False):
        if not hasattr(self, 'X_train'):
            self.load_data()
            
        self.train_xgboost(hyperparameter_tuning)
        self.train_random_forest(hyperparameter_tuning)
        self.train_logistic_regression()
        
        self.compare_models()
        
    def compare_models(self):
        if not self.model_scores:
            print("No models trained yet!")
            return
            
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame(self.model_scores).T
        comparison_df = comparison_df.round(4)
        print(comparison_df[['test_accuracy', 'test_auc', 'test_precision', 'test_recall', 'test_f1', 'cv_auc_mean', 'cv_auc_std']])
        
        best_model_name = comparison_df['test_auc'].idxmax()
        print(f"\nBest model by Test AUC: {best_model_name}")
        
    def get_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        return self.models[model_name]
    
    def get_feature_names(self):
        return self.feature_names
    
    def get_data(self):
        if not hasattr(self, 'X_train'):
            self.load_data()
        return self.X_train, self.X_test, self.y_train, self.y_test, self.sensitive_train, self.sensitive_test
    
    def save_models(self, path=None):
        if path is None:
            path = config.MODELS_DIR
        
        os.makedirs(str(path), exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = path / f"{model_name}.pkl"
            joblib.dump(model, str(model_path))
            print(f"Saved {model_name} to {model_path}")
            
        scores_path = path / "model_scores.pkl"
        joblib.dump(self.model_scores, str(scores_path))
        
        metrics_df = pd.DataFrame(self.model_scores).T
        metrics_csv_path = path / "model_metrics.csv"
        metrics_df.to_csv(str(metrics_csv_path))
        
        print(f"Saved model scores to {scores_path}")
        print(f"Saved metrics CSV to {metrics_csv_path}")
        
    def load_models(self, path=None):
        if path is None:
            path = config.MODELS_DIR
        model_files = {
            'xgboost': 'xgboost.pkl',
            'random_forest': 'random_forest.pkl',
            'logistic_regression': 'logistic_regression.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = path / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(str(model_path))
                print(f"Loaded {model_name} from {model_path}")
                
        scores_path = path / "model_scores.pkl"
        if scores_path.exists():
            self.model_scores = joblib.load(str(scores_path))
            print(f"Loaded model scores from {scores_path}")

def train_baseline_models(hyperparameter_tuning=False, save_models=True):
    credit_models = CreditRiskModels()
    credit_models.load_data()
    credit_models.train_all_models(hyperparameter_tuning=hyperparameter_tuning)
    
    if save_models:
        credit_models.save_models()
    
    return credit_models

if __name__ == "__main__":
    models = train_baseline_models(hyperparameter_tuning=False, save_models=True)
    
    print("\nTraining completed!")
    print("Models saved and ready for explainability analysis.")