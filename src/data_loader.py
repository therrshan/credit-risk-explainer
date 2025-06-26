import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class GermanCreditLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = str(config.RAW_DATA_DIR)
        else:
            self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.sensitive_attributes = ['sex', 'age']
        
    def load_raw_data(self):
        print("Loading German Credit dataset...")
        data = fetch_openml("credit-g", version=1, as_frame=True)
        df = data.frame.copy()
        
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        df.to_csv(config.RAW_DATA_DIR / "german_credit_raw.csv", index=False)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:\n{df['class'].value_counts()}")
        print(f"\nFeature types:\n{df.dtypes.value_counts()}")
        
        return df
    
    def preprocess_data(self, df):
        df_processed = df.copy()
        
        target_col = "class"
        df_processed[target_col] = df_processed[target_col].map({"bad": 0, "good": 1})
        
        sensitive_attrs = {}
        
        if 'age' in df_processed.columns:
            age_median = df_processed['age'].median()
            sensitive_attrs['age_group'] = (df_processed['age'] > age_median).astype(int)
        
        sex_columns = [col for col in df_processed.columns if 'sex' in col.lower() or 'personal_status' in col.lower()]
        if sex_columns:
            sensitive_attrs['gender_derived'] = df_processed[sex_columns[0]]
        
        cat_cols = df_processed.select_dtypes(include=["category", "object"]).columns.tolist()
        if target_col in cat_cols:
            cat_cols.remove(target_col)
        
        num_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        
        print(f"Categorical columns: {cat_cols}")
        print(f"Numerical columns: {num_cols}")
        
        for col in cat_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        self.feature_names = [col for col in df_processed.columns if col != target_col]
        
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        for attr_name, attr_values in sensitive_attrs.items():
            X[attr_name] = attr_values
        
        return X, y, sensitive_attrs
    
    def load_and_split(self, test_size=0.2, random_state=42, scale_features=False):
        raw_data_path = config.RAW_DATA_DIR / "german_credit_raw.csv"
        if raw_data_path.exists():
            print("Loading existing raw data...")
            df = pd.read_csv(raw_data_path)
        else:
            print("Raw data not found, downloading...")
            df = self.load_raw_data()
        
        X, y, sensitive_attrs = self.preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        sensitive_train = {}
        sensitive_test = {}
        for attr_name in sensitive_attrs.keys():
            if attr_name in X_train.columns:
                sensitive_train[attr_name] = X_train[attr_name].copy()
                sensitive_test[attr_name] = X_test[attr_name].copy()
                X_train = X_train.drop(columns=[attr_name])
                X_test = X_test.drop(columns=[attr_name])
        
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        self.feature_names = X_train.columns.tolist()
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test
    
    def get_feature_names(self):
        return self.feature_names
    
    def get_label_encoders(self):
        return self.label_encoders
    
    def save_preprocessors(self, path=None):
        if path is None:
            path = config.PREPROCESSORS_DIR
        
        os.makedirs(path, exist_ok=True)
        
        for col_name, encoder in self.label_encoders.items():
            joblib.dump(encoder, path / f"le_{col_name}.pkl")
        
        if hasattr(self.scaler, 'mean_'):
            joblib.dump(self.scaler, path / "scaler.pkl")
        
        print(f"Preprocessors saved to {path}")

def load_german_credit(test_size=0.2, random_state=42, scale_features=False):
    loader = GermanCreditLoader()
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = loader.load_and_split(
        test_size=test_size, 
        random_state=random_state, 
        scale_features=scale_features
    )
    
    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, loader

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test, loader = load_german_credit()
    
    print("\nSensitive attributes for fairness analysis:")
    for attr_name, attr_values in sensitive_train.items():
        print(f"{attr_name}: {attr_values.value_counts().to_dict()}")
    
    print(f"\nFeature names: {loader.get_feature_names()}")