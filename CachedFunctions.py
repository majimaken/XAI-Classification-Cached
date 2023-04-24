import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score,ConfusionMatrixDisplay

import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

from sklearn.preprocessing import LabelEncoder

# Preparing Cached Model

# Load Data
@st.cache_data
def load_data(url = "https://raw.githubusercontent.com/majimaken/XAI-Classification/main/bank-full.csv"):
    df = pd.read_csv(url, sep = ";")
    return df
   
# df = load_data()

# Function for Preprocessing Dataset
def convert_categorical_to_numerical(df):
    """
    Convert categorical variables into numerical variables using LabelEncoder from scikit-learn
    
    Parameters:
    -----------
    df : pandas dataframe
        The dataframe containing the categorical variables
    
    Returns:
    --------
    df : pandas dataframe
        The dataframe with converted categorical variables
    
    label_encoders : dict
        Dictionary containing the LabelEncoder object for each categorical column
    """
    
    # Convert all yes/no to binary
    df.replace({'yes': 1, 'no': 0}, inplace=True)
    
    # Find all columns with categorical / object and assign to list
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# Cache: Prepare and splitting data
@st.cache_data
def prepare_data(df):
   
    # Preprocess the dataset
    df_preprocessed, label_enc = convert_categorical_to_numerical(df)
    
    # Split the dataset into training and testing sets
    features = df_preprocessed.drop("y", axis = 1).columns.values
    X = df_preprocessed[features]
    y = df_preprocessed["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state= 2023)
    
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = prepare_data(df)

# Cache Model
@st.cache_resource
def cached_xgb_model(X_train, y_train):
    xgb_model = XGBClassifier(scale_pos_weight = 6.5, 
                    eval_metric = "auc",
                    learning_rate = 0.3,
                    max_depth = 5,
                    n_estimators = 40)
    xgb_model.fit(X_train, y_train)
    
    # Return the trained classifier
    return xgb_model

# xgb_model = cached_xgb_model(X_train, y_train)


# Cache Predictions
@st.cache_resource
def predict_with_cached_model(_xgb_model, X_test):
    # Generate predicted values with xgb_model.predict
    y_pred = _xgb_model.predict(X_test)
    return y_pred
    
# y_pred = predict_with_cached_model(xgb_model, X_test)
