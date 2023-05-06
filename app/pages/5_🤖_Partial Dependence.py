# Bank Marketing XAI App

# Load Libraries
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

from sklearn.preprocessing import LabelEncoder

import shap

from CachedFunctions import load_data, prepare_data, cached_xgb_model, predict_with_cached_model

# Disable Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Cache: Load data
df = load_data()

# Cache: Split and preprocess data
X_train, X_test, y_train, y_test = prepare_data(df)

# Cache: Fit model
xgb_model = cached_xgb_model(X_train, y_train)

# Cache: Predict
y_pred = predict_with_cached_model(xgb_model, X_test)

# -----------------------------------------------------------------------

# Define app layout
# st.set_page_config(page_title='Partial Dependence', page_icon='🤖')
st.title('Partial Dependence')

st.header("Analyzing Model Behavior with Partial Dependence Plots")
st.markdown(
'''
Partial dependence plots (PDP) are a useful tool for visualizing the relationship between a feature and the target variable in a machine learning model. 
They show how the predicted outcome changes as the feature value varies, while holding all other features constant.
A PDP is created by first selecting a feature of interest, then generating a series of test cases where that feature is varied while all other features 
are held constant. For each test case, the predicted outcome is recorded, and the average outcome is computed for each unique value of the feature.
The resulting plot shows how the predicted outcome varies as the feature value changes. This can help to identify non-linear relationships between the 
feature and the target variable, as well as interactions between multiple features.

How to read the plot:
- y-axis shows the probability of the positive class (here: investment in term deposit).
- x-axis shows the range of a feature.
- If the line is flat, it indicates that the feature has little to no effect on the predicted outcome.
- A steep slope indicates a strong effect. 
'''
)

options = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

# Dropdown menu
selected_feature = st.selectbox('Select numerical feature to visualize:', options) # X_train.columns)

# Calculate partial dependence values for the selected feature
feature_index = X_train.columns.get_loc(selected_feature)

shap.partial_dependence_plot(
    ind = feature_index, 
    model = xgb_model.predict, 
    data = X_train,
    ice=False,
    model_expected_value=True, feature_expected_value=True
)
st.pyplot()
plt.rcdefaults()