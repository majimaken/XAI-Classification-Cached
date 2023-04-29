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

from CachedFunctions import load_data, prepare_data, cached_xgb_model, predict_with_cached_model, get_shap_summary_plot

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
# st.set_page_config(page_title='Feature Importance', page_icon='🔝')
st.title('Feature Importance')


st.header("Explaining XGBoost")

st.markdown("""
There are several ways to do XAI (Explainable Artificial Intelligence) with XGBoost, some of the most common methods include feature importance and SHAP (SHapley Additive exPlanations) values.

Feature importance measures the contribution of each feature in the model's accuracy, while Shapley values provide a way to explain the output of any machine 
learning model by measuring the impact of each feature on the prediction for a specific instance. By using both feature importance and Shap values, we can gain 
a better understanding of how the XGBoost model makes predictions and which features are most important in the decision-making process.
""")



# Display the SHAP values for the example
st.subheader('Feature Importance')
st.markdown('''
This method involves calculating the relative importance of each input feature in the XGBoost model. 
In our XGBoost model, feature importance is calculated based on the frequency and depth of the splits of each feature across all decision trees in the ensemble. 
The more a feature is used in the decision trees, the higher its importance score.

How to read the plot:
- The most important features are listed at the top.
- When feature importance is high, it means that the feature has a strong influence on the model's predictions.

''')


# Feature Importances
importances = xgb_model.feature_importances_
feature_names = X_train.columns

# Create a dataframe of feature importances
df_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort the dataframe by importance
df_importances = df_importances.sort_values('importance', ascending=True)

# Create a horizontal bar chart of feature importances
plt.rcdefaults()
plt.barh(df_importances['feature'], df_importances['importance']) #, color = "navy")
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
st.pyplot(plt)
plt.rcdefaults()

st.markdown('''
Explaining 'contact' using feature imporatance:
- The feature 'contact' has the highest feature importance.
- This suggests that the feature provides great information gain for the model and therefore is often used in the individual decision trees of the XGBoost model. 
- Feature importance does not tell us if 'contact' has a positive or negative influence on the prediction. 
''')




st.subheader("SHAP (SHapley Additive exPlanations) Values")
st.markdown('''
SHAP is a popular XAI technique that provides an explanation for each prediction of the model. SHAP values represent the contribution of each feature 
to the predicted outcome. These values can be used to understand how the model arrived at its prediction and to identify features that have the 
greatest impact on the model's output.

How to read the plot:
- The most important features are listed at the top. 
- The vertical line on the x-axis represent the average prediction of the model. The direction and length of the bar indicate the direction and magnitude of the feature's effect on the prediction. 
- When a feature has both high negative and positive SHAP (SHapley Additive exPlanations) values, it means that the feature can have both positive and negative effects on the model's prediction. 
''')

# st.header("Feature Importance")
# plt.title("Feature Importance based on SHAP")
# shap.summary_plot(shap_values, X_train)
# st.pyplot(bbox_inches = "tight")
# st.write("----")
explainer = shap.TreeExplainer(xgb_model)
fig = get_shap_summary_plot(explainer, X_train)
fig.suptitle("SHAP Values for XGBoost Model Predictions on the Training Set", fontsize=14)
fig.subplots_adjust(top=0.9) # adjust top spacing
st.pyplot(fig, bbox_inches="tight")
st.write("----")
plt.rcdefaults()

st.markdown('''
Explaining 'duration' using SHAP:
- The longer the last contact, the more likely it is that a term deposit will be made. 
- However, a customer with no contact probably has no reference to the product. 
- Therefore, the feature can have both great positive and negative effects on the model's prediction.
''')

st.header("Our Thoughts")
st.markdown('''
The feature importance plot suggests the features such as contact, poutcome, duration and housing as influential. 
Surprisingly, the summary of the SHAP values indicate that duration, contact, month and pdays are among the more impactful features.
We need to remember that they use different methods to calculate the importance of the feature. 
Feature importance is based on the reduction in impurity of decision trees in the ensemble, while SHAP values are based on the contribution 
of each feature to the prediction of each individual data point.
''')








