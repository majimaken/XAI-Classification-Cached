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



st.set_page_config(
	page_title = "Overview",
	page_icon = "🤖"
	)

st.sidebar.success("Please select a page.") 

st.title("XGBoost Classification App")

st.subheader("My Motivation for Developing this App")
st.markdown("""
I created this app using Streamlit that uses a machine learning technique called XGBoost 
to classify data. XGBoost is more complex than other techniques, which can make it difficult 
for people to understand.

I made this app as part of my specialization project of the Master of Science in Engineering, 
and it allows users to explore XGBoost and see how it makes predictions. By visualizing the data set and
and feature importances, users can gain insight into how the model works and what factors influence its predictions.

Ultimately, my goal is to help more people understand machine learning techniques like XGBoost and make them 
more accessible to a wider audience.
""")

st.subheader("Why you should be using this App")
st.markdown("""
This app provides an overview of the XGBoost algorithm used for a classification problem. 
The model is trained using the Bank Marketing Dataset from UCI,
which is an imbalanced dataset containing client information and whether a customer subscribed to a term depot or not. 

Please play around with different input variables and see how the model behaves! 😊
""")

st.subheader("Links")
st.markdown("""
Data souce: https://archive.ics.uci.edu/ml/datasets/bank+marketing

Source code: https://github.com/majimaken/XAI-Classification-Cached
""")











