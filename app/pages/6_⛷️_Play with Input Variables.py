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

# Cache: Load dfa
df = load_data()

# Cache: Split and preprocess dfa
X_train, X_test, y_train, y_test = prepare_data(df)

# Cache: Fit model
xgb_model = cached_xgb_model(X_train, y_train)

# Cache: Predict
y_pred = predict_with_cached_model(xgb_model, X_test)

# -----------------------------------------------------------------------

# Define app layout
# st.set_page_config(page_title='Classification of New dfa', page_icon='⛷️')
st.title("Play With Input Variables")

st.header("Prediction Tool")
st.markdown('''
The exploratory data analysis and XAI methods provide us with an overview of the data and the model from different angles. 
In the prediction tool, different values can be configured for features in order to make a prediction.  
This helps us get an even better feel for the model. 
''')

# Sidebar with Inputs
st.sidebar.header("Input for Classification")
age = st.sidebar.slider("Age", int(18), 
                                int(100), 
                                int(18), key="1")
job = st.sidebar.selectbox("Job", list(set(df["job"])), key="2")
marital = st.sidebar.selectbox("Marital status", list(set(df["marital"])), key="3")
education = st.sidebar.selectbox("Education", list(set(df["education"])), key="4")
default = st.sidebar.selectbox("Has credit in default (Verzug)", list(set(df["default"])), key="5")
balance = st.sidebar.slider("Balance of bank account", int(-10000), 
                                int(100000), 
                                int(0), key="6")
housing = st.sidebar.selectbox("Has housing mortgage", list(set(df["housing"])), key="7")
loan = st.sidebar.selectbox("Has personal loan", list(set(df["loan"])), key="8")
contact = st.sidebar.selectbox("Contact communication type", list(set(df["contact"])), key="9")
day = st.sidebar.selectbox("Weekday of last contact", list(set(df["day"])), key="10")
month = st.sidebar.selectbox("Month of last contact", list(set(df["month"])), key="11")
duration = st.sidebar.slider("Duration of last contact in seconds", int(df["duration"].min()), 
                                int(5000), 
                                int(0), key="12")
campaign = st.sidebar.slider("Number of contacts during this campaign", int(df["campaign"].min()), 
                                int(70), 
                                int(0), key = "13")           
pdays = st.sidebar.slider("Days since client was last contacted", int(df["pdays"].min()), 
                                int(1000), 
                                int(0), key = "14")    
previous = st.sidebar.slider("Number of contacts before this campaign", int(df["previous"].min()), 
                                int(100), 
                                int(0), key = "15")                                  
poutcome = st.sidebar.selectbox("Outcome of the previous marketing campaign", list(set(df["poutcome"])), key = "16")

# Build new dataframe for prediction
new_data = [age, job, marital, education, default, balance, housing, loan, 
        contact, day, month, duration, campaign, pdays, previous, poutcome]

new_df = pd.DataFrame([new_data],
                      columns = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", 
                                "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"])
                                
st.subheader("Data Used for Classification")
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(data = new_df)

# Workaround LabelEncoder()
# Append new_df to X_test in order to get LabelEncoder() to work
new_df_append = df.append(new_df, ignore_index = True)

# Encode columns with object / categorical variables
# 1) Find all columns with categorical / object and assign to list
categorical_cols = new_df_append.select_dtypes(include="object").columns.tolist()    

# 2) Create df_encoded
new_df_encoded = new_df_append.copy()

# 3) Create new dataframe with preprocessed categorical variables
encoder = LabelEncoder()

for col in categorical_cols:
    new_df_encoded[col] = encoder.fit_transform(new_df_append[col])
    
# 4) Convert all yes/no to binary
new_df_encoded.replace({'yes': 1, 'no': 0}, inplace=True)      

# 5) Select last row of new_df_encoded and drop y-column
new_df_encoded = new_df_encoded.tail(1)
new_df_encoded = new_df_encoded.drop("y", axis = 1)

# Show Encoded dfaset
st.subheader("Preprocessed Data Used for Classification")
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.table(data = new_df_encoded) 

# # Make prediction
# example = pd.dfaFrame(inputs, index=[0])
prediction = xgb_model.predict(new_df_encoded)
probability = xgb_model.predict_proba(new_df_encoded)

# # # Display prediction and probability
st.header('Interested in term deposit?')

if prediction == 0:
    st.write('<span style="color: red;">NOT</span> interested in term deposit with a probability of ', 
         "{:.2%}".format(probability[0][0]),
         '! 😞',
         unsafe_allow_html=True)



else:
    #st.write('Interested in term deposit with a probability of: {:.2f}'.format(probability[0][1]))
    st.write('<span style="color: green;">INTERESTED</span> in term deposit with a probability of ', 
         "{:.2%}".format(probability[0][1]), 
         '! 😄',
         unsafe_allow_html=True)

prob_df = pd.DataFrame({'Probability': probability[0]}, index=['NOT interested', 'Interested'])
st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))






