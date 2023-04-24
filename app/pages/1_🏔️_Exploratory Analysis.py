# Bank Marketing XAI App

# Load Libraries
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from CachedFunctions import load_data, prepare_data, cached_xgb_model, predict_with_cached_model

# Disable Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Cache: Load data
df = load_data()

# Cache: Split and preprocess data
X_train, X_test, y_train, y_test = prepare_data(df)

# Cache: Fit model
xgb_model = cached_xgb_model(X_train, y_train)

# --------------------------------------------------------------

# Define app layout
# st.set_page_config(page_title='Exploratory Analysis', page_icon='🏔️')
st.title('Exploratory Analysis of Bank Marketing Dataset')

# Show Data Frame
st.dataframe(data = df)

st.header("Variables")
st.markdown("""
1.  age:            age of person 
    - numeric
2.  job:           type of job 
    - categorical: ('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3.  marital:        marital status 
    - categorical: ('divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4.  education:
    - categorical: ('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5.  default: has credit in default? 
    - categorical: ('no','yes','unknown')
6.  housing:        has housing loan? 
    - categorical: ('no','yes','unknown')
7.  loan: has personal loan? 
    - categorical: ('no','yes','unknown')
8.  contact: contact communication type 
    - categorical: ('cellular','telephone')
9.  month: last contact month of year 
    - categorical: ('jan', 'feb', 'mar', ..., 'nov', 'dec')
10. day_of_week: last contact day of the week 
    - categorical: ('mon','tue','wed','thu','fri')
11. duration: last contact duration, in seconds 
    - numeric 
12. campaign: number of contacts performed during this campaign and for this client 
    - numeric
13. pdays: number of days that passed by after the client was last contacted from a previous campaign 
    - numeric (999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client 
    - numeric
15. poutcome: outcome of the previous marketing campaign 
    - categorical: ('failure','nonexistent','success')
16. y (output variable): has the client subscribed a term deposit? 
    - binary: ('yes','no')
""")

st.header("Overview of Data Set")
pr = ProfileReport(df, 
                   explorative = True,
                   minimal = True,
                   )
st_profile_report(pr)


st.header("Correlation Matrix")
plt.figure(figsize=(15,6))
sns.heatmap(df.corr(method = "spearman"), 
            annot=True, 
            cmap = "YlGnBu") #, cmap = "YlGnBu")
plt.title("Correlation Matrix", fontsize = 24)
plt.show()
st.pyplot()



st.header("Imbalanced Data")
st.write(df["y"].value_counts())

plt.figure(figsize = (10,5))
value_counts = df["y"].value_counts()
value_counts.plot.bar()
plt.title("Value Counts of y")
plt.xlabel("y")
plt.ylabel("Count")
plt.show()
st.pyplot()


st.header("Who Accepted the Last Campaign?")

for _ in df.columns:
    if _=='y':
        pass
    else:
        fig = plt.figure(figsize = (10, 5))
        uniqe_list=[i for i in df[df['y']=="yes"][f'{_}'].unique()]
        uniqe_values=[len(df[df['y']=="yes"][f'{_}'][df[df['y']=="yes"][f'{_}']==i]) 
        for i in df[df['y']=="yes"][f'{_}'].unique()]

        plt.bar(uniqe_list, uniqe_values) ##, width = 0.4)
        plt.xlabel(f"{_} values of who subscribed to a term deposit",fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Frequency of " f"{_}",fontsize=10)
        plt.title("Histogram of Who Subscribed to a Term Deposit",fontsize=15);
        st.pyplot()






