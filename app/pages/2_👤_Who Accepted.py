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

# --------------------------------------------------------------

# Define app layout
# st.set_page_config(page_title='Exploratory Analysis', page_icon='🏔️')
st.title('Who Accepted the Term Deposit Deal?')

st.header("Analyzing Categorical Features")
st.markdown("""
The categorical variables invite the presentation of frequencies grouped by outcome. 
Thus, the distribution for both outcomes can be visualized and compared.  

The direct comparisons within the same class are particularly interesting. In this way, 
it can be estimated in which categories interested parties appear relatively equally often. 
""")


# Filter to only include categorical columns (excluding 'y')
cat_cols = [col for col in df.select_dtypes(include=['object']).columns.tolist() if col != 'y']

# Let the user select a categorical feature to visualize
selected_cat_feature = st.selectbox('Select a categorical feature to visualize:', cat_cols)


# Group the data by the selected feature and the binary predictor,
# and count the frequencies
grouped_df = df.groupby([selected_cat_feature, 'y']).size().reset_index(name='count')

# Create a barplot using seaborn
sns.set_style('whitegrid')
sns.barplot(data=grouped_df, x=selected_cat_feature, y='count', hue='y')

# Add labels and title to the plot
plt.xlabel(selected_cat_feature)
plt.ylabel('Count')
plt.title(f'Frequencies of {selected_cat_feature} by Binary Predictor')
plt.legend(title='Subscribed to term deposit?', loc='upper right')

# Rotate x-axis labels vertically
plt.xticks(rotation=90)

# Show the plot
plt.show()
st.pyplot()
plt.rcdefaults()

# -------------------------------------------------------------------

# st.subheader("Analyzing Numerical Features")

# # Dropdown menu for numerical features
# # Filter to only include numerical columns
# num_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()

# # Let the user select a numerical feature to visualize
# selected_feature = st.selectbox('Select a numerical feature to visualize:', num_cols)

# # Get the grouped dataframe using st.cache
# grouped_df = get_grouped_df(df, selected_feature)

# # Create a barplot using seaborn
# sns.set_style('whitegrid')
# sns.barplot(data=grouped_df, x=selected_feature, y='count', hue='y')

# # Add labels and title to the plot
# plt.xlabel(selected_feature)
# plt.ylabel('Count')
# plt.title(f'Frequencies of {selected_feature} by Binary Predictor')
# plt.legend(title='Subscribed to term deposit?', loc='upper right') #, labels=['no', 'yes'])

# # Rotate x-axis labels vertically
# plt.xticks(rotation=90)

# # Show the plot
# plt.show()
# st.pyplot()

# --------------------------------------------------------------------

# st.header("Who Accepted the Last Campaign?")

# for _ in df.columns:
    # if _=='y':
        # pass
    # else:
        # fig = plt.figure(figsize = (10, 5))
        # uniqe_list=[i for i in df[df['y']=="yes"][f'{_}'].unique()]
        # uniqe_values=[len(df[df['y']=="yes"][f'{_}'][df[df['y']=="yes"][f'{_}']==i]) 
        # for i in df[df['y']=="yes"][f'{_}'].unique()]

        # plt.bar(uniqe_list, uniqe_values) ##, width = 0.4)
        # plt.xlabel(f"{_} values of who subscribed to a term deposit",fontsize=10)
        # plt.xticks(rotation=45, ha='right')
        # plt.ylabel("Frequency of " f"{_}",fontsize=10)
        # plt.title("Histogram of Who Subscribed to a Term Deposit",fontsize=15);
        # st.pyplot()
