# Bank Marketing XAI App

# Load Libraries
from PIL import Image
from io import BytesIO
import requests
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score,ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

from sklearn.preprocessing import LabelEncoder

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


# --------------------------------------------------------------

# Define app layout
# st.set_page_config(page_title='Model Evaluation', page_icon='🔎')
st.title('Model Evaluation')



# Show chosen Hyperparameters
st.header("Hyperparameters")
st.markdown("""
Defining values for hyperparameters is important as it defines how the model will 
learn from the data. Our hyperparameters are evaluated using stratified k-fold cross-validation
involves splitting the dataset into k subsets, where each subset contains roughly the same 
proportion of target classes as the whole dataset, and training the model on k-1 subsets while 
using the remaining subset for testing.

Another important hyperparamter is scale_pos_weight, as it helps the model to learn the patterns
of the minority class better and improve its ability to handle imbalanced data sets.

Finally, we proceed by using the following values for training the XGBoost classification model:
""")

hyperparameters = {'scale_pos_weight': [6.5], 
                   'eval_metric': ["auc"],
                   'learning_rate': [0.3],
                   'max_depth': [5],
                   'n_estimators': [40]}

hyperparameters_df = pd.DataFrame.from_dict(hyperparameters)
st.table(hyperparameters_df)






# Display Confusion Matrix
st.header("Confusion Matrix")
st.markdown("""
A confusion matrix is a table that visualizes the performance of a machine learning
model by comparing the predicted and actual values of a classification problem.
It contains information about the performance of a classification model using:

- True positives (correctly classified as 1)
- True negatives (correctly classified as 0)
- False positives (mistakenly classified as 1)
- False negatives (mistakenly classified as 0)

The matrix provides the actual number of of true positives, false positives, true negatives, and false negatives - which 
are often used to calculate further metrics such as precision, recall and F1-score. 
""")
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm).plot(values_format = "", cmap = "YlGnBu")
plt.title("Confusion Matrix")
plt.text(-0.4, 0.2, "correctly classified as 0", fontsize = 9, color = "w")
plt.text(-0.43, 1.2, "mistakenly classified as 0", fontsize = 9) #, color = "b")
plt.text(0.56, 0.2, "mistakenly classified as 1", fontsize = 9) #, color = "b")
plt.text(0.6, 1.2, "correctly classified as 1", fontsize = 9) #, color = "b")
plt.grid(False)
plt.show()
st.pyplot()
plt.rcdefaults()


# Display the SHAP values for the example
st.header('ROC (Receiver Operating Characteristic) Curve')
st.markdown("""
The ROC curve is created by plotting the true positive rate (y-axis) against the false positive rate (x-axis).
The TPR is the ratio of true positives to the total number of actual positive cases, and the FPR is the ratio 
of false positives to the total number of actual negative cases.

The area under the ROC curve (AUC) is a commonly used metric to evaluate the overall performance of a binary 
classification model. AUC ranges from 0 to 1, where a model with an AUC of 1 is considered perfect, while a model 
with an AUC of 0.5 performs no better than random guessing.
Our AUC of 0.93 indicates that the XGBoost model performs well distinguishing between positive and negative classes.
""")

# Predict the probabilities of the positive class
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)

# Get FPR, TPR and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
st.pyplot()
plt.rcdefaults()









# Show decision tree stump
st.subheader("Visualizing a Decision Tree")

# Define smaller XGBoost model

st.markdown('''
XGBoost is an ensemble method and consists of multiple decision trees. In order to visualize one, a smaller decision tree is created with depth = 2.
The ellipses show the feature and the threshold used for splitting the data. These are connected to the next node using arrows. 
''')

dtc = DecisionTreeClassifier(max_depth = 2)
dtc.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(dtc, 
          max_depth = 2,
          filled=True, 
          feature_names=X_train.columns,
          impurity = False,
          class_names=['Not Interested', 'Interested'], 
          precision=2)
plt.show()
st.pyplot()


st.markdown('''
Finally, the single decision trees are combined using boosting.
In the figure, they are referred to as 'Classifier'.
''')

response = requests.get("https://raw.githubusercontent.com/majimaken/XAI-Classification/main/Boosting.png")
img = Image.open(BytesIO(response.content))

# Display the image using Streamlit
st.image(img, caption="Boosting")






