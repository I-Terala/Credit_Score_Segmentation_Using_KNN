# -*- coding: utf-8 -*-
"""Credit_Score_Segmentation_KNN.ipynb


Original file is located at
    https://colab.research.google.com/drive/1sb9tFTVKFAtDrksdYjA2SGD9l1WVzFxI
"""

# import necessary libraries
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_white'

# Read CSV file
data = pd.read_csv('credit_scoring.csv')
data.head(5)

# Check data info
data.info()

# Check for missing Values
data.isnull().sum()

# Check data types
data.dtypes

# Descriptive Stats f data
descriptive_stats = data.describe()
print(descriptive_stats)

# Create a boxplot to see credit utilization ratio.
credit_utilization_fig = px.box(data, y='Credit Utilization Ratio', title = 'Credit Utilization Ratio Distribution')
credit_utilization_fig.show()

# Get distribution of data
loan_amount_fig = px.histogram(data, x= 'Loan Amount', nbins=20, title = 'Loan Amount Distribution')
loan_amount_fig.show()

# Create Correlation mattrix for all the numeric features.
numeric_df = data[['Credit Utilization Ratio', 'Payment History', 'Number of Credit Accounts', 'Loan Amount', 'Interest Rate', 'Loan Term']]

correlation_matrix_fig = px.imshow(numeric_df.corr(), title = 'Correlation Matrix')
correlation_matrix_fig.show()

# Define the mapping for categorical features
education_level_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
employment_status_mapping = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}

# Apply mapping to categorical features
data['Education Level'] = data['Education Level'].map(education_level_mapping)
data['Employment Status'] = data['Employment Status'].map(employment_status_mapping)

# Calculate credit scores using the complete FICO formula
credit_scores = []

for index, row in data.iterrows():
    payment_history = row['Payment History']
    credit_utilization_ratio = row['Credit Utilization Ratio']
    number_of_credit_accounts = row['Number of Credit Accounts']
    education_level = row['Education Level']
    employment_status = row['Employment Status']

    # Apply the FICO formula to calculate the credit score
    credit_score = (payment_history * 0.35) + (credit_utilization_ratio * 0.30) + (number_of_credit_accounts * 0.15) + (education_level * 0.10) + (employment_status * 0.10)
    credit_scores.append(credit_score)

# Add the credit scores as a new column to the DataFrame
data['Credit Score'] = credit_scores

print(data.head())

from sklearn.cluster import KMeans


# Classification based on Credit Score.
X = data[['Credit Score']]
Kmeans = KMeans(n_clusters =5, random_state = 42)
Kmeans.fit(X)
data['Segment'] = Kmeans.labels_

# Convert the 'Segment' column to category data type
data['Segment'] = data['Segment'].astype('category')

# Visualize the segments using Plotly
fig = px.scatter(data, x=data.index, y='Credit Score', color='Segment',
                 color_discrete_sequence=['green', 'blue', 'yellow', 'red', 'orange'])
fig.update_layout(
    xaxis_title='Customer Index',
    yaxis_title='Credit Score',
    title='Customer Segmentation based on Credit Scores'
)
fig.show()

data['Segment'] = data['Segment'].map({2: 'Excellent', 4: 'Good', 0: 'Fair', 3: 'Poor', 1: 'Very Poor'})

# Convert Segment into Category datatype
data['Segment'] = data['Segment'].astype('category')

# Visualize the segments using Plotly
fig = px.scatter(data, x=data.index, y='Credit Score', color='Segment',
                 color_discrete_sequence=['green', 'blue', 'yellow', 'red', 'orange'])
fig.update_layout(
    xaxis_title='Customer Index',
    yaxis_title='Credit Score',
    title='Customer Segmentation based on Credit Scores'
)
fig.show()
