# Credit_Score_Segmentation_Using_KNN
Unsupervised customer segmentation using K-Means clustering on engineered credit scores. Includes feature encoding, FICO-inspired scoring logic, and interactive visualizations with Plotly.

## Credit Score Segmentation using K-Means Clustering
This project presents an end-to-end pipeline for segmenting individuals based on their credit scores through the application of K-Means clustering, an unsupervised machine learning technique. 

The objective is to group customers into meaningful categories based on a credit scoring model, which can be leveraged for personalized financial services, credit risk analysis, and decision support in lending or marketing strategies.

## Project Overview
The analysis begins with data preprocessing and exploratory data analysis (EDA) on a credit scoring dataset. 

Several key financial indicators, including payment history, credit utilization ratio, number of credit accounts, education level, and employment status, are used to construct a synthetic credit score inspired by the FICO scoring model.

Once the credit score is calculated, the project applies K-Means clustering to group customers into distinct segments. These segments are then labeled and visualized to provide intuitive insight into the creditworthiness of each group.

## Key Steps and Features
**Data Import and Exploration:** Utilizes pandas for reading and exploring the dataset, including summary statistics and missing value checks.

****Feature Engineering**:** Maps categorical variables (e.g., education level, employment status) to numerical representations suitable for modeling.

**Custom Credit Score Calculation:** Implements a weighted scoring approach reflecting the principles of the FICO score, combining multiple financial attributes into a single metric.

**K-Means Clustering: **Applies the K-Means algorithm from scikit-learn to identify five customer segments based on credit scores.

**Segment Interpretation:**Clusters are labeled into interpretable categories: Excellent, Good, Fair, Poor, and Very Poor.

**Data Visualization:** Leverages plotly for interactive and publication-quality plots, including histograms, box plots, scatter plots, and correlation heatmaps.

**Category Mapping and Insights:** Enhances interpretability by converting cluster labels into human-readable segments and visualizing their distribution.

## Tools and Technologies
Python

Pandas

Plotly

scikit-learn

## Use Cases
**This project is applicable in several domains, such as:**

Financial Services: Enhancing credit risk models and tailoring credit offerings.

Customer Analytics: Profiling customers based on financial behavior.

Educational Purposes: Demonstrating the use of unsupervised learning techniques in real-world scenarios.
