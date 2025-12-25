

# Food Delivery Time Prediction & Classification
## Project Overview
This project is a comprehensive Data Science pipeline designed to analyze and predict food delivery logistics. It goes beyond simple prediction by implementing both Regression (to estimate delivery time) and Classification (to categorize deliveries as 'Fast' or 'Delayed').

## Tech Stack
Language: Python
Librari
## Engineering Highlights
Feature Engineering: Extracted Latitude and Longitude coordinates from raw location strings to calculate spatial relationships.
Data Preprocessing: Handled missing values using mean imputation and performed One-Hot Encoding for categorical variables like Weather and Traffic.
Standard Scaling: Implemented StandardScaler to normalize numeric features, ensuring optimal model convergence.
EDA (Exploratory Data Analysis): Generated correlation heatmaps to identify key drivers of delivery delays.

## Model Performance
1. Linear Regression (Predictive)
Used to predict the exact Delivery_Time in minutes.
Metrics: Tracked Mean Squared Error (MSE) and R-squared (R²) scores.

2. Logistic Regression (Diagnostic)
Used to classify deliveries based on the median delivery time.
Evaluation: Optimized for Precision, Recall, and F1-Score.
Visualization: Generated an ROC Curve to measure the true positive rate vs. false positive rate.

## Key Insights
Traffic Impact: Traffic conditions were the highest predictor of "Delayed" status.
Logistics Optimization: Identified that vehicle selection based on weather patterns can significantly reduce delivery variance.