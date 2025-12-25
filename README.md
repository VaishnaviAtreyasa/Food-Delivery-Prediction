# Food Delivery: Predictor & Classifier
## What does this project do?
This project helps food delivery companies understand how long a delivery will take and whether it will be "On-Time" or "Delayed." It uses two different types of AI models to give both a specific time and a simple category.

## Tools Used
Python: The main coding language.

Pandas & NumPy: For cleaning and organizing the delivery data.

Seaborn & Matplotlib: To create charts that show patterns in delivery times.

Scikit-Learn: To build the AI models.

## 🛠️ Technical Highlights
Map Calculations: I turned location data (Latitude/Longitude) into usable numbers to see how distance affects speed.

Cleaning Data: I filled in missing info and changed words (like "Rainy" or "Traffic") into numbers that the computer can understand.

Fair Comparison: I used "Scaling" to make sure all numbers (like distance vs. time) were treated equally by the AI.

## How the AI Performed
The Timer (Regression): Predicts the exact number of minutes a delivery will take.

The Judge (Classification): Decides if a delivery is "Fast" or "Delayed" compared to the average.