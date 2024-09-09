# Diabetic Patients Food Type Classification

This project applies logistic regression to classify food types consumed by diabetic patients. The goal is to identify patterns in food consumption and their relationships to patient conditions.

## Overview

In this notebook, logistic regression is used to model the relationship between diabetic patients and the food types they consume. The dataset contains information on various food categories, and the task is to predict which food types are more commonly associated with diabetic conditions.

## Dataset

- The dataset used in this analysis contains features related to food consumption and diabetic conditions. You can replace this section with the specific details about the dataset you're using, or link to it if it's publicly available.

## Steps in the Analysis

1. **Data Preprocessing**
   - Loading and cleaning the data to ensure it is ready for analysis.
   - Handling missing values and any necessary feature engineering.

2. **Exploratory Data Analysis (EDA)**
   - Visualizing the data to understand the distribution of features and target variables.
   - Checking for correlations and trends within the dataset.

3. **Logistic Regression Model**
   - Applying logistic regression to classify food types and identify their association with diabetes.
   - Tuning the model and checking the performance metrics.

4. **Evaluation**
   - Assessing model performance using accuracy, precision, recall, and other relevant metrics.
   - Performing cross-validation to ensure model robustness.

## Results

- The logistic regression model achieved significant accuracy of 0.8 in classifying food types for diabetic patients.
-  Calories, total carbohydrate and total fats should be eaten less often by diabetic patients while Vitamin A, Calcium and    Fibre should be eaten more often
  


## Usage

To run the notebook:

1. Clone the repository:
   ```bash
   git clone https://github.com/sheriff18/classification

   Navigate to the directory:

   ```bash
   
   cd LogisticRegression

   Open the Jupyter Notebook and run it:

   ```bash

   jupyter notebook DiabeticPatientsFoodType.ipynb

## Requirements
Python 3.x
Jupyter Notebook
Pandas
Scikit-learn
Matplotlib
Seaborn


You can install the dependencies using the following command:

  ```bash

   pip install -r requirements.txt



