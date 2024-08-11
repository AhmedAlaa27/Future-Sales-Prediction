# Sales Prediction with Simple Linear Regression

## Overview

This project demonstrates a simple linear regression model used to predict sales based on advertising data. The dataset includes expenditures on TV, Radio, and Newspaper advertising, and the goal is to predict the impact of these expenditures on sales. The model achieves an accuracy of 90%, showcasing the effectiveness of linear regression in handling such predictive tasks.

## Project Structure

- **`main.py`**: The main script where the data is processed, the model is trained, and predictions are made.
- **`GD.py`**: This script is using `SGDRegressor`.
- **`How to Make Simple Linear Regerssion.ipynb`**: Here I used jupyter to run the script.
- **`Requirements/advertising.csv`**: The dataset used for training and testing the model.
- **`README.md`**: Project documentation.

## Dependencies

The following Python libraries are required to run this project:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Dataset
The dataset, advertising.csv, contains the following columns:

- TV: Advertising dollars spent on TV.
- Radio: Advertising dollars spent on Radio.
- Newspaper: Advertising dollars spent on Newspapers.
- Sales: Sales in thousands of units.

## Model Description
The linear regression model was trained using the following steps:

1. Data Loading: The dataset is loaded into a pandas DataFrame.
2. Correlation Analysis: The correlation between features and the target variable (Sales) is analyzed.
3. Data Preprocessing:
  - The features (TV, Radio, Newspaper) are normalized using StandardScaler.
  - The data is split into training and testing sets using train_test_split.
4. Model Training: A simple linear regression model is trained on the normalized data.
5. Evaluation: The model's performance is evaluated based on its accuracy score.

## Results
- Accuracy: The model achieved an accuracy of 90% on the training data.
- Prediction Example: Given the advertising spend on TV, Radio, and Newspaper, the model can predict sales effectively.
