# Student Score Prediction
This project demonstrates the application of machine learning techniques to predict math scores based on various student attributes such as reading scores, writing scores, parental education level, gender, lunch type, and test preparation course participation.
![Image](https://github.com/user-attachments/assets/7b5ce9ae-546a-47eb-8cdd-32f7ff4df1e8)

# 📊 Overview
The goal of this project is to create a predictive model that can estimate a student's math score based on other personal and academic factors. This is a regression problem where the target variable is the math score.
Data Overview:
Target Variable: math score
Features:
reading score
writing score
parental level of education
gender
lunch
test preparation course
race/ethnicity
The data is read from a CSV file, and we perform data preprocessing, including missing value imputation, encoding categorical variables, and feature scaling.

# ⚙️ Model Selection
Random Forest Regressor is chosen as the base model for predicting math scores due to its ability to handle non-linear relationships and its robustness to overfitting.
GridSearchCV is used to fine-tune the hyperparameters of the RandomForestRegressor model to achieve optimal performance.

# 🔍 Model Evaluation
The model is evaluated using R² (coefficient of determination), Mean Absolute Error (MAE), and Mean Squared Error (MSE) on the test set. The GridSearchCV provides the best parameters and scoring based on cross-validation.

# 📊 Performance Discussion
The model demonstrated strong performance with an R² score of 0.89, indicating that it explains 89% of the variance in math scores.
The GridSearchCV optimization helped improve the model’s performance over the default parameters.
