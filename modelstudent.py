import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("StudentScore.xls")
target = "math score"
x = data.drop(target, axis=1)
y = data[target]

#Split data
x_train, x_test, y_train, y_test =  train_test_split(x,y,train_size=0.8, random_state=42)

education_values = ["some high school", "high school", "some college", "associate's degree","bachelor's degree", "master's degree"]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("scaler", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("scaler", OneHotEncoder())
])
preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ordinal_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nominal_features", nom_transformer, ["race/ethnicity"])
])
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
parameters = {
    "regressor__n_estimators" : [ 50, 100, 200, 500],
    "regressor__criterion": ["squared_error", "absolute_error", "poisson"],
    # "regressor__max_depth": [None, 5, 10],
    # "regressor__max_features": ["sqrt", "log2"]
}
model = GridSearchCV(reg, param_grid=parameters, scoring="r2", cv=6, verbose=2, n_jobs=8)
model.fit(x_train, y_train)
print(model.best_score_)
print(model.best_params_)
# print("MAE {}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE {}".format(mean_squared_error(y_test, y_predict)))
# print("R2 {}".format(r2_score(y_test, y_predict)))
# for i,j in zip(y_test, y_predict):
#     print(f"Actual {i}, Predict {j}")



#check data sau khi transform
# result = nom_transformer.fit_transform(x_train[['race/ethnicity']])
# for i,j in zip(x_train['race/ethnicity'], result):
#     print(f"before {i} after {j}")
"""
xử lý dữ liệu bị thiếu
imputer = SimpleImputer(strategy='mean')
x= imputer.fit_transform(x[["cột muốn xử lý dữ liệu thiếu"]])
"""

"""
lazypredict để dự đoán nhiều mô hình( nhưng dùng pramaters mặc định)"""

# regressor