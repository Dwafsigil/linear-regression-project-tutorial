import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Check and remove duplicates
total_data = pd.read_csv("medical_insurance_cost.csv")
# print(total_data.head())
# print(total_data.info()) 

total_data = total_data.drop_duplicates().reset_index(drop = True)
# print(total_data.head())

total_data["sex_num"] = pd.factorize(total_data["sex"])[0]
total_data["smoker_num"] = pd.factorize(total_data["smoker"])[0]
total_data["region_num"] = pd.factorize(total_data["region"])[0]

num_variables = ["age", "bmi", "children", "sex_num", "smoker_num", "region_num"]

scaler = MinMaxScaler()
# Transform to 0-1 scale
scale_data = scaler.fit_transform(total_data[num_variables])

total_data_scaled = pd.DataFrame(scale_data, index=total_data.index, columns = num_variables)
total_data_scaled["charges"] = total_data["charges"]

# print(total_data_scaled.head())

# Selecting Features
#  Remove charges column
x = total_data_scaled.drop("charges", axis = 1)
y = total_data_scaled["charges"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Checks features for best connection with output (charges). Taking 4 features only
select_best = SelectKBest(f_regression, k = 4)
select_best.fit(x_train, y_train)

# Returns top 4 collumns withs strongest correlations
selected_columns = x_train.columns[select_best.get_support()]

# Trims the training and test data
x_train_selected = pd.DataFrame(select_best.transform(x_train), columns = selected_columns)
x_test_selected = pd.DataFrame(select_best.transform(x_test), columns = selected_columns)

# print(x_train_selected.head())
# print(x_test_selected.head())

# Adding charges back in
x_train_selected["charges"] = y_train.values
x_test_selected["charges"] = y_test.values

# Create csv
# x_train_selected.to_csv("clean_train.csv", index = False)
# x_test_selected.to_csv("clean_test.csv", index= False)

# Linear Regression Model
test_data = pd.read_csv("clean_test.csv")
train_data = pd.read_csv("clean_train.csv")

# print(train_data.head())

fig, axis = plt.subplots(4, 2, figsize = (12, 16))
total_data = pd.concat([train_data, test_data])

sns.regplot(data = total_data, x = "age", y = "charges", ax = axis[0, 0])
sns.heatmap(total_data[["charges", "age"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(data = total_data, x = "bmi", y = "charges", ax = axis[0,1])
sns.heatmap(total_data[["charges","bmi"]].corr(), annot = True, fmt = ".2f", ax = axis[1,1], cbar=False)

sns.regplot(data = total_data, x = "children", y = "charges", ax = axis[2,0])
sns.heatmap(total_data[["charges", "children"]].corr(), annot = True, fmt = ".2f", ax = axis[3,0], cbar = False)

sns.regplot(data = total_data, x = "smoker_num", y = "charges", ax = axis[2, 1])
sns.heatmap(total_data[["charges", "smoker_num"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 1], cbar = False)

plt.tight_layout()

# plt.show()

# Remove and set charges
x_train = train_data.drop(["charges"], axis = 1)
y_train = train_data["charges"]
x_test = test_data.drop(["charges"], axis=1)
y_test = test_data["charges"]

# Create an empty linear model and create best fit off training data
model = LinearRegression()
model.fit(x_train, y_train)

# Base prediction value where all features = 0
# print(f"Intercep (a): {model.intercept_}")

# How much the target variables changes based on a features increase by 1.
# print(f"Coefficients (b1, b2): {model.coef_}")

# Predicting target values (charges) based off x features
predict_y = model.predict(x_test)
# print(predict_y)

# Check how well the model predicted the data

# The squared value in dollars of how much the guess may vary
print(f"MSE VALUE: {mean_squared_error(y_test, predict_y)}")
# Basically means the linear model can predict about 75% of the variability in medical charges
print(f"R2 SCORE: {r2_score(y_test, predict_y)}")
print(f"Average Difference in Dollars compared to actual value {round(np.sqrt(31707445.16331572),2)}$")