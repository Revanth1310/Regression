# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load and preprocess the Titanic dataset
data = pd.read_csv('titanic.csv')  # Replace with your Titanic dataset path

# Check the first few rows of the dataset
print(data.head())

# Data preprocessing: Handling missing values
data = data.dropna(subset=['Age', 'Fare', 'Pclass', 'Survived'])  # Remove rows with missing values in these columns

# Feature selection for Simple Linear Regression (single feature)
X_simple = data[['Age']]  # Simple linear regression: predicting 'Survived' from 'Age'
y = data['Survived']

# Feature selection for Multiple Linear Regression (multiple features)
X_multiple = data[['Age', 'Fare', 'Pclass']]  # Multiple linear regression: predicting 'Survived' from 'Age', 'Fare', and 'Pclass'

# Step 2: Split the data into training and testing sets
X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_train_multiple, X_test_multiple, _, _ = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# Step 3: Create and fit the Simple Linear Regression model
simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)

# Step 4: Predict and evaluate Simple Linear Regression model
y_pred_simple = simple_model.predict(X_test_simple)

# Evaluate Simple Linear Regression using MAE, MSE, and R²
mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)
print("----------------------For Simple linear Regression model-------------------------")

print(f"Simple Linear Regression (Age to Survived):")
print(f'Mean Absolute Error (MAE): {mae_simple}')
print(f'Mean Squared Error (MSE): {mse_simple}')
print(f'R-squared (R²): {r2_simple}')

# Step 5: Visualize Simple Linear Regression model
plt.scatter(X_test_simple, y_test, color='blue', label='Actual data')  # Actual data points
plt.plot(X_test_simple, y_pred_simple, color='red', label='Regression line')  # Regression line
plt.xlabel('Age')
plt.ylabel('Survived')
plt.title('Simple Linear Regression: Age vs Survived')
plt.legend()
plt.show()

# Step 6: Create and fit the Multiple Linear Regression model
multiple_model = LinearRegression()
multiple_model.fit(X_train_multiple, y_train)

# Step 7: Predict and evaluate Multiple Linear Regression model
y_pred_multiple = multiple_model.predict(X_test_multiple)

# Evaluate Multiple Linear Regression using MAE, MSE, and R²
mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)
print("---------------------------For Multiple Regression model-----------------------------")

print(f"\nMultiple Linear Regression (Age, Fare, Pclass to Survived):")
print(f'Mean Absolute Error (MAE): {mae_multiple}')
print(f'Mean Squared Error (MSE): {mse_multiple}')
print(f'R-squared (R²): {r2_multiple}')

# Step 8: Interpret model coefficients for Multiple Linear Regression
print("\nMultiple Linear Regression Coefficients:")
print(f'Intercept: {multiple_model.intercept_}')
print(f'Coefficients for features Age, Fare, Pclass: {multiple_model.coef_}')
