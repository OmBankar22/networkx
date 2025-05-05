import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = 'https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/master/USA_Housing.csv'
data = pd.read_csv(url)

# Explore the dataset
print(data.head())
print(data.info())

# Preprocess the data (if necessary)
# Check for missing values
print(data.isnull().sum())

# Handle categorical features (Address) using one-hot encoding
data = pd.get_dummies(data, columns=['Address'], drop_first=True)

# Define features and target variable
X = data.drop('Price', axis=1)  # Features
y = data['Price']              # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f'Root Mean Squared Error: {rmse}')
print(f'R² Score: {r2}')


# 1. Scatter plot of Actual vs. Predicted Prices
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--') # Ideal line
plt.show()

