import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cars_data = pd.read_csv('cardekho.csv')
cars_data['Car_Age'] = 2026 - cars_data['year']
cars_data['selling_price'] = cars_data['selling_price'] / 100000  # Convert price to lakhs

#What does the data look like?
print(cars_data.head())
print(cars_data.info())
print(cars_data.describe())
print(cars_data.isnull().sum())

#How is the price distributed?
sns.histplot(cars_data['selling_price'],bins=30, kde=True)
plt.title('Distribution of Car Prices(lakhs)')
plt.xlabel('Price (lakhs)')
plt.ylabel('Frequency')
plt.show()

#which numeric columns are related to price?
nums = cars_data[[ 'selling_price', 'km_driven', 'Car_Age']]
sns.heatmap(nums.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Age of the car vs Price
sns.scatterplot(x='Car_Age', y='selling_price', data=cars_data)
plt.title('Car Age vs Selling Price')
plt.xlabel('Car Age (years)')
plt.ylabel('Selling Price (lakhs)')
plt.show()

#category effect on price
sns.boxplot(x='fuel', y='selling_price', data=cars_data)
plt.title('Fuel Type vs Selling Price')
plt.xlabel('Fuel Type')
plt.ylabel('Selling Price (lakhs)')
plt.show()

#Features
X = cars_data[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'Car_Age']]
y = cars_data['selling_price']

print("Features:")
print(X.head())
print("\ntarget:")
print(y.head())

#unique values in categorical columns
print("Unique values in categorical columns:")
print(cars_data['fuel'].unique())
print(cars_data['seller_type'].unique())
print(cars_data['transmission'].unique())
print(cars_data['owner'].unique())

#One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
print("Features after one-hot encoding:")
print(X_encoded.head())

X = X_encoded.drop('year', axis=1)  # Drop 'year' as we have 'Car_Age' now
y = cars_data['selling_price']
print("Final Features:")
print(X.head())

print(X.shape)
print(y.shape)

#splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("Training target size:", y_train.shape)
print("Test target size:", y_test.shape)

#Training a Linear Regression model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()

#training the model
linear_model.fit(X_train, y_train)
print("Linear Regression model trained.")

#Making predictions
y_pred = linear_model.predict(X_test)
print("Predictions made.")
print(y_pred[:10])  # Print first 10 predictions

#Evaluating the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

#Training a Random Forest Regressor model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest Regressor model trained.")

#Making predictions with the Random Forest model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest predictions made.")

#Evaluating the Random Forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor Model Evaluation:")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R²:", r2_rf)

#Which features are important in the Random Forest model?
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances from Random Forest Model:")
print(feature_importances)

import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, 'model/random_forest_car_price_model.pkl')
joblib.dump(X.columns.tolist(), 'model/feature_columns.pkl')
